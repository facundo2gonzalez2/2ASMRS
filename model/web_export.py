"""Export de decoders VAE para el sintetizador web (web/).

El decoder interpolado (`InterpolatedDecoder`) recibe dos state_dicts en el init y hace la interpolación de pesos
(SLERP por tensor, igual que `experiments.interpolate.interpolar_vae`) dentro del forward: forward(z, alpha).

Para el browser se exporta UN grafo ONNX por arquitectura, con los pesos de A y B como datos externos en dos archivos
("A.bin" y "B.bin"). Cada checkpoint se exporta como un único `.bin` con el mismo layout, así el browser puede
armar cualquier par de checkpoints mapeando sus `.bin` a "A.bin"/"B.bin" al crear la sesión (`externalData` de
ONNX Runtime Web). Los ángulos SLERP solo dependen de los pesos: ONNX Runtime los pliega como constantes al crear
la sesión, por frame solo se calcula lo que depende de α.

Layout de `web/public/models/`:
    manifest.json             lista de arquitecturas y checkpoints (lo lee la UI)
    graphs/{arch_id}.onnx     grafo interpolado por arquitectura (sin pesos)
    weights/{model_id}.bin    pesos del decoder en float32, orden de `decoder_param_keys`
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
import yaml
from onnx import TensorProto
from torch import nn

MODEL_DIR = Path(__file__).resolve().parent
WEB_MODELS_DIR = MODEL_DIR.parent / "web" / "public" / "models"
EXTERNAL_A = "A.bin"
EXTERNAL_B = "B.bin"
# Umbral de |cos θ| a partir del cual SLERP cae a LERP (mismo valor que interpolar_vae)
SLERP_LERP_THRESHOLD = 0.9995
BN_EPS = 1e-5


# --- Pesos del decoder ---


def decoder_param_keys(decoder_layers) -> list[str]:
    """Claves del state_dict del decoder (sin prefijo, sin num_batches_tracked) en orden canónico."""
    keys = []
    n_blocks = len(decoder_layers) - 1
    for i in range(n_blocks):
        lin, bn = 3 * i, 3 * i + 1
        keys += [f"layers.{lin}.weight", f"layers.{lin}.bias"]
        keys += [f"layers.{bn}.{p}" for p in ("weight", "bias", "running_mean", "running_var")]
    return keys


def decoder_state_from_checkpoint(checkpoint_path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    return {k.removeprefix("decoder."): v.float() for k, v in state.items() if k.startswith("decoder.")}


def arch_id(decoder_layers) -> str:
    return "dec_" + "-".join(str(n) for n in decoder_layers)


# --- Decoder interpolado ---


class InterpolatedDecoder(nn.Module):
    """Decoder con interpolación SLERP de pesos entre dos checkpoints dentro del forward.

    Equivale a `interpolar_vae(model_a, model_b, alpha).decoder(z)` en modo eval: cada tensor del state_dict se
    interpola como w1(α)·A + w2(α)·B. Para las capas lineales se usa W·x = w1·(A·x) + w2·(B·x), que da lo mismo
    sin materializar la matriz interpolada.
    """

    def __init__(self, state_a: dict, state_b: dict, decoder_layers):
        super().__init__()
        self.decoder_layers = tuple(decoder_layers)
        self.keys = decoder_param_keys(decoder_layers)
        for key in self.keys:
            self.register_buffer(_buffer_name("a", key), state_a[key].float().clone())
            self.register_buffer(_buffer_name("b", key), state_b[key].float().clone())

    def _pair(self, key):
        return getattr(self, _buffer_name("a", key)), getattr(self, _buffer_name("b", key))

    def _coeffs(self, key, alpha):
        a, b = self._pair(key)
        return slerp_coefficients(a, b, alpha)

    def _interp(self, key, alpha):
        a, b = self._pair(key)
        w1, w2 = slerp_coefficients(a, b, alpha)
        return w1 * a + w2 * b

    def forward(self, z, alpha):
        x = z
        n_blocks = len(self.decoder_layers) - 1
        for i in range(n_blocks):
            lin, bn = 3 * i, 3 * i + 1
            # Linear: w1·(x·Aᵀ) + w2·(x·Bᵀ) + bias interpolado
            wa, wb = self._pair(f"layers.{lin}.weight")
            w1, w2 = self._coeffs(f"layers.{lin}.weight", alpha)
            x = w1 * (x @ wa.t()) + w2 * (x @ wb.t()) + self._interp(f"layers.{lin}.bias", alpha)
            # BatchNorm1d en eval con estadísticas interpoladas
            mean = self._interp(f"layers.{bn}.running_mean", alpha)
            var = self._interp(f"layers.{bn}.running_var", alpha)
            gamma = self._interp(f"layers.{bn}.weight", alpha)
            beta = self._interp(f"layers.{bn}.bias", alpha)
            x = (x - mean) / torch.sqrt(var + BN_EPS) * gamma + beta
            if i < n_blocks - 1:
                x = torch.nn.functional.elu(x)
        return x


def _buffer_name(side: str, key: str) -> str:
    return f"{side}__{key.replace('.', '_')}"


def slerp_coefficients(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor):
    """Coeficientes (w1, w2) tales que SLERP(a, b, α) = w1·a + w2·b, con fallback a LERP como en interpolar_vae.

    θ solo depende de a y b (en ONNX se pliega como constante); w1 y w2 dependen de α.
    """
    norm_a = torch.sqrt(torch.sum(a * a))
    norm_b = torch.sqrt(torch.sum(b * b))
    degenerate = (norm_a == 0) | (norm_b == 0)
    denom = torch.where(degenerate, torch.ones_like(norm_a), norm_a * norm_b)
    dot = torch.clamp(torch.sum(a * b) / denom, -1.0, 1.0)
    use_lerp = degenerate | (torch.abs(dot) > SLERP_LERP_THRESHOLD)
    theta = torch.acos(dot)
    sin_theta = torch.where(use_lerp, torch.ones_like(theta), torch.sin(theta))
    w1 = torch.where(use_lerp, 1.0 - alpha, torch.sin((1.0 - alpha) * theta) / sin_theta)
    w2 = torch.where(use_lerp, alpha, torch.sin(alpha * theta) / sin_theta)
    return w1, w2


# --- ONNX ---


def export_arch_graph(decoder_layers, out_path: Path, state_a=None, state_b=None):
    """Exporta el grafo ONNX interpolado para una arquitectura, con los pesos A/B como datos externos.

    Las referencias externas apuntan a "A.bin"/"B.bin" con los offsets de `weights_layout`; los pesos usados en el
    export solo sirven de ejemplo (no se guardan).
    """
    keys = decoder_param_keys(decoder_layers)
    # A y B tienen que ser distintos: el exporter deduplica initializers idénticos
    if state_a is None:
        state_a = _random_decoder_state(decoder_layers, seed=0)
    if state_b is None:
        state_b = _random_decoder_state(decoder_layers, seed=1)
    module = InterpolatedDecoder(state_a, state_b, decoder_layers).eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    latent_dim = decoder_layers[0]
    tmp_path = out_path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        module,
        (torch.zeros(1, latent_dim), torch.zeros(1)),
        str(tmp_path),
        input_names=["z", "alpha"],
        output_names=["spectrum"],
        dynamic_axes={"z": {0: "frames"}, "spectrum": {0: "frames"}},
        opset_version=17,
        do_constant_folding=False,  # θ tiene que quedar en el grafo: depende del par de pesos
        dynamo=False,
    )
    model = onnx.load(str(tmp_path))
    tmp_path.unlink()

    # Mapear cada buffer a su referencia externa en A.bin / B.bin
    offsets = weights_layout(decoder_layers)
    by_name = {}
    for side, location in (("a", EXTERNAL_A), ("b", EXTERNAL_B)):
        for key in keys:
            by_name[_buffer_name(side, key)] = (location, key)

    for init in model.graph.initializer:
        if init.name not in by_name:
            continue
        location, key = by_name[init.name]
        offset, length = offsets[key]
        assert init.data_type == TensorProto.FLOAT
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
            entry = init.external_data.add()
            entry.key, entry.value = k, v

    missing = set(by_name) - {init.name for init in model.graph.initializer}
    if missing:
        raise RuntimeError(f"El export ONNX no conservó estos buffers como initializers: {sorted(missing)[:5]}")

    onnx.save(model, str(out_path))
    return out_path


def weights_layout(decoder_layers) -> dict[str, tuple[int, int]]:
    """Offset y largo en bytes de cada tensor dentro del `.bin` de un checkpoint."""
    layout = {}
    offset = 0
    for key, shape in _decoder_shapes(decoder_layers).items():
        length = int(np.prod(shape)) * 4
        layout[key] = (offset, length)
        offset += length
    return layout


def write_weights_bin(state: dict, decoder_layers, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shapes = _decoder_shapes(decoder_layers)
    with open(out_path, "wb") as f:
        for key in decoder_param_keys(decoder_layers):
            tensor = state[key].detach().cpu().float().numpy()
            if tuple(tensor.shape) != shapes[key]:
                raise ValueError(f"{key}: forma {tensor.shape}, se esperaba {shapes[key]}")
            f.write(np.ascontiguousarray(tensor, dtype="<f4").tobytes())


def _decoder_shapes(decoder_layers) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for i in range(len(decoder_layers) - 1):
        lin, bn = 3 * i, 3 * i + 1
        n_in, n_out = decoder_layers[i], decoder_layers[i + 1]
        shapes[f"layers.{lin}.weight"] = (n_out, n_in)
        shapes[f"layers.{lin}.bias"] = (n_out,)
        for p in ("weight", "bias", "running_mean", "running_var"):
            shapes[f"layers.{bn}.{p}"] = (n_out,)
    return shapes


def _random_decoder_state(decoder_layers, seed):
    gen = torch.Generator().manual_seed(seed)
    state = {}
    for key, shape in _decoder_shapes(decoder_layers).items():
        tensor = torch.randn(shape, generator=gen) * 0.1
        state[key] = tensor.abs() + 0.5 if key.endswith("running_var") else tensor
    return state


# --- Estadísticas del espacio latente ---


def latent_stats(mu: np.ndarray, X: np.ndarray | None = None) -> dict:
    """Rango de los sliders (percentiles 1-99 + 10% de margen) y Z inicial (mediana de los frames más fuertes)."""
    lo, hi = np.percentile(mu, 1, axis=0), np.percentile(mu, 99, axis=0)
    margin = 0.1 * (hi - lo)
    lo, hi = lo - margin, hi + margin
    if X is not None and len(X) == len(mu):
        energy = np.asarray(X).mean(axis=1)
        loud = energy >= np.percentile(energy, 75)
        z_init = np.median(mu[loud], axis=0)
    else:
        z_init = np.median(mu, axis=0)
    return {
        "zRange": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(lo, hi)],
        "zInit": [round(float(v), 4) for v in z_init],
    }


def default_latent_stats(latent_dim: int) -> dict:
    """Sin datos: el prior N(0, 1) del β-VAE."""
    return {"zRange": [[-3.0, 3.0]] * latent_dim, "zInit": [0.0] * latent_dim}


# --- Manifest ---


def export_checkpoint(
    checkpoint_path: Path,
    hps: dict,
    model_id: str,
    label: str,
    group: str,
    stats: dict | None = None,
    out_dir: Path = WEB_MODELS_DIR,
):
    """Exporta los pesos del decoder de un checkpoint y lo registra en el manifest (crea el grafo si hace falta)."""
    checkpoint_path = Path(checkpoint_path)
    decoder_layers = (hps["latent_dim"],) + tuple(hps["decoder_layers"])
    state = decoder_state_from_checkpoint(checkpoint_path)

    weights_path = out_dir / "weights" / f"{model_id}.bin"
    write_weights_bin(state, decoder_layers, weights_path)

    aid = arch_id(decoder_layers)
    graph_path = out_dir / "graphs" / f"{aid}.onnx"
    if not graph_path.exists():
        export_arch_graph(decoder_layers, graph_path)

    if stats is None:
        stats = default_latent_stats(hps["latent_dim"])

    entry = {
        "id": model_id,
        "label": label,
        "group": group,
        "arch": aid,
        "weights": f"weights/{model_id}.bin",
        "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16],
        "latentDim": int(hps["latent_dim"]),
        "sampleRate": int(hps["target_sampling_rate"]),
        "winLength": int(hps["win_length"]),
        "hopLength": int(hps["hop_length"]),
        "xMax": float(hps["Xmax"]),
        "dbMinNorm": float(hps["db_min_norm"]),
        "beta": float(hps.get("beta", 0.0)),
        **stats,
        "source": _relative_source(checkpoint_path),
        "exportedAt": datetime.now().isoformat(timespec="seconds"),
    }
    update_manifest(entry, decoder_layers, out_dir)
    return entry


def update_manifest(entry: dict, decoder_layers, out_dir: Path = WEB_MODELS_DIR):
    manifest_path = out_dir / "manifest.json"
    manifest = {"version": 1, "archs": {}, "models": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    aid = entry["arch"]
    manifest["archs"][aid] = {
        "graph": f"graphs/{aid}.onnx",
        "layers": list(decoder_layers),
        "externalData": [EXTERNAL_A, EXTERNAL_B],
        "weightsBytes": sum(length for _, length in weights_layout(decoder_layers).values()),
    }
    manifest["models"] = [m for m in manifest["models"] if m["id"] != entry["id"]] + [entry]
    manifest["models"].sort(key=lambda m: (m["group"], m["id"]))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def export_version_dir(version_dir: Path, stats: dict | None = None, out_dir: Path = WEB_MODELS_DIR):
    """Exporta un directorio `{log_path}/{run_name}/version_N` de Lightning (toma el checkpoint más reciente)."""
    version_dir = Path(version_dir)
    ckpts = sorted(Path(version_dir, "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No hay checkpoints en {version_dir / 'checkpoints'}")
    hps = load_hparams(version_dir)
    model_id, label, group = describe_run(version_dir, hps)
    return export_checkpoint(ckpts[-1], hps, model_id, label, group, stats=stats, out_dir=out_dir)


def describe_run(version_dir: Path, hps: dict) -> tuple[str, str, str]:
    """(id, label, grupo) a partir de `{grupo}/{run_name}/version_N`."""
    run_name = version_dir.parent.name
    group = version_dir.parent.parent.name
    version = version_dir.name.removeprefix("version_")
    model_id = run_name if version == "0" else f"{run_name}_v{version}"
    beta = float(hps.get("beta", 0.0))
    beta_str = f"β={beta:g}" if beta > 0 else "sin β"
    label = run_name.replace(f"_beta_{beta:g}", "").replace("_beta_" + str(beta), "").replace("_no_beta", "")
    label = label.replace("_from_", " · ").replace("_", " ")
    label = f"{label} · {beta_str}" + (f" · v{version}" if version != "0" else "")
    return model_id, label, group


def load_hparams(version_dir: Path) -> dict:
    with open(Path(version_dir, "hparams.yaml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODEL_DIR))
    except ValueError:
        return str(path)

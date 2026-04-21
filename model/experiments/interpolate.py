import collections
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allow running this script directly via `python experiments/interpolate.py`
# by making the parent `model/` directory importable.
MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from VariationalAutoEncoder import VariationalAutoEncoder  # noqa: E402
from audio_utils import get_spectrograms_from_audios, save_audio  # noqa: E402
from scripts.vae_predict import predict_audio  # noqa: E402


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    root_dir: str
    suffix: str


REGIMES = {
    "checkpoint_no_beta": RegimeConfig(
        name="checkpoint_no_beta",
        root_dir="inference_models/instruments_from_checkpoint",
        suffix="from_checkpoint_no_beta",
    ),
    "checkpoint_beta": RegimeConfig(
        name="checkpoint_beta",
        root_dir="inference_models/instruments_from_checkpoint",
        suffix="from_checkpoint_beta_0.001",
    ),
    "scratch_no_beta": RegimeConfig(
        name="scratch_no_beta",
        root_dir="inference_models/instruments_from_scratch",
        suffix="from_scratch_no_beta",
    ),
    "scratch_beta": RegimeConfig(
        name="scratch_beta",
        root_dir="inference_models/instruments_from_scratch",
        suffix="from_scratch_beta_0.001",
    ),
}


REGIME_DISPLAY_LABELS = {
    "checkpoint_no_beta": "Checkpoint, β=0",
    "checkpoint_beta": "Checkpoint, β=0.001",
    "scratch_no_beta": "Scratch, β=0",
    "scratch_beta": "Scratch, β=0.001",
}


FAMILY_STYLES = [
    ("source", "tab:blue"),
    ("target", "tab:orange"),
    ("witness", "tab:green"),
]


METRIC_YLABELS = {
    "cosine": "Similitud (coseno)",
    "fad": "Similitud (FAD)",
}


METRIC_DISPLAY_NAMES = {
    "cosine": "coseno",
    "fad": "FAD",
}


@dataclass
class ModelArtifact:
    instrument: str
    regime: str
    model_dir: Path
    checkpoint_path: Path
    hparams: dict[str, Any]


def interpolar_vae(
    model_a: VariationalAutoEncoder,
    model_b: VariationalAutoEncoder,
    alpha: float,
    encoder_layers,
    decoder_layers,
    latent_dim,
    interpolation_mode: str = "slerp",
) -> VariationalAutoEncoder:
    # 1. Obtener los diccionarios de estado (parámetros)
    theta_a = model_a.state_dict()
    theta_b = model_b.state_dict()

    # 2. Crear un nuevo diccionario para los pesos interpolados
    theta_interp = collections.OrderedDict()

    def _slerp_tensor(tensor_a: torch.Tensor, tensor_b: torch.Tensor, t: float):
        # SLERP está definido para vectores en una esfera; aplanamos y luego restauramos forma.
        original_dtype = tensor_a.dtype
        vec_a = tensor_a.reshape(-1)
        vec_b = tensor_b.reshape(-1)

        if not torch.is_floating_point(vec_a):
            return tensor_a if t < 0.5 else tensor_b

        vec_a_f = vec_a.float()
        vec_b_f = vec_b.float()

        norm_a = torch.norm(vec_a_f)
        norm_b = torch.norm(vec_b_f)
        if norm_a.item() == 0.0 or norm_b.item() == 0.0:
            interpolated = (1.0 - t) * vec_a_f + t * vec_b_f
            return interpolated.reshape_as(tensor_a).to(dtype=original_dtype)

        unit_a = vec_a_f / norm_a
        unit_b = vec_b_f / norm_b

        dot = torch.clamp(torch.sum(unit_a * unit_b), -1.0, 1.0)

        # Si el ángulo es muy pequeño (o casi opuesto), hacemos fallback a LERP por estabilidad.
        if torch.abs(dot) > 0.9995:
            interpolated = (1.0 - t) * vec_a_f + t * vec_b_f
        else:
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1.0 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            interpolated = w1 * vec_a_f + w2 * vec_b_f

        return interpolated.reshape_as(tensor_a).to(dtype=original_dtype)

    if interpolation_mode not in {"linear", "slerp"}:
        raise ValueError(f"interpolation_mode inválido: {interpolation_mode}. Usa 'linear' o 'slerp'.")

    # 3. Iterar sobre todos los parámetros
    for key in theta_a:
        if key in theta_b:
            # 4. Calcular la interpolación configurada (lineal o SLERP)
            if interpolation_mode == "slerp":
                theta_interp[key] = _slerp_tensor(theta_a[key], theta_b[key], alpha)
            else:
                tensor_a = theta_a[key]
                tensor_b = theta_b[key]
                if not torch.is_floating_point(tensor_a):
                    theta_interp[key] = tensor_a if alpha < 0.5 else tensor_b
                else:
                    theta_interp[key] = (1.0 - alpha) * tensor_a + alpha * tensor_b
        else:
            # Esto no debería pasar si las arquitecturas son idénticas
            raise KeyError(f"Clave '{key}' no encontrada en model_b. Las arquitecturas no coinciden.")

    # 5. Crear una nueva instancia del modelo VAE
    #    ¡Importante! No pasamos 'checkpoint_path' aquí.
    modelo_interpolado = VariationalAutoEncoder(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        latent_dim=latent_dim,
    )

    # 6. Cargar los pesos interpolados en el nuevo modelo
    modelo_interpolado.load_state_dict(theta_interp)

    return modelo_interpolado


class EmbeddingCache:
    def __init__(self):
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._matrix_embedding_cache: dict[str, np.ndarray] = {}

    def get_embedding(self, audio_path: Path) -> np.ndarray:
        key = str(audio_path)
        if key not in self._embedding_cache:
            self._embedding_cache[key] = get_embedding(key)
        return self._embedding_cache[key]

    def get_matrix_embedding(self, audio_path: Path) -> np.ndarray:
        key = str(audio_path)
        if key not in self._matrix_embedding_cache:
            self._matrix_embedding_cache[key] = get_matrix_embedding(key)
        return self._matrix_embedding_cache[key]


def cosine_similarity_from_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(emb1, emb2) / denom)


def parse_alphas(alpha_text: str) -> list[float]:
    values = []
    for part in alpha_text.split(","):
        token = part.strip()
        if not token:
            continue
        alpha = float(token)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Alpha inválido {alpha}. Debe estar en [0, 1].")
        values.append(alpha)

    if not values:
        raise ValueError("No se proveyeron alphas válidos.")

    return sorted(set(values))


def get_latest_version_dir(model_family_dir: Path) -> Path | None:
    if not model_family_dir.exists():
        return None

    version_dirs = [p for p in model_family_dir.glob("version_*") if p.is_dir()]
    if not version_dirs:
        return None

    def version_key(path: Path) -> int:
        try:
            return int(path.name.split("_")[-1])
        except ValueError:
            return -1

    return sorted(version_dirs, key=version_key)[-1]


def get_first_checkpoint(version_dir: Path) -> Path | None:
    checkpoints_dir = version_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    checkpoints = sorted(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return checkpoints[0]


def resolve_model_artifact(
    project_root: Path,
    instrument: str,
    regime: RegimeConfig,
) -> ModelArtifact | None:
    family_dir = project_root / regime.root_dir / f"{instrument}_{regime.suffix}"
    version_dir = get_latest_version_dir(family_dir)
    if version_dir is None:
        return None

    checkpoint_path = get_first_checkpoint(version_dir)
    hparams_path = version_dir / "hparams.yaml"

    if checkpoint_path is None or not hparams_path.exists():
        return None

    with open(hparams_path) as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    return ModelArtifact(
        instrument=instrument,
        regime=regime.name,
        model_dir=version_dir,
        checkpoint_path=checkpoint_path,
        hparams=hparams,
    )


def build_model(artifact: ModelArtifact) -> VariationalAutoEncoder:
    return VariationalAutoEncoder(
        encoder_layers=artifact.hparams["encoder_layers"],
        decoder_layers=artifact.hparams["decoder_layers"],
        latent_dim=artifact.hparams["latent_dim"],
        checkpoint_path=artifact.checkpoint_path,
    )


def validate_architecture(artifact_a: ModelArtifact, artifact_b: ModelArtifact):
    checks = [
        (
            "encoder_layers",
            artifact_a.hparams["encoder_layers"],
            artifact_b.hparams["encoder_layers"],
        ),
        (
            "decoder_layers",
            artifact_a.hparams["decoder_layers"],
            artifact_b.hparams["decoder_layers"],
        ),
        (
            "latent_dim",
            artifact_a.hparams["latent_dim"],
            artifact_b.hparams["latent_dim"],
        ),
    ]
    for name, left, right in checks:
        if left != right:
            raise ValueError(f"Arquitectura incompatible en {name}: {artifact_a.model_dir} vs {artifact_b.model_dir}")


def collect_audio_files(data_root: Path, instrument: str) -> list[Path]:
    instrument_dir = data_root / instrument
    if not instrument_dir.exists():
        return []

    extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    files: list[Path] = []
    for pattern in extensions:
        files.extend(instrument_dir.glob(pattern))
    files = sorted([p for p in files if p.is_file()])
    return files


def sample_indices(total_items: int, n_samples: int, rng: random.Random) -> list[int]:
    if total_items <= 0:
        return []

    indices = list(range(total_items))
    if total_items <= n_samples:
        return indices

    return sorted(rng.sample(indices, k=n_samples))


def build_pairs(instruments: list[str], anchor_instrument: str | None) -> list[tuple[str, str]]:
    if anchor_instrument and anchor_instrument not in instruments:
        raise ValueError(f"El instrumento ancla '{anchor_instrument}' no está en la lista de instrumentos.")

    if anchor_instrument:
        pairs = []
        for other in instruments:
            if other == anchor_instrument:
                continue
            pairs.append((anchor_instrument, other))
            # pairs.append((other, anchor_instrument))
        return pairs

    pairs = []
    for source in instruments:
        for target in instruments:
            if source == target:
                continue
            # pairs.append((source, target))
    return pairs


def choose_witness_instrument(
    instruments: Iterable[str],
    source_instrument: str,
    target_instrument: str,
) -> str | None:
    for instrument in instruments:
        if instrument not in {source_instrument, target_instrument}:
            return instrument
    return None


def reconstruct_audio(
    model: VariationalAutoEncoder,
    hparams: dict[str, Any],
    input_audio_path: Path,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X, phases, _, _ = get_spectrograms_from_audios(
        [input_audio_path],
        hparams["target_sampling_rate"],
        hparams["win_length"],
        hparams["hop_length"],
        db_min_norm=hparams["db_min_norm"],
        spec_in_db=hparams["spec_in_db"],
        normalize_each_audio=hparams["normalize_each_audio"],
    )

    Xmax = float(hparams.get("Xmax", 1.0))
    predicted_specgram = model.predict(X) * Xmax

    save_audio(
        predicted_specgram,
        hparams["db_min_norm"],
        phases,
        hparams["hop_length"],
        hparams["win_length"],
        hparams["target_sampling_rate"],
        str(output_path),
        hparams["spec_in_db"],
    )


def evaluate_reconstruction(
    reconstructed_path: Path,
    source_reference: Path,
    target_reference: Path,
    witness_reference: Path,
    cache: EmbeddingCache,
    include_fad: bool,
) -> dict[str, float]:
    recon_embedding = cache.get_embedding(reconstructed_path)

    source_embedding = cache.get_embedding(source_reference)
    target_embedding = cache.get_embedding(target_reference)
    witness_embedding = cache.get_embedding(witness_reference)

    cosine_source = cosine_similarity_from_embeddings(recon_embedding, source_embedding)
    cosine_target = cosine_similarity_from_embeddings(recon_embedding, target_embedding)
    cosine_witness = cosine_similarity_from_embeddings(recon_embedding, witness_embedding)

    metrics = {
        "cosine_source": cosine_source,
        "cosine_target": cosine_target,
        "cosine_witness": cosine_witness,
    }

    if include_fad:
        recon_matrix = cache.get_matrix_embedding(reconstructed_path)
        source_matrix = cache.get_matrix_embedding(source_reference)
        target_matrix = cache.get_matrix_embedding(target_reference)
        witness_matrix = cache.get_matrix_embedding(witness_reference)

        metrics["fad_source"] = float(get_audio_similarity_fad(recon_matrix, source_matrix))
        metrics["fad_target"] = float(get_audio_similarity_fad(recon_matrix, target_matrix))
        metrics["fad_witness"] = float(get_audio_similarity_fad(recon_matrix, witness_matrix))

    return metrics


def encode_audio_to_latent(
    model: VariationalAutoEncoder,
    hparams: dict[str, Any],
    audio_path: Path,
) -> torch.Tensor:
    X, _, _, _ = get_spectrograms_from_audios(
        [audio_path],
        hparams["target_sampling_rate"],
        hparams["win_length"],
        hparams["hop_length"],
        db_min_norm=hparams["db_min_norm"],
        spec_in_db=hparams["spec_in_db"],
        normalize_each_audio=hparams["normalize_each_audio"],
    )

    device = next(model.parameters()).device
    X = X.to(device)

    model.eval()
    model.encoder.eval()
    with torch.no_grad():
        mu, _ = model.encoder(X)

    return mu.mean(dim=0, keepdim=True).detach().cpu()


def generate_latent_seed(
    sampling_mode: str,
    latent_dim: int,
    latent_seed: int,
    rng: random.Random,
    source_model: VariationalAutoEncoder,
    source_hparams: dict[str, Any],
    source_ground_truth_files: list[Path],
    target_model: VariationalAutoEncoder,
    target_hparams: dict[str, Any],
    target_ground_truth_files: list[Path],
    witness_model: VariationalAutoEncoder,
    witness_hparams: dict[str, Any],
    witness_ground_truth_files: list[Path],
) -> tuple[torch.Tensor, dict[str, str]]:
    if sampling_mode == "gaussian":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(latent_seed)
        latent_vector = torch.randn((1, latent_dim), generator=generator)
        return latent_vector, {
            "latent_seed_reference_source": "",
            "latent_seed_reference_target": "",
            "latent_seed_reference_witness": "",
            "latent_seed_reference_index": "-1",
        }

    if sampling_mode != "encoded":
        raise ValueError(f"random_latent_sampling_mode inválido: {sampling_mode}. Usa 'gaussian' o 'encoded'.")

    available_count = min(
        len(source_ground_truth_files),
        len(target_ground_truth_files),
        len(witness_ground_truth_files),
    )
    if available_count <= 0:
        raise ValueError("No hay suficientes audios ground truth para construir un latente desde audios.")

    reference_index = rng.randrange(available_count)
    source_reference = source_ground_truth_files[reference_index]
    target_reference = target_ground_truth_files[reference_index]
    witness_reference = witness_ground_truth_files[reference_index]

    source_latent = encode_audio_to_latent(source_model, source_hparams, source_reference)
    target_latent = encode_audio_to_latent(target_model, target_hparams, target_reference)
    witness_latent = encode_audio_to_latent(
        witness_model,
        witness_hparams,
        witness_reference,
    )

    latent_vector = (source_latent + target_latent + witness_latent) / 3.0
    return latent_vector, {
        "latent_seed_reference_source": str(source_reference),
        "latent_seed_reference_target": str(target_reference),
        "latent_seed_reference_witness": str(witness_reference),
        "latent_seed_reference_index": str(reference_index),
    }


def decode_latent_to_audio(
    model: VariationalAutoEncoder,
    hparams: dict[str, Any],
    latent_vector: torch.Tensor,
    output_path: Path,
    num_frames: int,
    phase_option: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()
    model.decoder.eval()
    Xmax = float(hparams.get("Xmax", 1.0))

    with torch.no_grad():
        tiled_latent = latent_vector.to(device).repeat(num_frames, 1)
        predicted_specgram = model.decoder(tiled_latent) * Xmax
        predicted_specgram = torch.nan_to_num(
            predicted_specgram,
            nan=0.0,
            posinf=Xmax,
            neginf=0.0,
        )
        predicted_specgram = torch.clamp(predicted_specgram, min=0.0, max=Xmax)

    predict_audio(
        predicted_specgram=predicted_specgram.cpu(),
        hps=hparams,
        phase_option=phase_option,
        frames=num_frames,
        output_path=str(output_path),
    )


def get_row_experiment_type(row: dict[str, Any]) -> str:
    value = row.get("experiment_type")
    if value:
        return str(value)
    return "audio_reconstruction"


def get_row_latent_sampling_mode(row: dict[str, Any]) -> str:
    value = row.get("latent_sampling_mode")
    if value:
        return str(value)
    return ""


def get_experiment_plot_metadata(
    experiment_type: str,
    latent_sampling_mode: str,
) -> tuple[str, str]:
    if experiment_type == "random_latent":
        if latent_sampling_mode == "encoded":
            return " | desde latente derivado de audios", "_random_latent_encoded"
        return " | desde vector aleatorio", "_random_latent_gaussian"

    return "", ""


def read_raw_results(input_csv: Path) -> list[dict[str, Any]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de métricas crudas: {input_csv}")

    with open(input_csv, newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, Any]] = []
        float_fields = {
            "alpha",
            "cosine_source",
            "cosine_target",
            "cosine_witness",
            "fad_source",
            "fad_target",
            "fad_witness",
        }
        int_fields = {"sample_idx", "paired_audio_idx"}

        for row in reader:
            parsed_row: dict[str, Any] = dict(row)
            parsed_row.setdefault("experiment_type", "audio_reconstruction")
            parsed_row.setdefault("latent_sampling_mode", "")
            for field in float_fields:
                value = parsed_row.get(field)
                if value in (None, ""):
                    parsed_row[field] = float("nan")
                else:
                    parsed_row[field] = float(value)
            for field in int_fields:
                value = parsed_row.get(field)
                if value in (None, ""):
                    parsed_row[field] = -1
                else:
                    parsed_row[field] = int(value)
            rows.append(parsed_row)

    return rows


def plot_sample_metric_family(
    axis,
    regime_rows: list[dict[str, Any]],
    metric_key: str,
    color: str,
    label: str,
):
    grouped_samples: dict[int, list[dict[str, Any]]] = {}
    for row in regime_rows:
        grouped_samples.setdefault(row["paired_audio_idx"], []).append(row)

    first_line = True
    for sample_rows in grouped_samples.values():
        sample_rows.sort(key=lambda row: row["alpha"])
        alphas = [row["alpha"] for row in sample_rows]
        values = [row.get(metric_key, float("nan")) for row in sample_rows]
        axis.plot(
            alphas,
            values,
            color=color,
            linewidth=1.0,
            alpha=0.35,
            marker="o",
            markersize=3,
            label=label if first_line else None,
        )
        first_line = False


def run_interpolation_experiment(args):
    project_root = Path(args.project_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    data_ground_truth_root = (project_root / args.data_ground_truth).resolve()
    output_root = (project_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_root / "interpolation_raw_metrics.csv"
    summary_csv_path = output_root / "interpolation_summary_metrics.csv"
    skipped_json_path = output_root / "skipped_cases.json"

    if getattr(args, "plot_only", False):
        rows = read_raw_results(raw_csv_path)
        summary_rows = summarize_results(rows, include_fad=args.include_fad)
        write_summary_results(summary_rows, summary_csv_path)
        generate_plots(summary_rows, output_root, include_fad=args.include_fad)
        generate_sample_plots(rows, output_root, include_fad=args.include_fad)
        print(f"[DONE] Plots regenerados desde CSV: {raw_csv_path}")
        print(f"[DONE] Summary metrics: {summary_csv_path}")
        return

    instruments = [inst.strip() for inst in args.instruments.split(",") if inst.strip()]
    pairs = build_pairs(instruments, args.anchor_instrument)
    selected_regimes = [name.strip() for name in args.regimes.split(",") if name.strip()]
    alphas = parse_alphas(args.alphas)

    if not selected_regimes:
        raise ValueError("No se seleccionaron regímenes para ejecutar.")

    rng = random.Random(args.seed)
    cache = EmbeddingCache()
    rows: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, str]] = []

    for regime_name in selected_regimes:
        if regime_name not in REGIMES:
            raise ValueError(f"Régimen desconocido '{regime_name}'. Opciones: {list(REGIMES.keys())}")
        regime = REGIMES[regime_name]

        for source_instrument, target_instrument in pairs:
            witness_instrument = choose_witness_instrument(
                instruments,
                source_instrument,
                target_instrument,
            )
            if witness_instrument is None:
                skipped_cases.append(
                    {
                        "regime": regime_name,
                        "source": source_instrument,
                        "target": target_instrument,
                        "reason": "No se encontró instrumento witness.",
                    }
                )
                continue

            source_artifact = resolve_model_artifact(project_root, source_instrument, regime)
            target_artifact = resolve_model_artifact(project_root, target_instrument, regime)

            if source_artifact is None or target_artifact is None:
                skipped_cases.append(
                    {
                        "regime": regime_name,
                        "source": source_instrument,
                        "target": target_instrument,
                        "reason": "Modelo faltante (hparams/checkpoint/version).",
                    }
                )
                continue

            validate_architecture(source_artifact, target_artifact)

            print(
                f"[INFO] Ejecutando régimen={regime_name}, par={source_instrument}->{target_instrument}, witness={witness_instrument}"
            )

            model_source = build_model(source_artifact)
            model_target = build_model(target_artifact)

            source_input_files = collect_audio_files(data_root, source_instrument)
            source_ground_truth_files = collect_audio_files(data_ground_truth_root, source_instrument)
            target_ground_truth_files = collect_audio_files(data_ground_truth_root, target_instrument)
            witness_ground_truth_files = collect_audio_files(data_ground_truth_root, witness_instrument)

            available_count = min(
                len(source_input_files),
                len(source_ground_truth_files),
                len(target_ground_truth_files),
                len(witness_ground_truth_files),
            )
            pair_output_dir = output_root / regime_name / f"{source_instrument}_to_{target_instrument}"
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            selected_indices = sample_indices(
                available_count,
                args.samples_per_instrument,
                rng,
            )

            if not selected_indices:
                skipped_cases.append(
                    {
                        "experiment_type": "audio_reconstruction",
                        "regime": regime_name,
                        "source": source_instrument,
                        "target": target_instrument,
                        "reason": "No hay suficientes archivos alineables entre data_root y data_ground_truth para source/target/witness.",
                    }
                )
            else:
                for alpha in alphas:
                    interpolated_model = interpolar_vae(
                        model_source,
                        model_target,
                        alpha,
                        encoder_layers=source_artifact.hparams["encoder_layers"],
                        decoder_layers=source_artifact.hparams["decoder_layers"],
                        latent_dim=source_artifact.hparams["latent_dim"],
                    )

                    for sample_idx, file_index in enumerate(selected_indices):
                        source_audio = source_input_files[file_index]
                        source_ref = source_ground_truth_files[file_index]
                        target_ref = target_ground_truth_files[file_index]
                        witness_ref = witness_ground_truth_files[file_index]

                        reconstructed_path = (
                            pair_output_dir / f"alpha_{alpha:.2f}" / f"sample_{file_index:02d}_{source_audio.stem}.mp3"
                        )

                        reconstruct_audio(
                            interpolated_model,
                            source_artifact.hparams,
                            source_audio,
                            reconstructed_path,
                        )

                        metrics = evaluate_reconstruction(
                            reconstructed_path,
                            source_ref,
                            target_ref,
                            witness_ref,
                            cache,
                            args.include_fad,
                        )

                        rows.append(
                            {
                                "experiment_type": "audio_reconstruction",
                                "latent_sampling_mode": "",
                                "regime": regime_name,
                                "source_instrument": source_instrument,
                                "target_instrument": target_instrument,
                                "witness_instrument": witness_instrument,
                                "alpha": alpha,
                                "sample_idx": sample_idx,
                                "paired_audio_idx": file_index,
                                "source_audio": str(source_audio),
                                "source_ground_truth": str(source_ref),
                                "target_reference": str(target_ref),
                                "witness_reference": str(witness_ref),
                                "reconstructed_audio": str(reconstructed_path),
                                "latent_seed_reference_source": "",
                                "latent_seed_reference_target": "",
                                "latent_seed_reference_witness": "",
                                "latent_seed_reference_index": "",
                                **metrics,
                            }
                        )

            if args.random_latent_samples <= 0:
                continue

            witness_artifact = resolve_model_artifact(
                project_root,
                witness_instrument,
                regime,
            )
            if witness_artifact is None:
                skipped_cases.append(
                    {
                        "experiment_type": "random_latent",
                        "regime": regime_name,
                        "source": source_instrument,
                        "target": target_instrument,
                        "reason": "Modelo witness faltante para experimento desde latente.",
                    }
                )
                continue

            validate_architecture(source_artifact, witness_artifact)
            validate_architecture(target_artifact, witness_artifact)

            model_witness = build_model(witness_artifact)
            latent_output_dir = pair_output_dir / f"random_latent_{args.random_latent_sampling_mode}"
            latent_trials: list[dict[str, Any]] = []

            for latent_sample_idx in range(args.random_latent_samples):
                try:
                    latent_vector, latent_metadata = generate_latent_seed(
                        args.random_latent_sampling_mode,
                        source_artifact.hparams["latent_dim"],
                        args.seed + (1000 * latent_sample_idx) + len(latent_trials),
                        rng,
                        model_source,
                        source_artifact.hparams,
                        source_ground_truth_files,
                        model_target,
                        target_artifact.hparams,
                        target_ground_truth_files,
                        model_witness,
                        witness_artifact.hparams,
                        witness_ground_truth_files,
                    )
                except ValueError as err:
                    skipped_cases.append(
                        {
                            "experiment_type": "random_latent",
                            "regime": regime_name,
                            "source": source_instrument,
                            "target": target_instrument,
                            "reason": str(err),
                        }
                    )
                    latent_trials = []
                    break

                sample_output_dir = latent_output_dir / f"sample_{latent_sample_idx:02d}"
                source_reference_path = sample_output_dir / "source_reference.wav"
                target_reference_path = sample_output_dir / "target_reference.wav"
                witness_reference_path = sample_output_dir / "witness_reference.wav"

                decode_latent_to_audio(
                    model_source,
                    source_artifact.hparams,
                    latent_vector,
                    source_reference_path,
                    args.random_latent_num_frames,
                    args.random_latent_phase_option,
                )
                decode_latent_to_audio(
                    model_target,
                    target_artifact.hparams,
                    latent_vector,
                    target_reference_path,
                    args.random_latent_num_frames,
                    args.random_latent_phase_option,
                )
                decode_latent_to_audio(
                    model_witness,
                    witness_artifact.hparams,
                    latent_vector,
                    witness_reference_path,
                    args.random_latent_num_frames,
                    args.random_latent_phase_option,
                )

                latent_trials.append(
                    {
                        "latent_sample_idx": latent_sample_idx,
                        "latent_vector": latent_vector,
                        "source_reference": source_reference_path,
                        "target_reference": target_reference_path,
                        "witness_reference": witness_reference_path,
                        **latent_metadata,
                    }
                )

            for alpha in alphas:
                interpolated_model = interpolar_vae(
                    model_source,
                    model_target,
                    alpha,
                    encoder_layers=source_artifact.hparams["encoder_layers"],
                    decoder_layers=source_artifact.hparams["decoder_layers"],
                    latent_dim=source_artifact.hparams["latent_dim"],
                )

                for latent_trial in latent_trials:
                    reconstructed_path = (
                        latent_output_dir / f"sample_{latent_trial['latent_sample_idx']:02d}" / f"alpha_{alpha:.2f}.wav"
                    )

                    decode_latent_to_audio(
                        interpolated_model,
                        source_artifact.hparams,
                        latent_trial["latent_vector"],
                        reconstructed_path,
                        args.random_latent_num_frames,
                        args.random_latent_phase_option,
                    )

                    metrics = evaluate_reconstruction(
                        reconstructed_path,
                        latent_trial["source_reference"],
                        latent_trial["target_reference"],
                        latent_trial["witness_reference"],
                        cache,
                        args.include_fad,
                    )

                    rows.append(
                        {
                            "experiment_type": "random_latent",
                            "latent_sampling_mode": args.random_latent_sampling_mode,
                            "regime": regime_name,
                            "source_instrument": source_instrument,
                            "target_instrument": target_instrument,
                            "witness_instrument": witness_instrument,
                            "alpha": alpha,
                            "sample_idx": latent_trial["latent_sample_idx"],
                            "paired_audio_idx": latent_trial["latent_sample_idx"],
                            "source_audio": "",
                            "source_ground_truth": str(latent_trial["source_reference"]),
                            "target_reference": str(latent_trial["target_reference"]),
                            "witness_reference": str(latent_trial["witness_reference"]),
                            "reconstructed_audio": str(reconstructed_path),
                            "latent_seed_reference_source": latent_trial["latent_seed_reference_source"],
                            "latent_seed_reference_target": latent_trial["latent_seed_reference_target"],
                            "latent_seed_reference_witness": latent_trial["latent_seed_reference_witness"],
                            "latent_seed_reference_index": latent_trial["latent_seed_reference_index"],
                            **metrics,
                        }
                    )

    write_raw_results(rows, raw_csv_path)
    summary_rows = summarize_results(rows, include_fad=args.include_fad)
    write_summary_results(summary_rows, summary_csv_path)

    with open(skipped_json_path, "w") as file:
        json.dump(skipped_cases, file, indent=2)

    generate_plots(summary_rows, output_root, include_fad=args.include_fad)
    generate_sample_plots(rows, output_root, include_fad=args.include_fad)

    print(f"[DONE] Raw metrics: {raw_csv_path}")
    print(f"[DONE] Summary metrics: {summary_csv_path}")
    print(f"[DONE] Skipped cases: {skipped_json_path}")


def write_raw_results(rows: list[dict[str, Any]], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(output_csv, "w") as file:
            file.write("")
        return

    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(
    rows: list[dict[str, Any]],
    include_fad: bool,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    metrics = ["cosine_source", "cosine_target", "cosine_witness"]
    if include_fad:
        metrics.extend(["fad_source", "fad_target", "fad_witness"])

    grouped: dict[tuple, dict[str, list[float]]] = {}

    for row in rows:
        key = (
            get_row_experiment_type(row),
            get_row_latent_sampling_mode(row),
            row["regime"],
            row["source_instrument"],
            row["target_instrument"],
            row["witness_instrument"],
            row["alpha"],
        )
        grouped.setdefault(key, {metric: [] for metric in metrics})
        for metric in metrics:
            value = row.get(metric)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            grouped[key][metric].append(float(value))

    summary_rows = []
    for key, metric_values in grouped.items():
        (
            experiment_type,
            latent_sampling_mode,
            regime,
            source,
            target,
            witness,
            alpha,
        ) = key
        summary = {
            "experiment_type": experiment_type,
            "latent_sampling_mode": latent_sampling_mode,
            "regime": regime,
            "source_instrument": source,
            "target_instrument": target,
            "witness_instrument": witness,
            "alpha": alpha,
        }
        for metric, values in metric_values.items():
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_count"] = len(values)
            else:
                summary[f"{metric}_mean"] = float("nan")
                summary[f"{metric}_std"] = float("nan")
                summary[f"{metric}_count"] = 0

        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda x: (
            x.get("experiment_type", "audio_reconstruction"),
            x.get("latent_sampling_mode", ""),
            x["regime"],
            x["source_instrument"],
            x["target_instrument"],
            x["alpha"],
        )
    )
    return summary_rows


def write_summary_results(rows: list[dict[str, Any]], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(output_csv, "w") as file:
            file.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_plots(
    summary_rows: list[dict[str, Any]],
    output_root: Path,
    include_fad: bool,
):
    if not summary_rows:
        print("[WARN] No hay filas de resumen para plotear.")
        return

    metrics = ["cosine"]
    if include_fad:
        metrics.append("fad")

    pair_keys = sorted(
        {
            (
                get_row_experiment_type(row),
                get_row_latent_sampling_mode(row),
                row["source_instrument"],
                row["target_instrument"],
            )
            for row in summary_rows
        }
    )

    for (
        experiment_type,
        latent_sampling_mode,
        source_instrument,
        target_instrument,
    ) in pair_keys:
        pair_rows = [
            row
            for row in summary_rows
            if get_row_experiment_type(row) == experiment_type
            and get_row_latent_sampling_mode(row) == latent_sampling_mode
            if row["source_instrument"] == source_instrument and row["target_instrument"] == target_instrument
        ]
        regimes = sorted({row["regime"] for row in pair_rows})
        title_suffix, file_suffix = get_experiment_plot_metadata(
            experiment_type,
            latent_sampling_mode,
        )

        for metric in metrics:
            n_plots = len(regimes)
            ncols = 2
            nrows = math.ceil(n_plots / ncols)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(12, max(4, 4 * nrows)),
                squeeze=False,
            )

            for idx, regime in enumerate(regimes):
                axis = axes[idx // ncols][idx % ncols]
                regime_rows = [row for row in pair_rows if row["regime"] == regime]
                regime_rows.sort(key=lambda x: x["alpha"])

                alphas = np.asarray([row["alpha"] for row in regime_rows], dtype=float)
                witness_instrument = regime_rows[0]["witness_instrument"] if regime_rows else ""
                instrument_labels = {
                    "source": source_instrument,
                    "target": target_instrument,
                    "witness": witness_instrument,
                }

                for family, color in FAMILY_STYLES:
                    means = np.asarray(
                        [row.get(f"{metric}_{family}_mean", float("nan")) for row in regime_rows],
                        dtype=float,
                    )
                    stds = np.asarray(
                        [row.get(f"{metric}_{family}_std", float("nan")) for row in regime_rows],
                        dtype=float,
                    )
                    axis.plot(
                        alphas,
                        means,
                        color=color,
                        marker="o",
                        label=instrument_labels[family],
                    )
                    axis.fill_between(
                        alphas,
                        means - stds,
                        means + stds,
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                    )

                axis.set_title(REGIME_DISPLAY_LABELS.get(regime, regime))
                axis.set_xlabel("α")
                axis.set_ylabel(METRIC_YLABELS.get(metric, metric))
                axis.grid(True, alpha=0.3)
                axis.legend()

            for idx in range(n_plots, nrows * ncols):
                axis = axes[idx // ncols][idx % ncols]
                axis.axis("off")

            metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)
            fig.suptitle(
                f"Similitud ({metric_display}) | {source_instrument} → {target_instrument}",
                fontsize=13,
            )
            fig.tight_layout()

            plot_path = output_root / "plots" / f"{metric}{file_suffix}_{source_instrument}_to_{target_instrument}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path)
            plt.close(fig)


def generate_sample_plots(
    rows: list[dict[str, Any]],
    output_root: Path,
    include_fad: bool,
):
    if not rows:
        print("[WARN] No hay filas crudas para plotear por muestra.")
        return

    metrics = ["cosine"]
    if include_fad:
        metrics.append("fad")

    pair_keys = sorted(
        {
            (
                get_row_experiment_type(row),
                get_row_latent_sampling_mode(row),
                row["source_instrument"],
                row["target_instrument"],
            )
            for row in rows
        }
    )

    for (
        experiment_type,
        latent_sampling_mode,
        source_instrument,
        target_instrument,
    ) in pair_keys:
        pair_rows = [
            row
            for row in rows
            if get_row_experiment_type(row) == experiment_type
            and get_row_latent_sampling_mode(row) == latent_sampling_mode
            if row["source_instrument"] == source_instrument and row["target_instrument"] == target_instrument
        ]
        regimes = sorted({row["regime"] for row in pair_rows})
        title_suffix, file_suffix = get_experiment_plot_metadata(
            experiment_type,
            latent_sampling_mode,
        )

        for metric in metrics:
            n_plots = len(regimes)
            ncols = 2
            nrows = math.ceil(n_plots / ncols)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(12, max(4, 4 * nrows)),
                squeeze=False,
            )

            for idx, regime in enumerate(regimes):
                axis = axes[idx // ncols][idx % ncols]
                regime_rows = [row for row in pair_rows if row["regime"] == regime]
                witness_instrument = regime_rows[0]["witness_instrument"] if regime_rows else ""
                instrument_labels = {
                    "source": source_instrument,
                    "target": target_instrument,
                    "witness": witness_instrument,
                }

                for family_name, color in FAMILY_STYLES:
                    plot_sample_metric_family(
                        axis,
                        regime_rows,
                        f"{metric}_{family_name}",
                        color,
                        instrument_labels[family_name],
                    )

                axis.set_title(REGIME_DISPLAY_LABELS.get(regime, regime))
                axis.set_xlabel("α")
                axis.set_ylabel(METRIC_YLABELS.get(metric, metric))
                axis.grid(True, alpha=0.3)
                axis.legend()

            for idx in range(n_plots, nrows * ncols):
                axis = axes[idx // ncols][idx % ncols]
                axis.axis("off")

            metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)
            fig.suptitle(
                f"Similitud ({metric_display}) por muestra | {source_instrument} → {target_instrument}",
                fontsize=13,
            )
            fig.tight_layout()

            plot_path = (
                output_root / "plots" / f"{metric}_samples{file_suffix}_{source_instrument}_to_{target_instrument}.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path)
            plt.close(fig)


def main():
    default_output_dir = "interpolation_random_25_samples_ckpt_vs_scratch_gaussian_latest2"
    config = {
        "project_root": ".",
        "data_root": "data_test",
        "data_ground_truth": "data_test_gt",
        "output_dir": os.environ.get("OUTPUT_DIR", default_output_dir),
        "instruments": "piano,guitar,voice",
        "anchor_instrument": "piano",
        "regimes": "checkpoint_beta,checkpoint_no_beta,scratch_beta,scratch_no_beta",
        "alphas": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        "samples_per_instrument": 0,
        "seed": 42,
        "include_fad": True,
        "plot_only": True,
        "random_latent_samples": 25,
        "random_latent_sampling_mode": "gaussian",
        "random_latent_num_frames": 48,
        "random_latent_phase_option": "random",
    }

    if config["anchor_instrument"] is not None and str(config["anchor_instrument"]).strip() == "":
        config["anchor_instrument"] = None

    args = SimpleNamespace(**config)

    run_interpolation_experiment(args)


if __name__ == "__main__":
    from audio_comparator import (  # noqa: E402
        get_audio_similarity_fad,
        get_embedding,
        get_matrix_embedding,
    )

    main()

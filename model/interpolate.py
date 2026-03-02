import collections
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml

from VariationalAutoEncoder import VariationalAutoEncoder
from audio_comparator import (
    get_audio_similarity_fad,
    get_embedding,
    get_matrix_embedding,
)
from audio_utils import get_spectrograms_from_audios, save_audio


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    root_dir: str
    suffix: str


REGIMES = {
    "checkpoint_no_beta": RegimeConfig(
        name="checkpoint_no_beta",
        root_dir="instruments_from_checkpoint",
        suffix="from_checkpoint_no_beta",
    ),
    "checkpoint_beta": RegimeConfig(
        name="checkpoint_beta",
        root_dir="instruments_from_checkpoint",
        suffix="from_checkpoint_beta",
    ),
    "scratch_no_beta": RegimeConfig(
        name="scratch_no_beta",
        root_dir="instruments_from_scratch",
        suffix="from_scratch_no_beta",
    ),
    "scratch_beta": RegimeConfig(
        name="scratch_beta",
        root_dir="instruments_from_scratch",
        suffix="from_scratch_beta",
    ),
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
) -> VariationalAutoEncoder:
    theta_a = model_a.state_dict()
    theta_b = model_b.state_dict()

    theta_interp = collections.OrderedDict()
    for key in theta_a:
        if key not in theta_b:
            raise KeyError(
                f"Clave '{key}' no encontrada en model_b. Las arquitecturas no coinciden."
            )
        theta_interp[key] = (1.0 - alpha) * theta_a[key] + alpha * theta_b[key]

    modelo_interpolado = VariationalAutoEncoder(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        latent_dim=latent_dim,
    )
    modelo_interpolado.load_state_dict(theta_interp)
    modelo_interpolado.eval()
    modelo_interpolado.encoder.eval()
    modelo_interpolado.decoder.eval()
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
            raise ValueError(
                f"Arquitectura incompatible en {name}: {artifact_a.model_dir} vs {artifact_b.model_dir}"
            )


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


def sample_files(files: list[Path], n_samples: int, rng: random.Random) -> list[Path]:
    if not files:
        return []
    if len(files) <= n_samples:
        return files
    return sorted(rng.sample(files, k=n_samples))


def build_pairs(
    instruments: list[str], anchor_instrument: str | None
) -> list[tuple[str, str]]:
    if anchor_instrument and anchor_instrument not in instruments:
        raise ValueError(
            f"El instrumento ancla '{anchor_instrument}' no está en la lista de instrumentos."
        )

    if anchor_instrument:
        pairs = []
        for other in instruments:
            if other == anchor_instrument:
                continue
            pairs.append((anchor_instrument, other))
            pairs.append((other, anchor_instrument))
        return pairs

    pairs = []
    for source in instruments:
        for target in instruments:
            if source == target:
                continue
            pairs.append((source, target))
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
    cosine_witness = cosine_similarity_from_embeddings(
        recon_embedding, witness_embedding
    )

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

        metrics["fad_source"] = float(
            get_audio_similarity_fad(recon_matrix, source_matrix)
        )
        metrics["fad_target"] = float(
            get_audio_similarity_fad(recon_matrix, target_matrix)
        )
        metrics["fad_witness"] = float(
            get_audio_similarity_fad(recon_matrix, witness_matrix)
        )

    return metrics


def run_interpolation_experiment(args):
    project_root = Path(args.project_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    output_root = (project_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    instruments = [inst.strip() for inst in args.instruments.split(",") if inst.strip()]
    pairs = build_pairs(instruments, args.anchor_instrument)
    selected_regimes = [
        name.strip() for name in args.regimes.split(",") if name.strip()
    ]
    alphas = parse_alphas(args.alphas)

    if not selected_regimes:
        raise ValueError("No se seleccionaron regímenes para ejecutar.")

    rng = random.Random(args.seed)
    cache = EmbeddingCache()
    rows: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, str]] = []

    for regime_name in selected_regimes:
        if regime_name not in REGIMES:
            raise ValueError(
                f"Régimen desconocido '{regime_name}'. Opciones: {list(REGIMES.keys())}"
            )
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

            source_artifact = resolve_model_artifact(
                project_root, source_instrument, regime
            )
            target_artifact = resolve_model_artifact(
                project_root, target_instrument, regime
            )

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

            source_files = sample_files(
                collect_audio_files(data_root, source_instrument),
                args.samples_per_instrument,
                rng,
            )
            target_files = sample_files(
                collect_audio_files(data_root, target_instrument),
                args.samples_per_instrument,
                rng,
            )
            witness_files = sample_files(
                collect_audio_files(data_root, witness_instrument),
                args.samples_per_instrument,
                rng,
            )

            if not source_files or not target_files or not witness_files:
                skipped_cases.append(
                    {
                        "regime": regime_name,
                        "source": source_instrument,
                        "target": target_instrument,
                        "reason": "No hay suficientes archivos de audio para source/target/witness.",
                    }
                )
                continue

            pair_output_dir = (
                output_root
                / regime_name
                / f"{source_instrument}_to_{target_instrument}"
            )
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            for alpha in alphas:
                interpolated_model = interpolar_vae(
                    model_source,
                    model_target,
                    alpha,
                    encoder_layers=source_artifact.hparams["encoder_layers"],
                    decoder_layers=source_artifact.hparams["decoder_layers"],
                    latent_dim=source_artifact.hparams["latent_dim"],
                )

                for idx, source_audio in enumerate(source_files):
                    target_ref = target_files[idx % len(target_files)]
                    witness_ref = witness_files[idx % len(witness_files)]

                    reconstructed_path = (
                        pair_output_dir
                        / f"alpha_{alpha:.2f}"
                        / f"sample_{idx:02d}_{source_audio.stem}.mp3"
                    )

                    reconstruct_audio(
                        interpolated_model,
                        source_artifact.hparams,
                        source_audio,
                        reconstructed_path,
                    )

                    metrics = evaluate_reconstruction(
                        reconstructed_path,
                        source_audio,
                        target_ref,
                        witness_ref,
                        cache,
                        args.include_fad,
                    )

                    rows.append(
                        {
                            "regime": regime_name,
                            "source_instrument": source_instrument,
                            "target_instrument": target_instrument,
                            "witness_instrument": witness_instrument,
                            "alpha": alpha,
                            "sample_idx": idx,
                            "source_audio": str(source_audio),
                            "target_reference": str(target_ref),
                            "witness_reference": str(witness_ref),
                            "reconstructed_audio": str(reconstructed_path),
                            **metrics,
                        }
                    )

    raw_csv_path = output_root / "interpolation_raw_metrics.csv"
    summary_csv_path = output_root / "interpolation_summary_metrics.csv"
    skipped_json_path = output_root / "skipped_cases.json"

    write_raw_results(rows, raw_csv_path)
    summary_rows = summarize_results(rows, include_fad=args.include_fad)
    write_summary_results(summary_rows, summary_csv_path)

    with open(skipped_json_path, "w") as file:
        json.dump(skipped_cases, file, indent=2)

    generate_plots(summary_rows, output_root, include_fad=args.include_fad)

    print(f"[DONE] Raw metrics: {raw_csv_path}")
    print(f"[DONE] Summary metrics: {summary_csv_path}")
    print(f"[DONE] Skipped cases: {skipped_json_path}")


def write_raw_results(rows: list[dict[str, Any]], output_csv: Path):
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
        regime, source, target, witness, alpha = key
        summary = {
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
        {(row["source_instrument"], row["target_instrument"]) for row in summary_rows}
    )

    for source_instrument, target_instrument in pair_keys:
        pair_rows = [
            row
            for row in summary_rows
            if row["source_instrument"] == source_instrument
            and row["target_instrument"] == target_instrument
        ]
        regimes = sorted({row["regime"] for row in pair_rows})

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

                alphas = [row["alpha"] for row in regime_rows]
                source_values = [
                    row.get(f"{metric}_source_mean", float("nan"))
                    for row in regime_rows
                ]
                target_values = [
                    row.get(f"{metric}_target_mean", float("nan"))
                    for row in regime_rows
                ]
                witness_values = [
                    row.get(f"{metric}_witness_mean", float("nan"))
                    for row in regime_rows
                ]

                axis.plot(alphas, source_values, marker="o", label="source")
                axis.plot(alphas, target_values, marker="o", label="target")
                axis.plot(alphas, witness_values, marker="o", label="witness")
                axis.set_title(regime)
                axis.set_xlabel("alpha")
                axis.set_ylabel(metric)
                axis.grid(True, alpha=0.3)
                axis.legend()

            for idx in range(n_plots, nrows * ncols):
                axis = axes[idx // ncols][idx % ncols]
                axis.axis("off")

            fig.suptitle(
                f"{metric.upper()} | {source_instrument} -> {target_instrument}",
                fontsize=13,
            )
            fig.tight_layout()

            plot_path = (
                output_root
                / "plots"
                / f"{metric}_{source_instrument}_to_{target_instrument}.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path)
            plt.close(fig)


def main():
    config = {
        "project_root": ".",
        "data_root": "data_instruments",
        "output_dir": "interpolation_regime_experiment",
        "instruments": "piano,guitar,voice,bass",
        "anchor_instrument": "piano",
        "regimes": "checkpoint_no_beta,checkpoint_beta,scratch_no_beta,scratch_beta",
        "alphas": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        "samples_per_instrument": 12,
        "seed": 42,
        "include_fad": True,
    }

    if (
        config["anchor_instrument"] is not None
        and str(config["anchor_instrument"]).strip() == ""
    ):
        config["anchor_instrument"] = None

    args = SimpleNamespace(**config)

    run_interpolation_experiment(args)


if __name__ == "__main__":
    main()

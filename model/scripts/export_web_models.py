"""Exporta checkpoints entrenados al sintetizador web (web/public/models/).

Recorre los directorios de logs de Lightning, exporta los pesos del decoder de cada `version_N` (un `.bin` por
checkpoint, ver web_export.py) y actualiza `manifest.json`, que es lo que lista la UI. Re-correrlo es seguro: los
checkpoints ya exportados se sobrescriben.

Los rangos de los sliders Z y el Z inicial salen del encoder sobre algunos audios de entrenamiento (si están en
DATA_DIR); si no, se usa el prior N(0, 1).

Uso (desde cualquier directorio):
    .venv/bin/python model/scripts/export_web_models.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audio_utils import get_spectrograms_from_audios  # noqa: E402
from VariationalAutoEncoder import VariationalAutoEncoder  # noqa: E402
from web_export import WEB_MODELS_DIR, export_version_dir, latent_stats, load_hparams  # noqa: E402

INSTRUMENTS = ["piano", "guitar", "voice", "bass"]
SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main():
    # --- Config ---
    # Directorios de logs a recorrer (cada uno con {run_name}/version_N/). Agregar acá los de nuevos entrenamientos.
    roots = [
        MODEL_DIR / "inference_models" / "base_model",
        MODEL_DIR / "inference_models" / "instruments_from_checkpoint",
        MODEL_DIR / "inference_models" / "instruments_from_scratch",
        # MODEL_DIR / "experiments_models" / "experiment_latent_dim_piano",
    ]
    data_dir = MODEL_DIR / "data_instruments"
    max_files_per_instrument = 3  # audios por instrumento para estimar rangos de Z (más = más lento)
    out_dir = WEB_MODELS_DIR

    version_dirs = sorted(p.parent for root in roots for p in root.glob("*/version_*/hparams.yaml"))
    print(f"{len(version_dirs)} modelos encontrados → {out_dir}")

    spectrogram_cache = {}
    for version_dir in version_dirs:
        hps = load_hparams(version_dir)
        audio_files = training_audio_files(version_dir.parent.name, data_dir, max_files_per_instrument)
        stats = None
        if audio_files:
            stats = encoder_stats(version_dir, hps, audio_files, spectrogram_cache)
        entry = export_version_dir(version_dir, stats=stats, out_dir=out_dir)
        origin = "datos" if stats else "prior N(0,1)"
        print(f"  {entry['id']:<40} zInit={entry['zInit']} (rangos: {origin})")


def training_audio_files(run_name: str, data_dir: Path, max_files: int) -> list[Path]:
    """Audios del instrumento del run (todos los instrumentos para el modelo base)."""
    instruments = [i for i in INSTRUMENTS if run_name.startswith(i)] or INSTRUMENTS
    files = []
    for instrument in instruments:
        folder = data_dir / instrument
        if folder.is_dir():
            found = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
            files += found[:max_files]
    return files


def encoder_stats(version_dir: Path, hps: dict, audio_files: list[Path], cache: dict) -> dict:
    key = (tuple(audio_files), hps["target_sampling_rate"], hps["win_length"], hps["hop_length"], hps["db_min_norm"])
    if key not in cache:
        X, _, x_max_local, _ = get_spectrograms_from_audios(
            audio_files, hps["target_sampling_rate"], hps["win_length"], hps["hop_length"], db_min_norm=hps["db_min_norm"]
        )
        cache[key] = X * x_max_local  # dB sobre db_min_norm, sin normalizar
    X = cache[key] / hps["Xmax"]  # misma normalización que en el entrenamiento

    ckpt = sorted(Path(version_dir, "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    model = VariationalAutoEncoder(hps["encoder_layers"], hps["decoder_layers"], hps["latent_dim"], checkpoint_path=ckpt)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(X.float())
    return latent_stats(mu.numpy(), X.numpy())


if __name__ == "__main__":
    main()

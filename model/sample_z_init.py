import argparse
import json
from pathlib import Path

import torch
import yaml

from audio_utils import get_spectrograms_from_audios
from VariationalAutoEncoder import VariationalAutoEncoder

INSTRUMENTS = ["piano", "guitar", "vocals", "bass"]
INSTRUMENT_DIR_NAMES = {
    "piano": "piano",
    "guitar": "guitar",
    "vocals": "voice",
    "bass": "bass",
}
SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_model(instrument_key: str, beta_config: str, training_source: str):
    dir_name = INSTRUMENT_DIR_NAMES[instrument_key]
    model_dir = Path(
        f"instruments_from_{training_source}",
        f"{dir_name}_from_{training_source}_{beta_config}",
        "version_0",
    )

    if not model_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del modelo: {model_dir}")

    ckpt_files = list(Path(model_dir, "checkpoints").glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No se encontraron checkpoints en: {model_dir / 'checkpoints'}"
        )
    checkpoint_path = ckpt_files[0]

    hparams_path = model_dir / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"No se encontró hparams.yaml en: {model_dir}")

    with open(hparams_path) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)

    model = VariationalAutoEncoder(
        encoder_layers=hps["encoder_layers"],
        decoder_layers=hps["decoder_layers"],
        latent_dim=hps["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    model.eval()

    return model, hps


def find_audio_files(data_root: Path, instrument_key: str, max_files: int | None):
    dir_name = INSTRUMENT_DIR_NAMES[instrument_key]
    instrument_dir = data_root / dir_name
    if not instrument_dir.exists():
        return []

    files = sorted(
        path
        for path in instrument_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    )

    if max_files is not None:
        files = files[:max_files]

    return files


def compute_z_stats(model, hps, audio_files, device: str):
    if not audio_files:
        raise ValueError("No se encontraron audios para este instrumento.")

    sum_mu = None
    sum_mu_sq = None
    total_frames = 0

    model.to(device)

    for audio_path in audio_files:
        X, _, _, _ = get_spectrograms_from_audios(
            [audio_path],
            hps["target_sampling_rate"],
            hps["win_length"],
            hps["hop_length"],
            db_min_norm=hps["db_min_norm"],
            spec_in_db=hps["spec_in_db"],
            normalize_each_audio=hps["normalize_each_audio"],
        )

        X = X.to(device)

        with torch.no_grad():
            mu, _ = model.encoder(X)

        if sum_mu is None:
            sum_mu = torch.zeros(mu.shape[1], device=device)
            sum_mu_sq = torch.zeros(mu.shape[1], device=device)

        sum_mu += mu.sum(dim=0)
        sum_mu_sq += (mu**2).sum(dim=0)
        total_frames += mu.shape[0]

    if total_frames == 0:
        raise ValueError("No se obtuvieron frames para calcular estadisticas.")

    mean = sum_mu / total_frames
    var = sum_mu_sq / total_frames - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    return mean.cpu(), std.cpu(), total_frames


def resolve_data_root(explicit_root: str | None):
    if explicit_root:
        return Path(explicit_root)

    for candidate in ["data_instruments_small", "data_instruments"]:
        if Path(candidate).exists():
            return Path(candidate)

    return Path("data_instruments")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza Z init por instrumento usando la distribucion del encoder."
    )
    parser.add_argument(
        "--beta-config",
        default="beta_0.001",
        choices=["beta_0.001", "no_beta"],
    )
    parser.add_argument(
        "--training-source",
        default="checkpoint",
        choices=["checkpoint", "scratch"],
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-files", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-format", default="python", choices=["python", "json"])
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)

    results = {}
    stats = {}
    latent_dim = None

    for instrument in INSTRUMENTS:
        model, hps = load_model(instrument, args.beta_config, args.training_source)
        latent_dim = hps["latent_dim"]

        audio_files = find_audio_files(data_root, instrument, args.max_files)
        if not audio_files:
            print(f"[WARN] Sin audios para {instrument} en {data_root}. Se omite.")
            continue

        mean, std, total_frames = compute_z_stats(
            model, hps, audio_files, device=args.device
        )

        results[instrument] = [round(v, 5) for v in mean.tolist()]
        stats[instrument] = {
            "std": [round(v, 5) for v in std.tolist()],
            "frames": int(total_frames),
            "files": len(audio_files),
        }

    print("Latent dim:", latent_dim)
    print("Data root:", data_root)
    print("Beta config:", args.beta_config, "Training:", args.training_source)

    for instrument in sorted(stats.keys()):
        info = stats[instrument]
        print(
            f"{instrument}: files={info['files']} frames={info['frames']} std={info['std']}"
        )

    if args.print_format == "json":
        print(json.dumps(results, indent=2))
    else:
        print("Z_INIT_BY_INSTRUMENT = {")
        for instrument, values in results.items():
            print(f'    "{instrument}": {values},')
        print("}")

    if latent_dim is not None and latent_dim != 4:
        print(
            f"[WARN] latent_dim={latent_dim}. Ajusta los sliders en app.py si es necesario."
        )


if __name__ == "__main__":
    main()

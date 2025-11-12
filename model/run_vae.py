from fire import Fire
from pathlib import Path
import pandas as pd
import torch

from audio_utils import (
    get_spectrograms_from_audios,
    save_audio,
    save_latentscore,
    save_specgram,
)
from VariationalAutoEncoder import VariationalAutoEncoder


def train(
    audio_path_list,
    run_name,
    target_sampling_rate=22050,
    hop_length_samples=512,
    win_length_samples=2048,
    encoder_layers=(1024, 512, 256, 128, 64, 32, 16, 8, 4),
    seed=None,
    db_min_norm=-60,
    spec_in_db=True,
    normalize_each_audio=False,
    validation_size=0.05,
    learning_rate=0.001,
    epochs=1000,
    batch_size=256,
    beta=0,
    latent_dim=4,
    log_path=None,
    accelerator="auto",
    checkpoint_path=None,
):
    hop_length = hop_length_samples
    win_length = win_length_samples

    print(f"hop_length: {hop_length}, win_length: {win_length}")

    X, phases, Xmax, y = get_spectrograms_from_audios(
        audio_path_list,
        target_sampling_rate,
        win_length,
        hop_length,
        db_min_norm=db_min_norm,
        spec_in_db=spec_in_db,
        normalize_each_audio=normalize_each_audio,
    )
    print(
        f"X shape: {X.shape}, phases shape: {phases.shape}, Xmax: {Xmax}, y shape: {y.shape}"
    )

    if isinstance(encoder_layers, str):
        encoder_layers = tuple(map(int, encoder_layers[1:-1].split(",")))
    elif isinstance(encoder_layers, list):
        encoder_layers = tuple(encoder_layers)

    encoder_layers = (win_length // 2 + 1,) + encoder_layers
    decoder_layers = encoder_layers[::-1][1:]

    vae = VariationalAutoEncoder(
        encoder_layers,
        decoder_layers,
        latent_dim=latent_dim,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )

    vae.load_data(X, y, Xmax, db_min_norm=db_min_norm, spec_in_db=spec_in_db)
    vae.split_data(validation_size=validation_size)

    # log all hyperparameters and paramters
    hps = {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "latent_dim": latent_dim,
        "win_length": win_length,
        "hop_length": hop_length,
        "loss": "MAE + KL",
        "beta": beta,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "spec_in_db": spec_in_db,
        "db_min_norm": db_min_norm,
        "normalize_each_audio": normalize_each_audio,
        "target_sampling_rate": target_sampling_rate,
        "Xmax": Xmax,
    }

    if log_path is None:
        log_path = "tb_logs_vae"

    vae.log_hyperparameters(**hps)
    trainer, metrics_tracker = vae.train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        beta=beta,
        run_name=run_name,
        accelerator=accelerator,
        log_path=log_path,
    )

    pd.DataFrame(metrics_tracker.collection).astype(float).to_csv(
        Path(trainer.log_dir, "metrics_history_vae.csv")  # type: ignore
    )

    vae.export_decoder()

    predicted_specgram = vae.predict(X) * Xmax

    save_audio(
        predicted_specgram,
        db_min_norm,
        phases,
        hop_length,
        win_length,
        target_sampling_rate,
        Path(trainer.log_dir, "reconstructed_audios.mp3"),  # type: ignore
        spec_in_db,
    )

    X = X[: 2 * 60 * target_sampling_rate // hop_length]
    predicted_specgram = vae.predict(X) * Xmax

    save_specgram(predicted_specgram, hop_length, trainer.log_dir)

    with torch.no_grad():
        mu, logvar = vae.encoder(X)
        Z = mu.cpu().numpy()

    if trainer.log_dir is not None:
        path = Path(trainer.log_dir, "mu_latent_score.png")
        save_latentscore(Z, hop_length, target_sampling_rate, path)

        path = Path(trainer.log_dir, "logvar_latent_score.png")
        save_latentscore(logvar.cpu().numpy(), hop_length, target_sampling_rate, path)


def main(path=None, **kwargs):
    # betas = [0.01, 0.001, 0.0001, 0.00001, 0]

    path = Path("data/piano/fur_elise_piano.mp3")
    # checkpoint_path = Path("tb_logs_vae/piano/version_0/checkpoints").glob("*.ckpt")
    # checkpoint_path = list(checkpoint_path)[0]

    if path is None:
        print("No path provided, using example")
        # Download example from url if already not exists
        path = Path("data", "Mozart25_2min.wav")
        if not path.exists():
            import requests

            path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://www.dropbox.com/s/iar9y2beo884zah/Mozart25_2min.wav?dl=1"
            r = requests.get(url, allow_redirects=True)
            path.write_bytes(r.content)

    path = Path(path)
    if path.is_file():
        audio_list = [path]
    else:
        # Load all wavfiles in directory
        audio_list = list(path.glob("*.*"))

    arqs = [
        ((1024, 512, 256, 128, 64, 32, 16, 8, 4, 2), 2),
        ((1024, 512, 256, 128, 64, 32, 16, 8, 4, 3), 3),
        ((1024, 512, 256, 128, 64, 32, 16, 8, 4), 4),
        ((1024, 512, 256, 128, 64, 32, 16, 8, 6), 6),
        ((1024, 512, 256, 128, 64, 32, 16, 8), 8),
    ]

    for encoder_layers, latent_dim in arqs:
        run_name = f"vae_latentdim_{latent_dim}"
        print("=" * 60)
        print(f"Running experiment: {run_name}")
        train(
            audio_list,
            run_name=run_name,
            encoder_layers=encoder_layers,
            latent_dim=latent_dim,
            log_path="experiment_latent_dim",
            **kwargs,
        )


if __name__ == "__main__":
    Fire(main)

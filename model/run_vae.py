from fire import Fire
from pathlib import Path
import pandas as pd
import torch
import datetime

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
    # seed=42,
    db_min_norm=-60,
    spec_in_db=True,
    normalize_each_audio=False,
    validation_size=0.05,
    learning_rate=0.001,
    epochs=2000,
    batch_size=128,
    # loss="MAE+MSE",
    beta=0.01,
    latent_dim=4,
    log_path="logs_vae",
    accelerator="cpu",
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

    vae = VariationalAutoEncoder(encoder_layers, decoder_layers, latent_dim=latent_dim)

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

    vae.log_hyperparameters(**hps)
    trainer, metrics_tracker = vae.train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        beta=beta,
        run_name=run_name,
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
        trainer.log_dir,
        spec_in_db,
    )

    X = X[: 2 * 60 * target_sampling_rate // hop_length]
    predicted_specgram = vae.predict(X) * Xmax

    save_specgram(predicted_specgram, hop_length, trainer.log_dir)

    with torch.no_grad():
        mu, logvar = vae.encoder(X)
        Z = mu.cpu().numpy()
    save_latentscore(Z, hop_length, target_sampling_rate, trainer.log_dir)


def main(path=None, **kwargs):
    # path = Path(
    #     "data_model_a/fur-elise-by-ludwig-van-beethoven-classic-guitar-ahmad-mousavipour-13870.mp3"
    # )
    path = Path("data_model_b/Bagatelle no. 25 ''Für Elise'', WoO 59.mp3")
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
    print(f"Training on {path}")
    train(audio_list, run_name="fur_elise_piano", **kwargs)


if __name__ == "__main__":
    Fire(main)

from fire import Fire
from pathlib import Path
import pandas as pd
import torch
from datetime import datetime

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
    beta: float = 0,
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


def experiment_latent_dim_instruments(kwargs):
    instruments_paths = [
        Path("data_instruments_small/piano"),
        Path("data_instruments_small/voice"),
        Path("data_instruments_small/guitar"),
        Path("data_instruments_small/bass"),
    ]
    for it in range(5):
        print(f"================= EXPERIMENT ROUND {it + 1} =================")
        now = datetime.now()
        for path in instruments_paths:
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
                run_name = f"beta_vae_latentdim_{latent_dim}"
                print("=" * 60)
                print(f"Running experiment: {run_name}")
                train(
                    audio_list,
                    run_name=run_name,
                    encoder_layers=encoder_layers,
                    latent_dim=latent_dim,
                    beta=0.001,
                    log_path=f"experiment_latent_dim_beta_{path.name}_small",
                    **kwargs,
                )
        print(f"Total duration: {datetime.now() - now}")
        print("=" * 60)
        print("\n\n")


def experiment_latent_dim_base_model(kwargs, beta=0.001):
    instruments_paths = [
        Path("data_instruments/piano"),
        Path("data_instruments/voice"),
        Path("data_instruments/guitar"),
        Path("data_instruments/bass"),
    ]

    audio_files = []
    for path in instruments_paths:
        if path.is_file():
            audio_files.append(path)
        else:
            # Load all wavfiles in directory
            audio_files.extend(list(path.glob("*.*")))

    for it in range(5):
        print(f"================= EXPERIMENT ROUND {it + 1} =================")
        now = datetime.now()
        arqs = [
            ((1024, 512, 256, 128, 64, 32, 16, 8, 4, 2), 2),
            ((1024, 512, 256, 128, 64, 32, 16, 8, 4, 3), 3),
            ((1024, 512, 256, 128, 64, 32, 16, 8, 4), 4),
            ((1024, 512, 256, 128, 64, 32, 16, 8, 6), 6),
            ((1024, 512, 256, 128, 64, 32, 16, 8), 8),
        ]

        for encoder_layers, latent_dim in arqs:
            run_name = (
                f"beta_vae_latentdim_{latent_dim}"
                if beta > 0
                else f"vae_latentdim_{latent_dim}"
            )
            print("=" * 60)
            print(f"Running experiment: {run_name}")
            train(
                audio_files,
                run_name=run_name,
                encoder_layers=encoder_layers,
                latent_dim=latent_dim,
                beta=beta,
                log_path="experiment_latent_dim_base_model",
                **kwargs,
            )
        print(f"Total duration: {datetime.now() - now}")
        print("=" * 60)
        print("\n\n")


def train_model_base(beta, **kwargs):
    instruments_paths = [
        Path("data_instruments/piano"),
        Path("data_instruments/voice"),
        Path("data_instruments/guitar"),
        Path("data_instruments/bass"),
    ]

    audio_files = []
    for path in instruments_paths:
        if path.is_file():
            audio_files.append(path)
        else:
            # Load all wavfiles in directory
            audio_files.extend(list(path.glob("*.*")))

    now = datetime.now()

    run_name = f"base_model_beta_{beta}" if beta > 0 else "base_model_no_beta"

    train(
        audio_files,
        run_name=run_name,
        encoder_layers=(1024, 512, 256, 128, 64, 32, 16, 8, 4),
        latent_dim=4,
        beta=beta,
        log_path="base_model",
        **kwargs,
    )
    print(f"Total duration for base model with beta training: {datetime.now() - now}")


def train_model_instruments(from_checkpoint=False, beta=0.0, **kwargs):
    """
    Entrenar modelo particular de cada instrumento.

    :param from_checkpoint: Si es True entrena a partir del último checkpoint guardado de base_model
    :param beta: El valor de beta para el loss KL
    :param kwargs: Description
    """
    log_path = (
        "instruments_from_checkpoint" if from_checkpoint else "instruments_from_scratch"
    )
    beta_str = "beta" if beta > 0 else "no_beta"
    ckpt_str = "from_checkpoint" if from_checkpoint else "from_scratch"

    for instrument in ["piano", "voice", "guitar", "bass"]:
        audio_path = [
            Path(f"data_instruments/{instrument}"),
        ]

        audio_files = []
        for path in audio_path:
            if path.is_file():
                audio_files.append(path)
            else:
                # Load all wavfiles in directory
                audio_files.extend(list(path.glob("*.*")))

        now = datetime.now()

        checkpoint_path = None
        if from_checkpoint:
            checkpoint_path = Path(
                f"base_model/base_model_{beta_str}/version_0/checkpoints"
            ).glob("*.ckpt")
            checkpoint_path = list(checkpoint_path)[0]

        run_name = f"{instrument}_{ckpt_str}_{beta_str}"

        train(
            audio_files,
            run_name=run_name,
            encoder_layers=(1024, 512, 256, 128, 64, 32, 16, 8, 4),
            latent_dim=4,
            beta=beta,
            log_path=log_path,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        mode_str = "with" if beta > 0 else "without"
        print(
            f"Total duration for {instrument} {ckpt_str.replace('_', ' ')} {mode_str} beta training: {datetime.now() - now}"
        )


def experiment_multiple_betas_base_model(kwargs):
    instruments_paths = [
        Path("data_instruments/piano"),
        Path("data_instruments/voice"),
        Path("data_instruments/guitar"),
        Path("data_instruments/bass"),
    ]

    audio_files = []
    for path in instruments_paths:
        if path.is_file():
            audio_files.append(path)
        else:
            # Load all wavfiles in directory
            audio_files.extend(list(path.glob("*.*")))

    betas = [0.01, 0.001, 0.0001, 0.00001, 0]

    for beta in betas:
        now = datetime.now()
        train(
            audio_files,
            run_name=f"base_model_beta_{beta}",
            encoder_layers=(1024, 512, 256, 128, 64, 32, 16, 8, 4),
            latent_dim=4,
            beta=beta,
            log_path="base_model_beta_variation",
            **kwargs,
        )
        print(f"Total duration for base model with beta {beta}: {datetime.now() - now}")


def train_base_model_with_full_latent_dim(kwargs):
    instruments_paths = [
        Path("data_instruments/piano"),
        Path("data_instruments/voice"),
        Path("data_instruments/guitar"),
        Path("data_instruments/bass"),
    ]

    audio_files = []
    for path in instruments_paths:
        if path.is_file():
            audio_files.append(path)
        else:
            # Load all wavfiles in directory
            audio_files.extend(list(path.glob("*.*")))

    now = datetime.now()

    train(
        audio_files,
        run_name="base_model_big_no_beta",
        encoder_layers=(2048, 1024, 512, 256, 128, 64, 32, 16, 8),
        latent_dim=8,
        beta=0,
        log_path="base_model_big",
        **kwargs,
    )

    train(
        audio_files,
        run_name="base_model_big_beta",
        encoder_layers=(2048, 1024, 512, 256, 128, 64, 32, 16, 8),
        latent_dim=8,
        beta=0.001,
        log_path="base_model_big",
        **kwargs,
    )
    print(
        f"Total duration for base model with full latent dim training: {datetime.now() - now}"
    )


def train_model_instruments_with_full_latent_dim(
    from_checkpoint=False, beta=0.0, **kwargs
):
    """
    Entrenar modelo particular de cada instrumento.

    :param from_checkpoint: Si es True entrena a partir del último checkpoint guardado de base_model
    :param beta: El valor de beta para el loss KL
    :param kwargs: Description
    """
    log_path = (
        "instruments_from_checkpoint_big"
        if from_checkpoint
        else "instruments_from_scratch_big"
    )
    beta_str = "beta" if beta > 0 else "no_beta"
    ckpt_str = "from_checkpoint" if from_checkpoint else "from_scratch"

    for instrument in ["piano", "voice", "guitar", "bass"]:
        audio_path = [
            Path(f"data_instruments/{instrument}"),
        ]

        audio_files = []
        for path in audio_path:
            if path.is_file():
                audio_files.append(path)
            else:
                # Load all wavfiles in directory
                audio_files.extend(list(path.glob("*.*")))

        now = datetime.now()

        checkpoint_path = None
        if from_checkpoint:
            checkpoint_path = Path(
                f"base_model_big/base_model_big_{beta_str}/version_0/checkpoints"
            ).glob("*.ckpt")
            checkpoint_path = list(checkpoint_path)[0]

        run_name = f"{instrument}_{ckpt_str}_{beta_str}"

        train(
            audio_files,
            run_name=run_name,
            encoder_layers=(2048, 1024, 512, 256, 128, 64, 32, 16, 8),
            latent_dim=8,
            beta=beta,
            log_path=log_path,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        mode_str = "with" if beta > 0 else "without"
        print(
            f"Total duration for {instrument} {ckpt_str.replace('_', ' ')} {mode_str} beta training: {datetime.now() - now}"
        )


def test_train_only_guitar(kwargs):
    audio_path = [
        Path("data_instruments/guitar_mono"),
    ]

    audio_files = []
    for path in audio_path:
        if path.is_file():
            audio_files.append(path)
        else:
            # Load all wavfiles in directory
            audio_files.extend(list(path.glob("*.*")))

    now = datetime.now()

    train(
        audio_files,
        run_name="guitar_only_no_beta",
        encoder_layers=(1024, 512, 256, 128, 64, 32, 16, 8, 4),
        latent_dim=4,
        beta=0,
        log_path="guitar_only",
        **kwargs,
    )

    train(
        audio_files,
        run_name="guitar_only_beta",
        encoder_layers=(1024, 512, 256, 128, 64, 32, 16, 8, 4),
        latent_dim=4,
        beta=0.001,
        log_path="guitar_only",
        **kwargs,
    )
    print(f"Total duration for guitar only training: {datetime.now() - now}")


def main(path=None, **kwargs):
    # betas = [0.01, 0.001, 0.0001, 0.00001, 0]
    # checkpoint_path = Path("tb_logs_vae/piano/version_0/checkpoints").glob("*.ckpt")
    # checkpoint_path = list(checkpoint_path)[0]

    # experiment_latent_dim(kwargs)
    # train_model_base(**kwargs)

    # train_model_instruments(from_checkpoint=True, beta=0, **kwargs)
    # train_model_instruments(from_checkpoint=True, beta=0.001, **kwargs)
    # train_model_instruments(from_checkpoint=False, beta=0, **kwargs)
    # train_model_instruments(from_checkpoint=False, beta=0.001, **kwargs)

    # experiment_multiple_betas_base_model(kwargs)
    # experiment_latent_dim_base_model(kwargs, beta=0.001)
    # experiment_latent_dim_base_model(kwargs, beta=0)

    # TOOD: por ahora dejo siempre beta=0.001, ver si mejora reconstrucción bajando un poquito el beta
    # train_model_instruments(from_checkpoint=True, beta=0.0001, **kwargs)
    # train_model_base(beta=0.0001, **kwargs)

    # train_base_model_with_full_latent_dim(kwargs)
    # train_model_instruments_with_full_latent_dim(
    #     from_checkpoint=True, beta=0.001, **kwargs
    # )
    # train_model_instruments_with_full_latent_dim(from_checkpoint=True, beta=0, **kwargs)

    test_train_only_guitar(kwargs)


if __name__ == "__main__":
    Fire(main)

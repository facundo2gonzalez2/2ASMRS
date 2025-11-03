from VariationalAutoEncoder import VariationalAutoEncoder
from pathlib import Path
import yaml
import torch
import numpy as np
from audio_utils import spectrogram2audio
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


def generate_audio(Y, hps, phase_option, frames):
    """Generate audio from spectrogram using the specified phase reconstruction method"""
    fc = 1
    hl = hps["hop_length"] * fc
    wl = hps["win_length"] * fc
    Y_ = torch.nn.functional.interpolate(Y[:, None, :], (wl // 2 + 1,))[:, 0, :]

    if phase_option == "pv":
        griffinlim = False
        phase = torch.rand(Y_.shape[1]) * torch.pi * 2
        grid = torch.meshgrid(
            torch.arange(0, frames, dtype=torch.float64),
            torch.zeros(Y_.shape[1], dtype=torch.float64),
        )[0]
        freqs = torch.linspace(0, hps["target_sampling_rate"] // 2, wl // 2 + 1)
        dt = hl / hps["target_sampling_rate"]
        phase = phase + freqs * 2 * torch.pi * dt * grid
    elif phase_option == "random":
        griffinlim = False
        phase = (torch.rand(Y_.shape) * 2 - 1) * torch.pi
    elif phase_option == "griffinlim":
        phase = (torch.rand(Y_.shape) * 2 - 1) * torch.pi
        griffinlim = True

    audio = (
        spectrogram2audio(
            Y_,
            hps["db_min_norm"],
            phase,
            hl,
            wl,
            hps["spec_in_db"],
            griffinlim=griffinlim,
        )
        .cpu()
        .numpy()
    )

    return audio


if __name__ == "__main__":
    beta = 0.001
    path_a = "tb_logs_vae/fur_elise_guitar/version_1"
    # path_b = "tb_logs_vae/fur_elise_piano/version_1"
    path_b = "tb_logs_vae/piano/version_0"

    checkpoint_path_a = list(Path(path_a, "checkpoints").glob("*.ckpt"))[0]
    checkpoint_path_b = list(Path(path_b, "checkpoints").glob("*.ckpt"))[0]

    latent_dim = 4

    with open(Path(path_a, "hparams.yaml")) as file:
        hps_a = yaml.load(file, Loader=yaml.FullLoader)

    with open(Path(path_b, "hparams.yaml")) as file:
        hps_b = yaml.load(file, Loader=yaml.FullLoader)

    vae_a = VariationalAutoEncoder(
        encoder_layers=hps_a["encoder_layers"],
        decoder_layers=hps_a["decoder_layers"],
        latent_dim=latent_dim,
        checkpoint_path=checkpoint_path_a,
    )

    vae_b = VariationalAutoEncoder(
        encoder_layers=hps_b["encoder_layers"],
        decoder_layers=hps_b["decoder_layers"],
        latent_dim=latent_dim,
        checkpoint_path=checkpoint_path_b,
    )

    vae_a.decoder.eval()
    vae_b.decoder.eval()

    npoints = 10
    frames_per_point = 100
    torch.manual_seed(42)
    zdim = latent_dim
    zs = []
    z1 = torch.randn(1, zdim)
    for i in range(npoints):
        z2 = torch.randn(1, zdim)
        for t in np.linspace(0, 1, frames_per_point):
            zs.append(torch.lerp(z1, z2, t))
        z1 = z2
        z2 = torch.randn(1, zdim)
    frames = len(zs)
    z = torch.vstack(zs)

    with torch.no_grad():
        Y_a = vae_a.decoder(z) * hps_a["Xmax"]
        Y_b = vae_b.decoder(z) * hps_b["Xmax"]

    # phase options
    phase_option = "pv"
    # phase_option = 'griffinlim'
    phase_option = "random"

    # Generate audio from model A
    audio_a = generate_audio(Y_a, hps_a, phase_option, frames)

    # Generate audio from model B
    audio_b = generate_audio(Y_b, hps_b, phase_option, frames)

    # Save both audio files
    sf.write(
        f"outputs/output_model_a_beta_{beta}.wav",
        audio_a,
        hps_a["target_sampling_rate"],
    )
    sf.write(
        f"outputs/output_model_piano_beta_{beta}.wav",
        audio_b,
        hps_b["target_sampling_rate"],
    )

    print(f"Generated audio from model A (guitar): output_model_a_beta_{beta}.wav")
    print(f"Generated audio from model B (piano): output_model_piano_beta_{beta}.wav")

    # Create comparison spectrograms
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Model A spectrogram
    S_a = librosa.stft(
        y=audio_a, n_fft=hps_a["win_length"], hop_length=hps_a["hop_length"]
    )
    S_dB_a = librosa.power_to_db(np.abs(S_a) ** 2, ref=np.max)
    librosa.display.specshow(
        S_dB_a,
        x_axis="time",
        y_axis="linear",
        sr=hps_a["target_sampling_rate"],
        ax=axes[0],
    )
    axes[0].set_title("Model A (Guitar)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(axes[0].collections[0], ax=axes[0], format="%+2.0f dB")

    # Model B spectrogram
    S_b = librosa.stft(
        y=audio_b, n_fft=hps_b["win_length"], hop_length=hps_b["hop_length"]
    )
    S_dB_b = librosa.power_to_db(np.abs(S_b) ** 2, ref=np.max)
    librosa.display.specshow(
        S_dB_b,
        x_axis="time",
        y_axis="linear",
        sr=hps_b["target_sampling_rate"],
        ax=axes[1],
    )
    axes[1].set_title("Model B (Piano)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(axes[1].collections[0], ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(f"outputs/comparison_spectrograms_beta_{beta}.png", dpi=150)
    print(
        f"Saved comparison spectrograms to: outputs/comparison_spectrograms_beta_{beta}.png"
    )
    # plt.show()

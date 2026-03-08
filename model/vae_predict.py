from fire import Fire
from pathlib import Path
import yaml
import torch
import soundfile as sf

from audio_utils import (
    get_spectrograms_from_audios,
    save_audio,
    spectrogram2audio,
)
from VariationalAutoEncoder import VariationalAutoEncoder


def predict_audio(
    audio_path_list=None,
    model_path=None,
    output_path=None,
    db_min_norm=-60,
    spec_in_db=True,
    normalize_each_audio=False,
    predicted_specgram=None,
    hps=None,
    phase_option="random",
    frames=None,
    return_audio=False,
    Xmax_overload=None,
):
    if predicted_specgram is not None:
        if hps is None:
            raise ValueError("hps es requerido cuando predicted_specgram no es None")

        hl = hps["hop_length"]
        wl = hps["win_length"]

        if frames is None:
            frames = predicted_specgram.shape[0]

        Y_ = torch.nn.functional.interpolate(
            predicted_specgram[:, None, :], (wl // 2 + 1,)
        )[:, 0, :]

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
        else:
            raise ValueError(f"phase_option inválido: {phase_option}")

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

        if output_path is not None:
            sf.write(output_path, audio, hps["target_sampling_rate"])

        if return_audio:
            return audio
        return None

    if audio_path_list is None or model_path is None or output_path is None:
        raise ValueError(
            "audio_path_list, model_path y output_path son requeridos en modo inferencia desde audio"
        )

    checkpoint_path = list(Path(model_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_path, "hparams.yaml")) as file:
        hps_a = yaml.load(file, Loader=yaml.FullLoader)

    hop_length = hps_a["hop_length"]
    win_length = hps_a["win_length"]
    target_sampling_rate = hps_a["target_sampling_rate"]

    X, phases, Xmax, y = get_spectrograms_from_audios(
        audio_path_list,
        target_sampling_rate,
        win_length,
        hop_length,
        db_min_norm=db_min_norm,
        spec_in_db=spec_in_db,
        normalize_each_audio=normalize_each_audio,
    )

    vae = VariationalAutoEncoder(
        encoder_layers=hps_a["encoder_layers"],
        decoder_layers=hps_a["decoder_layers"],
        latent_dim=hps_a["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    vae.decoder.eval()

    predicted_specgram = vae.predict(X) * (
        Xmax if Xmax_overload is None else Xmax_overload
    )

    save_audio(
        predicted_specgram,
        db_min_norm,
        phases,
        hop_length,
        win_length,
        target_sampling_rate,
        output_path,
        spec_in_db,
    )


def main(path=None, **kwargs):
    voice_model_path = (
        "instruments_from_checkpoint/voice_from_checkpoint_no_beta/version_0"
    )
    piano_model_path = (
        "instruments_from_checkpoint/piano_from_checkpoint_no_beta/version_0"
    )
    guitar_model_path = (
        "instruments_from_checkpoint/guitar_from_checkpoint_no_beta/version_0"
    )
    bass_model_path = (
        "instruments_from_checkpoint/bass_from_checkpoint_no_beta/version_0"
    )

    voice_path = Path("data_test/voice/voice_test.wav")
    piano_path = Path("data_test/piano_test.wav")
    guitar_path = Path("data_test/guitar_test.wav")
    bass_path = Path("data_test/bass/bass_test.wav")

    path = guitar_path
    model_path = guitar_model_path

    if path.is_file():
        audio_list = [path]
    else:
        # Load all wavfiles in directory
        audio_list = list(path.glob("*.*"))

    predict_audio(audio_list, model_path, output_path="test.wav", Xmax_overload=200.0)


if __name__ == "__main__":
    Fire(main)

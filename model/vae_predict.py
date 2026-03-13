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


def predict_all(instrument, path):
    xmax_values = {
        "guitar": 150.0,
        "piano": 120.0,
        "voice": 140.0,
        "bass": 120.0,
    }
    model_path = (
        f"instruments_from_checkpoint/{instrument}_from_checkpoint_no_beta/version_0"
    )
    data_path = Path(path, instrument)
    for audio_path in data_path.glob("*.wav"):
        output_path = Path(f"data_test_gt/{instrument}", f"{audio_path.stem}.wav")
        print(f"Predicting {audio_path} -> {output_path}...")
        predict_audio(
            audio_path_list=[audio_path],
            model_path=model_path,
            output_path=output_path,
            Xmax_overload=xmax_values.get(instrument, 120.0),
        )


def main(path=None, **kwargs):
    for instrument in ["guitar", "piano", "voice", "bass"]:
        print(f"Predicting {instrument}...")
        predict_all(instrument, "data_test")


if __name__ == "__main__":
    Fire(main)

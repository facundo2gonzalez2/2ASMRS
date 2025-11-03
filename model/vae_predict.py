from fire import Fire
from pathlib import Path
import yaml

from audio_utils import (
    get_spectrograms_from_audios,
    save_audio,
)
from VariationalAutoEncoder import VariationalAutoEncoder


def predict_audio(
    audio_path_list,
    model_path,
    output_path,
    db_min_norm=-60,
    spec_in_db=True,
    normalize_each_audio=False,
):
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

    predicted_specgram = vae.predict(X) * Xmax

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
    # model_path = "tb_logs_vae/fur_elise_guitar/version_1"
    # model_path = "tb_logs_vae/fur_elise_piano/version_1"
    model_path = "tb_logs_vae/playground/version_0"

    path = Path("data/playground/pianoclassic.mp3")
    # path = Path("data/fur_elise_piano.mp3")
    # path = Path("data/mozart_sonata_piano.mp3")

    if path.is_file():
        audio_list = [path]
    else:
        # Load all wavfiles in directory
        audio_list = list(path.glob("*.*"))

    predict_audio(
        audio_list, model_path, output_path="outputs/playground_reconstructed.mp3"
    )


if __name__ == "__main__":
    Fire(main)

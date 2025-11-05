from VariationalAutoEncoder import VariationalAutoEncoder
import collections
from pathlib import Path
import yaml
import torch
from explor_vae import generate_audio
import soundfile as sf
from audio_utils import get_spectrograms_from_audios


def interpolar_vae(
    model_a: VariationalAutoEncoder,
    model_b: VariationalAutoEncoder,
    alpha: float,
    encoder_layers,
    decoder_layers,
    latent_dim,
) -> VariationalAutoEncoder:
    # 1. Obtener los diccionarios de estado (parámetros)
    theta_a = model_a.state_dict()
    theta_b = model_b.state_dict()

    # 2. Crear un nuevo diccionario para los pesos interpolados
    theta_interp = collections.OrderedDict()

    # 3. Iterar sobre todos los parámetros
    for key in theta_a:
        if key in theta_b:
            # 4. Calcular la interpolación lineal (LERP)
            theta_interp[key] = (1.0 - alpha) * theta_a[key] + alpha * theta_b[key]
        else:
            # Esto no debería pasar si las arquitecturas son idénticas
            raise KeyError(
                f"Clave '{key}' no encontrada en model_b. Las arquitecturas no coinciden."
            )

    # 5. Crear una nueva instancia del modelo VAE
    #    ¡Importante! No pasamos 'checkpoint_path' aquí.
    modelo_interpolado = VariationalAutoEncoder(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        latent_dim=latent_dim,
    )

    # 6. Cargar los pesos interpolados en el nuevo modelo
    modelo_interpolado.load_state_dict(theta_interp)

    return modelo_interpolado


if __name__ == "__main__":
    # model_path_a = "tb_logs_vae/playground/version_6"
    # model_path_b = "tb_logs_vae/playground/version_7"
    model_path_a = "tb_logs_vae/model_fine_tunning_guitar/version_0"
    model_path_b = "tb_logs_vae/model_fine_tunning/version_0"
    output_dir = "outputs/interpolate_fine_tuned/"

    checkpoint_path_a = list(Path(model_path_a, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_path_a, "hparams.yaml")) as file:
        hps_a = yaml.load(file, Loader=yaml.FullLoader)

    checkpoint_path_b = list(Path(model_path_b, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_path_b, "hparams.yaml")) as file:
        hps_b = yaml.load(file, Loader=yaml.FullLoader)

    print("Cargando modelo A...")
    model_a = VariationalAutoEncoder(
        encoder_layers=hps_a["encoder_layers"],
        decoder_layers=hps_a["decoder_layers"],
        latent_dim=hps_a["latent_dim"],
        checkpoint_path=checkpoint_path_a,
    )

    print("Cargando modelo B...")
    model_b = VariationalAutoEncoder(
        encoder_layers=hps_b["encoder_layers"],
        decoder_layers=hps_b["decoder_layers"],
        latent_dim=hps_b["latent_dim"],
        checkpoint_path=checkpoint_path_b,
    )

    assert hps_a["encoder_layers"] == hps_b["encoder_layers"], (
        "Las arquitecturas de los modelos no coinciden."
    )
    assert hps_a["decoder_layers"] == hps_b["decoder_layers"], (
        "Las arquitecturas de los modelos no coinciden."
    )
    assert hps_a["latent_dim"] == hps_b["latent_dim"], (
        "Las arquitecturas de los modelos no coinciden."
    )

    # --- 3. Interpola los modelos ---
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    encoders = [("guitar", model_a), ("voice", model_b), ("interpolated", None)]
    audios = [
        Path("data/playground/c-major-scale-90710.mp3"),
        Path("data/playground/c-major-scale-child-102262.mp3"),
    ]
    for alpha in alphas:
        for encoder_name, encoder_model in encoders:
            for audio_path in audios:
                print(f"Interpolando modelos con alpha={alpha}...")
                modelo_interpolado = interpolar_vae(
                    model_a,
                    model_b,
                    alpha,
                    encoder_layers=hps_a["encoder_layers"],
                    decoder_layers=hps_a["decoder_layers"],
                    latent_dim=hps_a["latent_dim"],
                )
                print("¡Interpolación completada!")

                # --- 4. Usa el modelo interpolado para predicción ---
                modelo_interpolado.eval()
                modelo_interpolado.decoder.eval()

                if encoder_model is None:
                    encoder_model = modelo_interpolado

                encoder_model.eval()
                encoder_model.encoder.eval()

                # Load and process audio
                X, phases, Xmax, y = get_spectrograms_from_audios(
                    [audio_path],
                    hps_a["target_sampling_rate"],
                    hps_a["win_length"],
                    hps_a["hop_length"],
                    db_min_norm=hps_a["db_min_norm"],
                    spec_in_db=hps_a["spec_in_db"],
                    normalize_each_audio=hps_a["normalize_each_audio"],
                )

                # Step 1: Encode audio using model A's encoder
                print(f"Encoding audio with model {encoder_name} encoder...")
                with torch.no_grad():
                    mu, logvar = encoder_model.encoder(X)
                    z = mu  # Use the mean of the latent distribution

                print(f"Encoded {z.shape[0]} frames with latent dimension {z.shape[1]}")

                # Step 2: Decode using the interpolated model's decoder
                print("Decoding with interpolated model's decoder...")
                with torch.no_grad():
                    Y = modelo_interpolado.decoder(z) * hps_a["Xmax"]

                frames = Y.shape[0]

                # Phase reconstruction options
                # phase_option = "pv"
                # phase_option = 'griffinlim'
                phase_option = "random"

                print(
                    f"Generating audio with {frames} frames using phase method: {phase_option}"
                )
                audio = generate_audio(Y, hps_a, phase_option, frames)

                output_path = (
                    output_dir + f"exp_{alpha}_{encoder_name}_{audio_path.stem}.wav"
                )
                sf.write(output_path, audio, hps_a["target_sampling_rate"])
                print(f"Audio saved to: {output_path}")

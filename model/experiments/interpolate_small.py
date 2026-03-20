from pathlib import Path
import yaml
import torch
import numpy as np
import sys
import soundfile as sf

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from VariationalAutoEncoder import VariationalAutoEncoder # noqa: E402
from scripts.vae_predict import predict_audio # noqa: E402
from audio_utils import get_spectrograms_from_audios # noqa: E402
from experiments.interpolate import interpolar_vae # noqa: E402

def run_interpolation_latent_experiment(
    model_a_path: str,
    model_b_path: str,
    output_dir: str,
    num_frames: int = 128,
    random_seed: int = 42,
    sampling_mode: str = "encoded",
    reference_audio_paths=None,
    model_a_xmax: float = 230.0,
    model_b_xmax: float = 120.0,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path_a = list(Path(model_a_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_a_path, "hparams.yaml")) as file:
        hps_a = yaml.load(file, Loader=yaml.FullLoader)

    checkpoint_path_b = list(Path(model_b_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_b_path, "hparams.yaml")) as file:
        hps_b = yaml.load(file, Loader=yaml.FullLoader)

    print("Cargando modelo A (piano)...")
    model_a = VariationalAutoEncoder(
        encoder_layers=hps_a["encoder_layers"],
        decoder_layers=hps_a["decoder_layers"],
        latent_dim=hps_a["latent_dim"],
        checkpoint_path=checkpoint_path_a,
    )

    print("Cargando modelo B (guitarra)...")
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

    torch.manual_seed(random_seed)
    latent_dim = hps_a["latent_dim"]

    if sampling_mode == "gaussian":
        z_base = torch.randn(1, latent_dim)
    elif sampling_mode == "encoded":
        if reference_audio_paths is None:
            model_paths_text = f"{model_a_path} {model_b_path}".lower()
            instrument_candidates = []
            for instrument in ["voice", "piano", "guitar", "bass"]:
                if instrument in model_paths_text:
                    instrument_candidates.append(instrument)

            discovered_files = []
            # Support common audio extensions found in this repository.
            supported_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
            for instrument in instrument_candidates:
                source_dirs = [
                    Path("data_instruments", instrument),
                    Path("data_instruments_small", instrument),
                ]
                for source_dir in source_dirs:
                    if source_dir.exists():
                        files = sorted(
                            file
                            for file in source_dir.iterdir()
                            if file.is_file() and file.suffix.lower() in supported_exts
                        )
                        if files:
                            discovered_files.append(files[0])
                            break
            reference_audio_paths = discovered_files

        if not reference_audio_paths:
            raise ValueError(
                "No se encontraron audios de referencia para sampling_mode='encoded'. "
                "Pasa reference_audio_paths manualmente o usa sampling_mode='gaussian'."
            )

        X_ref, _, _, _ = get_spectrograms_from_audios(
            [Path(path) for path in reference_audio_paths],
            hps_a["target_sampling_rate"],
            hps_a["win_length"],
            hps_a["hop_length"],
            db_min_norm=hps_a["db_min_norm"],
            spec_in_db=hps_a["spec_in_db"],
            normalize_each_audio=hps_a["normalize_each_audio"],
        )

        with torch.no_grad():
            mu_a, _ = model_a.encoder(X_ref)
            mu_b, _ = model_b.encoder(X_ref)
            z_base = 0.5 * (
                mu_a.mean(dim=0, keepdim=True) + mu_b.mean(dim=0, keepdim=True)
            )
            print(f"Latent vector obtenido por encoding de referencia: {z_base.shape}")
            print(z_base)
    else:
        raise ValueError(
            f"sampling_mode inválido: {sampling_mode}. Usa 'encoded' o 'gaussian'."
        )

    alphas = [round(i * 0.1, 1) for i in range(11)]
    phase_option = "griffinlim"
    generated_audios = []

    for alpha in alphas:
        model_xmax = (1.0 - alpha) * model_a_xmax + alpha * model_b_xmax
        print(f"Interpolando modelos con alpha={alpha}...")
        modelo_interpolado: VariationalAutoEncoder = interpolar_vae(
            model_a,
            model_b,
            alpha,
            encoder_layers=hps_a["encoder_layers"],
            decoder_layers=hps_a["decoder_layers"],
            latent_dim=hps_a["latent_dim"],
            interpolation_mode="slerp",
        )

        modelo_interpolado.eval()
        modelo_interpolado.decoder.eval()

        with torch.no_grad():
            z = z_base.repeat(num_frames, 1)
            predicted_specgram = modelo_interpolado.decoder(z) * model_xmax
            predicted_specgram = torch.nan_to_num(
                predicted_specgram, nan=0.0, posinf=model_xmax, neginf=0.0
            )
            predicted_specgram = torch.clamp(
                predicted_specgram, min=0.0, max=model_xmax
            )

        try:
            audio = predict_audio(
                predicted_specgram=predicted_specgram,
                hps=hps_a,
                phase_option=phase_option,
                frames=num_frames,
                return_audio=True,
            )
        except Exception as err:
            print(
                f"[WARN] alpha={alpha}: falló '{phase_option}' ({err}). Reintentando con phase_option='random'."
            )
            audio = predict_audio(
                predicted_specgram=predicted_specgram,
                hps=hps_a,
                phase_option="random",
                frames=num_frames,
                return_audio=True,
            )

        if audio is None:
            raise RuntimeError(
                f"predict_audio devolvió None para alpha={alpha} en run_random_interpolation_experiment"
            )

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        generated_audios.append(audio)
        current_output = output_path / f"latent_alpha_{alpha:.1f}.wav"
        sf.write(str(current_output), audio, hps_a["target_sampling_rate"])
        print(f"Audio guardado en: {current_output}")

    if generated_audios:
        transition_audio = np.concatenate(generated_audios)
        transition_output = output_path / "latent_transition_alpha_0_to_1.wav"
        sf.write(
            str(transition_output), transition_audio, hps_a["target_sampling_rate"]
        )
        print(f"Audio transición guardado en: {transition_output}")


def cross_interpolation(random_seed, sampling_mode, xmax_values):
    instruments = ["piano", "voice", "guitar", "bass"]
    pairs = [(src, tgt) for src in instruments for tgt in instruments if src != tgt]

    for checkpoint_str in ["checkpoint", "scratch"]:
        for beta_str in ["beta_0.001", "no_beta"]:
            for source, target in pairs:
                model_a_path = f"inference_models/instruments_from_{checkpoint_str}/{source}_from_{checkpoint_str}_{beta_str}/version_0"
                model_b_path = f"inference_models/instruments_from_{checkpoint_str}/{target}_from_{checkpoint_str}_{beta_str}/version_0"

                output_dir = f"AudiosInterpolacion/{beta_str}/{checkpoint_str}/interpolation_{source}_to_{target}/"

                run_interpolation_latent_experiment(
                    model_a_path,
                    model_b_path,
                    output_dir,
                    random_seed=random_seed,
                    sampling_mode=sampling_mode,
                    model_a_xmax=xmax_values[source],
                    model_b_xmax=xmax_values[target],
                    num_frames=48,
                )


if __name__ == "__main__":
    instrument_a = "guitar"
    instrument_b = "voice"
    beta = True
    sampling_mode = 0  # 0 para encoded, 1 para gaussian

    xmax_values = {
        "guitar": 150.0,
        "piano": 150.0,
        "voice": 140.0,
        "bass": 120.0,
    }

    model_a_path = f"inference_models/instruments_from_checkpoint/{instrument_a}_from_checkpoint_{'beta_0.001' if beta else 'no_beta'}/version_0"
    model_b_path = f"inference_models/instruments_from_checkpoint/{instrument_b}_from_checkpoint_{'beta_0.001' if beta else 'no_beta'}/version_0"

    output_dir = f"interpolation_audio_{instrument_a}_to_{instrument_b}_{'beta_0.001' if beta else 'no_beta'}/"

    run_interpolation_latent_experiment(
        model_a_path,
        model_b_path,
        output_dir,
        sampling_mode="encoded" if sampling_mode == 0 else "gaussian",
        num_frames=64,
        random_seed=42,
        model_a_xmax=xmax_values[instrument_a],
        model_b_xmax=xmax_values[instrument_b],
    )

    # cross_interpolation(
    #     random_seed=42,
    #     sampling_mode="encoded" if sampling_mode == 0 else "gaussian",
    #     xmax_values=xmax_values,
    # )

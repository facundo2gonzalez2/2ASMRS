from VariationalAutoEncoder import VariationalAutoEncoder
import collections
from pathlib import Path
import yaml
import torch
import numpy as np
from scripts.vae_predict import predict_audio
import soundfile as sf
from audio_utils import get_spectrograms_from_audios, save_audio
import matplotlib.pyplot as plt


def interpolar_vae(
    model_a: VariationalAutoEncoder,
    model_b: VariationalAutoEncoder,
    alpha: float,
    encoder_layers,
    decoder_layers,
    latent_dim,
    interpolation_mode: str = "linear",
) -> VariationalAutoEncoder:
    # 1. Obtener los diccionarios de estado (parámetros)
    theta_a = model_a.state_dict()
    theta_b = model_b.state_dict()

    # 2. Crear un nuevo diccionario para los pesos interpolados
    theta_interp = collections.OrderedDict()

    def _slerp_tensor(tensor_a: torch.Tensor, tensor_b: torch.Tensor, t: float):
        # SLERP está definido para vectores en una esfera; aplanamos y luego restauramos forma.
        original_dtype = tensor_a.dtype
        vec_a = tensor_a.reshape(-1)
        vec_b = tensor_b.reshape(-1)

        if not torch.is_floating_point(vec_a):
            return tensor_a if t < 0.5 else tensor_b

        vec_a_f = vec_a.float()
        vec_b_f = vec_b.float()

        norm_a = torch.norm(vec_a_f)
        norm_b = torch.norm(vec_b_f)
        if norm_a.item() == 0.0 or norm_b.item() == 0.0:
            interpolated = (1.0 - t) * vec_a_f + t * vec_b_f
            return interpolated.reshape_as(tensor_a).to(dtype=original_dtype)

        unit_a = vec_a_f / norm_a
        unit_b = vec_b_f / norm_b

        dot = torch.clamp(torch.sum(unit_a * unit_b), -1.0, 1.0)

        # Si el ángulo es muy pequeño (o casi opuesto), hacemos fallback a LERP por estabilidad.
        if torch.abs(dot) > 0.9995:
            interpolated = (1.0 - t) * vec_a_f + t * vec_b_f
        else:
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1.0 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            interpolated = w1 * vec_a_f + w2 * vec_b_f

        return interpolated.reshape_as(tensor_a).to(dtype=original_dtype)

    if interpolation_mode not in {"linear", "slerp"}:
        raise ValueError(
            f"interpolation_mode inválido: {interpolation_mode}. Usa 'linear' o 'slerp'."
        )

    # 3. Iterar sobre todos los parámetros
    for key in theta_a:
        if key in theta_b:
            # 4. Calcular la interpolación configurada (lineal o SLERP)
            if interpolation_mode == "slerp":
                theta_interp[key] = _slerp_tensor(theta_a[key], theta_b[key], alpha)
            else:
                tensor_a = theta_a[key]
                tensor_b = theta_b[key]
                if not torch.is_floating_point(tensor_a):
                    theta_interp[key] = tensor_a if alpha < 0.5 else tensor_b
                else:
                    theta_interp[key] = (1.0 - alpha) * tensor_a + alpha * tensor_b
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


def _first_experiment():
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
                audio = predict_audio(
                    predicted_specgram=Y,
                    hps=hps_a,
                    phase_option=phase_option,
                    frames=frames,
                    return_audio=True,
                )
                if audio is None:
                    raise RuntimeError(
                        "predict_audio devolvió None en _first_experiment"
                    )

                output_path = (
                    output_dir + f"exp_{alpha}_{encoder_name}_{audio_path.stem}.wav"
                )
                sf.write(output_path, audio, hps_a["target_sampling_rate"])
                print(f"Audio saved to: {output_path}")


def run_interpolation_experiment(
    model_a_path: str,
    model_b_path: str,
    output_dir: str,
    base_model_path: str = "inference_models/base_model/base_model_no_beta/version_0",
    encoder_with_base_model: bool = False,
):
    checkpoint_path_a = list(Path(model_a_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_a_path, "hparams.yaml")) as file:
        hps_a = yaml.load(file, Loader=yaml.FullLoader)

    checkpoint_path_b = list(Path(model_b_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_b_path, "hparams.yaml")) as file:
        hps_b = yaml.load(file, Loader=yaml.FullLoader)

    checkpoint_base_model = list(Path(base_model_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(base_model_path, "hparams.yaml")) as file:
        hps_base_model = yaml.load(file, Loader=yaml.FullLoader)

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

    print("Cargando base_model...")
    base_model = VariationalAutoEncoder(
        encoder_layers=hps_base_model["encoder_layers"],
        decoder_layers=hps_base_model["decoder_layers"],
        latent_dim=hps_base_model["latent_dim"],
        checkpoint_path=checkpoint_base_model,
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

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    audios = [
        Path("piano-c-major-scale.wav"),
        # Path("data_instruments/guitar/00_SS2-107-Ab_comp_hex.wav"),
        # Path("umap_experiment/bass_cut.mp3"),
    ]

    for alpha in alphas:
        print(f"Interpolando modelos con alpha={alpha}...")
        for audio_path in audios:
            modelo_interpolado: VariationalAutoEncoder = interpolar_vae(
                model_a,
                model_b,
                alpha,
                encoder_layers=hps_a["encoder_layers"],
                decoder_layers=hps_a["decoder_layers"],
                latent_dim=hps_a["latent_dim"],
            )

            modelo_interpolado.decoder.eval()
            base_model.encoder.eval()

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

            if encoder_with_base_model:
                # 1 encode with base model
                with torch.no_grad():
                    mu, logvar = base_model.encoder(X)
                    z = mu  # Use the mean of the latent distribution

                # 2 decode with interpolated model
                with torch.no_grad():
                    predicted_specgram = modelo_interpolado.decoder(z) * Xmax
            else:
                predicted_specgram = modelo_interpolado.predict(X) * Xmax

            output_path = output_dir + f"exp_{alpha}_{audio_path.stem}.wav"

            save_audio(
                predicted_specgram,
                hps_a["db_min_norm"],
                phases,
                hps_a["hop_length"],
                hps_a["win_length"],
                hps_a["target_sampling_rate"],
                output_path,
                hps_a["spec_in_db"],
            )

            print(f"Audio saved to: {output_path}")

    audio_path_name = "piano-c-major-scale"
    results = {}

    reference_audios = [
        Path("piano-c-major-scale-predicted-no_beta.wav"),
        Path("guitar-c-major-scale-predicted.wav"),
        Path("umap_experiment/bass_cut.mp3"),
    ]

    for alpha in alphas:
        results[alpha] = {}
        for audio in reference_audios:
            print(f"Calculando similitud para alpha={alpha} y audio={audio}...")
            interp_file = output_dir + f"exp_{alpha}_{audio_path_name}.wav"
            sim = get_cosine_similarity(interp_file, str(audio))
            results[alpha][str(audio)] = sim

    return results


def run_random_interpolation_experiment(
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
    # assert hps_a["latent_dim"] == 4, (
    #     "Este experimento está configurado para latent_dim=4."
    # )

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
    # model_xmax = float(hps_a.get("Xmax", 1.0))
    generated_audios = []

    for alpha in alphas:
        # model_xmax = (1.0 - alpha) * model_a_xmax + alpha * model_b_xmax
        model_xmax = (model_a_xmax + model_b_xmax) / 2.0
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

        # meter un poco de ruido al z_base:
        z_base += torch.randn_like(z_base) * 0.05

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
        current_output = output_path / f"random_latent4_alpha_{alpha:.1f}.wav"
        sf.write(str(current_output), audio, hps_a["target_sampling_rate"])
        print(f"Audio guardado en: {current_output}")

    if generated_audios:
        transition_audio = np.concatenate(generated_audios)
        transition_output = output_path / "random_latent4_transition_alpha_0_to_1.wav"
        sf.write(
            str(transition_output), transition_audio, hps_a["target_sampling_rate"]
        )
        print(f"Audio transición guardado en: {transition_output}")


def generate_plots_from_results(
    results, output_path="interpolation_outputs_6/similarity_plot.png"
):
    if not results:
        raise ValueError("results está vacío. Ejecuta la interpolación primero.")

    alphas = sorted(results.keys())
    instrument_paths = list(next(iter(results.values())).keys())

    series = {instrument: [] for instrument in instrument_paths}
    for alpha in alphas:
        for instrument in instrument_paths:
            series[instrument].append(results[alpha][instrument])

    plt.figure(figsize=(10, 6))
    for instrument, sims in series.items():
        label = Path(instrument).stem
        plt.plot(alphas, sims, marker="o", label=label)

    plt.title("Similitud de coseno vs alpha")
    plt.xlabel("alpha")
    plt.ylabel("similitud")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot guardado en: {output_path}")


def cross_interpolation(random_seed, sampling_mode, xmax_values):
    instruments = ["piano", "voice", "guitar", "bass"]
    pairs = [(src, tgt) for src in instruments for tgt in instruments if src != tgt]

    for checkpoint_str in ["checkpoint", "scratch"]:
        for beta_str in ["beta_0.001", "no_beta"]:
            for source, target in pairs:
                model_a_path = f"inference_models/instruments_from_{checkpoint_str}/{source}_from_{checkpoint_str}_{beta_str}/version_0"
                model_b_path = f"inference_models/instruments_from_{checkpoint_str}/{target}_from_{checkpoint_str}_{beta_str}/version_0"

                output_dir = f"AudiosInterpolacion/{beta_str}/{checkpoint_str}/interpolation_{source}_to_{target}/"

                run_random_interpolation_experiment(
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
    from audio_comparator import get_cosine_similarity

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

    output_dir = f"interpolation_outputs_random_{instrument_a}_to_{instrument_b}_{'beta_0.001' if beta else 'no_beta'}/"

    run_random_interpolation_experiment(
        model_a_path,
        model_b_path,
        output_dir,
        sampling_mode="encoded" if sampling_mode == 0 else "gaussian",
        num_frames=64,
        random_seed=42,
        model_a_xmax=xmax_values[instrument_a],
        model_b_xmax=xmax_values[instrument_b],
    )

    cross_interpolation(
        random_seed=42,
        sampling_mode="encoded" if sampling_mode == 0 else "gaussian",
        xmax_values=xmax_values,
    )

    # print("Resultados de similitud de coseno:")
    # for alpha, sims in results.items():
    #     print(f"Alpha {alpha}:")
    #     for audio, sim in sims.items():
    #         print(f"  {audio}: Similitud = {sim:.4f}")

    # generate_plots_from_results(results)

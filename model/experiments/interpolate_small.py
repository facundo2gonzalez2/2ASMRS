from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import yaml
import torch
import numpy as np
import sys
import soundfile as sf
from typing import Iterable

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from VariationalAutoEncoder import VariationalAutoEncoder  # noqa: E402
from scripts.vae_predict import predict_audio  # noqa: E402
from audio_utils import get_spectrograms_from_audios  # noqa: E402
from experiments.interpolate import interpolar_vae  # noqa: E402


def _resolve_num_frames(
    hps: dict, num_frames: int | None, duration_seconds: float | None
) -> int:
    if duration_seconds is not None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds debe ser mayor a 0.")
        frames_per_second = hps["target_sampling_rate"] / hps["hop_length"]
        return max(2, int(round(duration_seconds * frames_per_second)))

    if num_frames is None:
        raise ValueError("Debes pasar num_frames o duration_seconds.")

    if num_frames < 2:
        raise ValueError("num_frames debe ser >= 2 para interpolar.")

    return num_frames


def _discover_reference_audios(model_a_path: str, model_b_path: str) -> list[Path]:
    model_paths_text = f"{model_a_path} {model_b_path}".lower()
    instrument_candidates = []
    for instrument in ["voice", "piano", "guitar", "bass"]:
        if instrument in model_paths_text:
            instrument_candidates.append(instrument)

    discovered_files: list[Path] = []
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

    return discovered_files


def _as_latent_tensor(value, latent_dim: int, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().reshape(1, -1)
    else:
        tensor = torch.tensor(list(value), dtype=torch.float32).reshape(1, -1)

    if tensor.shape[1] != latent_dim:
        raise ValueError(
            f"{name} tiene dimensión {tensor.shape[1]}, pero latent_dim={latent_dim}."
        )

    return tensor


def _compute_latent_endpoints(
    model_a: VariationalAutoEncoder,
    model_b: VariationalAutoEncoder,
    hps: dict,
    sampling_mode: str,
    latent_dim: int,
    reference_audio_paths,
    model_a_path: str,
    model_b_path: str,
):
    if sampling_mode == "gaussian":
        z_start = torch.randn(1, latent_dim)
        z_end = torch.randn(1, latent_dim)
        return z_start, z_end

    if sampling_mode != "encoded":
        raise ValueError(
            f"sampling_mode inválido: {sampling_mode}. Usa 'encoded' o 'gaussian'."
        )

    if reference_audio_paths is None:
        reference_audio_paths = _discover_reference_audios(model_a_path, model_b_path)

    if not reference_audio_paths:
        raise ValueError(
            "No se encontraron audios de referencia para sampling_mode='encoded'. "
            "Pasa reference_audio_paths manualmente o usa sampling_mode='gaussian'."
        )

    def _encode_audio_mean(path_like) -> torch.Tensor:
        X_ref, _, _, _ = get_spectrograms_from_audios(
            [Path(path_like)],
            hps["target_sampling_rate"],
            hps["win_length"],
            hps["hop_length"],
            db_min_norm=hps["db_min_norm"],
            spec_in_db=hps["spec_in_db"],
            normalize_each_audio=hps["normalize_each_audio"],
        )

        with torch.no_grad():
            mu_a, _ = model_a.encoder(X_ref)
            mu_b, _ = model_b.encoder(X_ref)
            return 0.5 * (
                mu_a.mean(dim=0, keepdim=True) + mu_b.mean(dim=0, keepdim=True)
            )

    z_start = _encode_audio_mean(reference_audio_paths[0])
    z_end = _encode_audio_mean(
        reference_audio_paths[1]
        if len(reference_audio_paths) > 1
        else reference_audio_paths[0]
    )

    print(f"Latent start: {z_start.shape} | Latent end: {z_end.shape}")
    return z_start, z_end


def _report_and_autogain(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    print(f"Audio stats antes de normalizar: peak={peak:.6f}, rms={rms:.6f}")

    if peak <= 0.0:
        raise RuntimeError(
            "El audio generado es completamente silencioso (peak=0). "
            "Prueba con start_z/end_z manuales o sampling_mode='gaussian'."
        )

    if peak < 0.05:
        gain = target_peak / peak
        audio = np.clip(audio * gain, -1.0, 1.0)
        new_peak = float(np.max(np.abs(audio)))
        new_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        print(
            "Se aplico auto-gain por bajo nivel de salida: "
            f"gain={gain:.2f}, peak={new_peak:.6f}, rms={new_rms:.6f}"
        )

    return audio


def run_interpolation_latent_experiment(
    model_a_path: str,
    model_b_path: str,
    output_dir: str,
    num_frames: int | None = 128,
    duration_seconds: float | None = None,
    random_seed: int = 42,
    sampling_mode: str = "encoded",
    reference_audio_paths=None,
    model_a_xmax: float = 230.0,
    model_b_xmax: float = 120.0,
    start_alpha: float = 0.0,
    end_alpha: float = 1.0,
    start_z: Iterable[float] | torch.Tensor | None = None,
    end_z: Iterable[float] | torch.Tensor | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path_a = list(Path(model_a_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_a_path, "hparams.yaml")) as file:
        hps_a = yaml.load(file, Loader=yaml.FullLoader)

    checkpoint_path_b = list(Path(model_b_path, "checkpoints").glob("*.ckpt"))[0]
    with open(Path(model_b_path, "hparams.yaml")) as file:
        hps_b = yaml.load(file, Loader=yaml.FullLoader)

    model_a = VariationalAutoEncoder(
        encoder_layers=hps_a["encoder_layers"],
        decoder_layers=hps_a["decoder_layers"],
        latent_dim=hps_a["latent_dim"],
        checkpoint_path=checkpoint_path_a,
    )

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

    model_a.eval()
    model_b.eval()
    model_a.decoder.eval()
    model_b.decoder.eval()

    torch.manual_seed(random_seed)
    latent_dim = hps_a["latent_dim"]
    num_frames = _resolve_num_frames(hps_a, num_frames, duration_seconds)

    if not (0.0 <= start_alpha <= 1.0 and 0.0 <= end_alpha <= 1.0):
        raise ValueError("start_alpha y end_alpha deben estar en el rango [0, 1].")

    if start_z is not None:
        z_start = _as_latent_tensor(start_z, latent_dim, "start_z")
    else:
        z_start = None

    if end_z is not None:
        z_end = _as_latent_tensor(end_z, latent_dim, "end_z")
    else:
        z_end = None

    if z_start is None or z_end is None:
        z_start_auto, z_end_auto = _compute_latent_endpoints(
            model_a=model_a,
            model_b=model_b,
            hps=hps_a,
            sampling_mode=sampling_mode,
            latent_dim=latent_dim,
            reference_audio_paths=reference_audio_paths,
            model_a_path=model_a_path,
            model_b_path=model_b_path,
        )
        if z_start is None:
            z_start = z_start_auto
        if z_end is None:
            z_end = z_end_auto

    alpha_schedule = torch.linspace(start_alpha, end_alpha, steps=num_frames)
    z_schedule = torch.lerp(
        z_start, z_end, torch.linspace(0.0, 1.0, steps=num_frames)[:, None]
    )

    phase_option = "griffinlim"

    predicted_frames = []
    print(
        "Generando espectrograma continuo con interpolacion temporal "
        f"(frames={num_frames}, alpha={start_alpha:.3f}->{end_alpha:.3f})..."
    )

    for frame_idx in range(num_frames):
        alpha = float(alpha_schedule[frame_idx].item())
        model_xmax = (1.0 - alpha) * model_a_xmax + alpha * model_b_xmax
        interpolated_model = interpolar_vae(
            model_a,
            model_b,
            alpha,
            encoder_layers=hps_a["encoder_layers"],
            decoder_layers=hps_a["decoder_layers"],
            latent_dim=hps_a["latent_dim"],
            interpolation_mode="linear",
        )
        interpolated_model.eval()

        with torch.no_grad():
            z_frame = z_schedule[frame_idx : frame_idx + 1]
            z_pred = interpolated_model.decoder(z_frame) * model_xmax
            # pred_a = model_a.decoder(z_frame)
            # pred_b = model_b.decoder(z_frame)
            # predicted_frame = ((1.0 - alpha) * pred_a + alpha * pred_b) * model_xmax

            predicted_frame = torch.nan_to_num(
                z_pred, nan=0.0, posinf=model_xmax, neginf=0.0
            )
            predicted_frame = torch.clamp(predicted_frame, min=0.0, max=model_xmax)
            predicted_frames.append(predicted_frame)

    predicted_specgram = torch.cat(predicted_frames, dim=0)

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
            f"[WARN] falló '{phase_option}' ({err}). Reintentando con phase_option='random'."
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
            "predict_audio devolvió None en run_interpolation_latent_experiment"
        )

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = _report_and_autogain(audio)
    transition_output = output_path / "latent_transition_continuous.wav"
    sf.write(str(transition_output), audio, hps_a["target_sampling_rate"])
    print(f"Audio continuo guardado en: {transition_output}")

    metadata_output = output_path / "latent_transition_metadata.yaml"
    with open(metadata_output, "w") as file:
        yaml.safe_dump(
            {
                "num_frames": int(num_frames),
                "duration_seconds": float(
                    num_frames * hps_a["hop_length"] / hps_a["target_sampling_rate"]
                ),
                "start_alpha": float(start_alpha),
                "end_alpha": float(end_alpha),
                "sampling_mode": sampling_mode,
                "model_a_xmax": float(model_a_xmax),
                "model_b_xmax": float(model_b_xmax),
            },
            file,
            sort_keys=False,
        )
    print(f"Metadata guardada en: {metadata_output}")


def _cross_interpolation_worker(task: dict) -> tuple[str, bool, str | None]:
    source = task["source"]
    target = task["target"]
    checkpoint_str = task["checkpoint_str"]
    beta_str = task["beta_str"]
    label = f"{checkpoint_str}/{beta_str}/{source}_to_{target}"

    try:
        # Avoid CPU oversubscription when several processes use PyTorch.
        torch.set_num_threads(1)
        run_interpolation_latent_experiment(
            model_a_path=task["model_a_path"],
            model_b_path=task["model_b_path"],
            output_dir=task["output_dir"],
            sampling_mode=task["sampling_mode"],
            duration_seconds=task["duration_seconds"],
            random_seed=task["random_seed"],
            model_a_xmax=task["model_a_xmax"],
            model_b_xmax=task["model_b_xmax"],
            start_alpha=task["start_alpha"],
            end_alpha=task["end_alpha"],
            start_z=task["start_z"],
            end_z=task["end_z"],
        )
        return label, True, None
    except Exception as err:
        return label, False, str(err)


def cross_interpolation(
    random_seed,
    sampling_mode,
    xmax_values,
    start_z_values,
    duration_seconds: float = 20.0,
    max_workers: int | None = None,
):
    instruments = ["piano", "voice", "guitar", "bass"]
    pairs = [(src, tgt) for src in instruments for tgt in instruments if src != tgt]
    if sampling_mode not in {"encoded", "gaussian"}:
        raise ValueError("sampling_mode debe ser 'encoded' o 'gaussian'.")

    tasks: list[dict] = []

    for checkpoint_str in ["checkpoint", "scratch"]:
        for beta_str in ["beta_0.001", "no_beta"]:
            for source, target in pairs:
                model_a_path = f"inference_models/instruments_from_{checkpoint_str}/{source}_from_{checkpoint_str}_{beta_str}/version_0"
                model_b_path = f"inference_models/instruments_from_{checkpoint_str}/{target}_from_{checkpoint_str}_{beta_str}/version_0"

                output_dir = f"AudiosInterpolacion-Nuevos-Linear/{beta_str}/{checkpoint_str}/interpolation_{source}_to_{target}/"

                tasks.append(
                    {
                        "source": source,
                        "target": target,
                        "checkpoint_str": checkpoint_str,
                        "beta_str": beta_str,
                        "model_a_path": model_a_path,
                        "model_b_path": model_b_path,
                        "output_dir": output_dir,
                        "sampling_mode": sampling_mode,
                        "duration_seconds": duration_seconds,
                        "random_seed": random_seed,
                        "model_a_xmax": xmax_values[source],
                        "model_b_xmax": xmax_values[target],
                        "start_alpha": 0.0,
                        "end_alpha": 1.0,
                        "start_z": start_z_values[source],
                        "end_z": start_z_values[target],
                    }
                )

    total = len(tasks)
    if total == 0:
        print("No hay tareas para ejecutar.")
        return

    if max_workers is None:
        # Keep default conservative for heavy audio/Torch workloads.
        workers = min(total, max(1, min(4, os.cpu_count() or 1)))
    else:
        workers = max(1, min(max_workers, total))

    print(f"Lanzando {total} interpolaciones en paralelo con {workers} workers...")
    failures = []
    completed = 0

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_cross_interpolation_worker, task) for task in tasks]
        for future in as_completed(futures):
            label, ok, error_msg = future.result()
            completed += 1
            if ok:
                print(f"[{completed}/{total}] OK   {label}")
            else:
                failures.append((label, error_msg))
                print(f"[{completed}/{total}] FAIL {label}: {error_msg}")

    print(f"Interpolaciones completadas: {completed - len(failures)}/{total}")
    if failures:
        print("Fallos detectados:")
        for label, error_msg in failures:
            print(f" - {label}: {error_msg}")


if __name__ == "__main__":
    instrument_a = "voice"
    instrument_b = "guitar"
    beta = False
    sampling_mode = 1  # 0 para encoded, 1 para gaussian

    xmax_values = {
        "guitar": 130.0,
        "piano": 150.0,
        "voice": 140.0,
        "bass": 120.0,
    }

    Z_INIT_BY_INSTRUMENT = {
        "piano": [-0.00056, 0.10692, -0.02252, -0.09278],
        "guitar": [-0.00103, 0.1799, -0.11215, -0.21214],
        "voice": [0.00184, 0.05723, 0.19695, 0.62355],
        "bass": [7e-05, 0.14084, -2e-05, 0.06244],
    }

    model_a_path = f"inference_models/instruments_from_checkpoint/{instrument_a}_from_checkpoint_{'beta_0.001' if beta else 'no_beta'}/version_0"
    model_b_path = f"inference_models/instruments_from_checkpoint/{instrument_b}_from_checkpoint_{'beta_0.001' if beta else 'no_beta'}/version_0"

    output_dir = f"interpolation_audio_{'encoded' if sampling_mode == 0 else 'gaussian'}_{instrument_a}_to_{instrument_b}_{'beta_0.001' if beta else 'no_beta'}/"

    # run_interpolation_latent_experiment(
    #     model_a_path,
    #     model_b_path,
    #     output_dir,
    #     sampling_mode="encoded" if sampling_mode == 0 else "gaussian",
    #     duration_seconds=15.0,
    #     random_seed=42,
    #     model_a_xmax=xmax_values[instrument_a],
    #     model_b_xmax=xmax_values[instrument_b],
    #     start_alpha=0.0,
    #     end_alpha=1.0,
    #     start_z=Z_INIT_BY_INSTRUMENT[instrument_a],
    #     end_z=Z_INIT_BY_INSTRUMENT[instrument_b],
    # )

    cross_interpolation(
        random_seed=42,
        sampling_mode="encoded" if sampling_mode == 0 else "gaussian",
        xmax_values=xmax_values,
        start_z_values=Z_INIT_BY_INSTRUMENT,
        duration_seconds=20.0,
        max_workers=4,
    )

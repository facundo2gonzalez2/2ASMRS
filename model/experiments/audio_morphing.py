from pathlib import Path
import sys
import yaml
import torch
import numpy as np
import soundfile as sf

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from VariationalAutoEncoder import VariationalAutoEncoder
from scripts.vae_predict import predict_audio
from audio_utils import get_spectrograms_from_audios
from experiments.interpolate import interpolar_vae


def _load_instrument_model(instrument, source, beta):
    model_dir = (
        MODEL_DIR / f"inference_models/instruments_from_{source}" / f"{instrument}_from_{source}_{beta}" / "version_0"
    )
    checkpoint_path = list((model_dir / "checkpoints").glob("*.ckpt"))[0]
    with open(model_dir / "hparams.yaml") as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)

    model = VariationalAutoEncoder(
        encoder_layers=hps["encoder_layers"],
        decoder_layers=hps["decoder_layers"],
        latent_dim=hps["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    model.eval()
    model.decoder.eval()
    return model, hps


def _encode_audio_frames(model, hps, audio_path):
    X, _, _, _ = get_spectrograms_from_audios(
        [Path(audio_path)],
        hps["target_sampling_rate"],
        hps["win_length"],
        hps["hop_length"],
        db_min_norm=hps["db_min_norm"],
        spec_in_db=hps["spec_in_db"],
        normalize_each_audio=hps["normalize_each_audio"],
    )
    with torch.no_grad():
        mu, _ = model.encoder(X)
    return mu, hps["Xmax"]


def _discover_audio(instrument):
    for data_dir in ["data_instruments", "data_instruments_small"]:
        folder = MODEL_DIR / data_dir / instrument
        if not folder.exists():
            continue
        for ext in ("*.wav", "*.mp3"):
            files = sorted(folder.glob(ext))
            if files:
                return files[0]
    raise FileNotFoundError(f"No se encontro audio para '{instrument}' en data_instruments/ ni data_instruments_small/")


def _autogain(audio, target_peak=0.9):
    peak = float(np.max(np.abs(audio)))

    gain = target_peak / peak
    audio = np.clip(audio * gain, -1.0, 1.0)
    print(f"Auto-gain aplicado: gain={gain:.2f}")

    return audio


def main():
    # ── Config ──────────────────────────────────────────
    instrument_a = "piano"
    instrument_b = "guitar"
    audio_a = "data_audio_morphing/piano_new2_trimmed.wav"
    audio_b = "data_audio_morphing/guitar_new2_trimmed.wav"
    duration_a = 2  # seconds
    duration_transition = 5
    duration_b = 2
    source = "checkpoint"  # "checkpoint" or "scratch"
    beta = "beta_0.001"  # "beta_0.001" or "no_beta"
    interpolation_mode = "slerp"  # "linear" or "slerp"
    phase_reconstruction = "pghi"  # "griffinlim", "pghi", or "random"
    output_dir = f"audio_morphing_output_test_{phase_reconstruction}"
    # ────────────────────────────────────────────────────

    if audio_a is None:
        audio_a = _discover_audio(instrument_a)
    if audio_b is None:
        audio_b = _discover_audio(instrument_b)
    print(f"Audio A: {audio_a}")
    print(f"Audio B: {audio_b}")

    output_path = MODEL_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo {instrument_a}...")
    model_a, hps_a = _load_instrument_model(instrument_a, source, beta)
    print(f"Cargando modelo {instrument_b}...")
    model_b, hps_b = _load_instrument_model(instrument_b, source, beta)

    assert hps_a["encoder_layers"] == hps_b["encoder_layers"], "Arquitecturas no coinciden"
    assert hps_a["decoder_layers"] == hps_b["decoder_layers"], "Arquitecturas no coinciden"
    assert hps_a["latent_dim"] == hps_b["latent_dim"], "Arquitecturas no coinciden"

    print(f"Codificando audio A ({instrument_a})...")
    Z_a, Xmax_a = _encode_audio_frames(model_a, hps_a, audio_a)
    print(f"  Z_a shape: {Z_a.shape}, Xmax_a: {Xmax_a:.2f}")

    print(f"Codificando audio B ({instrument_b})...")
    Z_b, Xmax_b = _encode_audio_frames(model_b, hps_b, audio_b)
    print(f"  Z_b shape: {Z_b.shape}, Xmax_b: {Xmax_b:.2f}")

    fps = hps_a["target_sampling_rate"] / hps_a["hop_length"]
    n_a = int(round(duration_a * fps))
    n_transition = int(round(duration_transition * fps))
    n_b = int(round(duration_b * fps))

    if n_a > Z_a.shape[0]:
        print(
            f"[WARN] Audio A tiene {Z_a.shape[0]} frames ({Z_a.shape[0]/fps:.1f}s), "
            f"se pidieron {n_a} ({duration_a}s). Usando frames disponibles."
        )
        n_a = Z_a.shape[0]

    # Transition uses Z_a[n_a:n_a+n_transition] and Z_b[0:n_transition].
    # Phase 3 uses Z_b[n_transition:n_transition+n_b].
    # Check we have enough frames from both audios.
    needed_a = n_a + n_transition
    if needed_a > Z_a.shape[0]:
        n_transition_a = Z_a.shape[0] - n_a
        print(
            f"[WARN] Audio A tiene {Z_a.shape[0]} frames, se necesitan {needed_a} "
            f"(fase1+transicion). Transicion usara {n_transition_a} frames de A "
            f"y clampeara el resto."
        )

    needed_b = n_transition + n_b
    if needed_b > Z_b.shape[0]:
        print(
            f"[WARN] Audio B tiene {Z_b.shape[0]} frames ({Z_b.shape[0]/fps:.1f}s), "
            f"se necesitan {needed_b} (transicion+fase3). Se clampeara al ultimo frame disponible."
        )

    total_frames = n_a + n_transition + n_b
    print(f"Frames: {n_a} (A) + {n_transition} (transicion) + {n_b} (B) = {total_frames} " f"({total_frames/fps:.1f}s)")

    # Phase 1: Pure A
    print("Fase 1: decodificando instrumento A...")
    with torch.no_grad():
        raw_a = model_a.decoder(Z_a[:n_a])
        print(
            f"  [diag] Phase 1 decoder out: mean={raw_a.mean():.4f}, min={raw_a.min():.4f}, "
            f"max={raw_a.max():.4f}, nans={torch.isnan(raw_a).sum().item()}"
        )
        spec_a = raw_a * Xmax_a
        spec_a = torch.clamp(
            torch.nan_to_num(spec_a, nan=0.0, posinf=Xmax_a, neginf=0.0),
            min=0.0,
            max=Xmax_a,
        )

    # Phase 2: Transition
    # Use sequential Z frames from both audios: A continues playing, B starts entering.
    # Z_a[n_a + i] blended with Z_b[i], weighted by alpha.
    print("Fase 2: transicion...")
    transition_frames = []

    for i in range(n_transition):
        alpha = i / max(n_transition - 1, 1)
        xmax_i = (1.0 - alpha) * Xmax_a + alpha * Xmax_b

        idx_a = min(n_a + i, Z_a.shape[0] - 1)
        idx_b = min(i, Z_b.shape[0] - 1)
        z_i = (1.0 - alpha) * Z_a[idx_a : idx_a + 1] + alpha * Z_b[idx_b : idx_b + 1]

        model_i = interpolar_vae(
            model_a,
            model_b,
            alpha,
            encoder_layers=hps_a["encoder_layers"],
            decoder_layers=hps_a["decoder_layers"],
            latent_dim=hps_a["latent_dim"],
            interpolation_mode=interpolation_mode,
        )
        model_i.eval()

        with torch.no_grad():
            raw_out = model_i.decoder(z_i)
            if i % max(n_transition // 5, 1) == 0:
                print(
                    f"  [diag] frame {i}/{n_transition}, alpha={alpha:.3f}, "
                    f"decoder out: mean={raw_out.mean():.4f}, min={raw_out.min():.4f}, "
                    f"max={raw_out.max():.4f}, nans={torch.isnan(raw_out).sum().item()}"
                )
            spec_i = raw_out * xmax_i
            spec_i = torch.nan_to_num(spec_i, nan=0.0, posinf=xmax_i, neginf=0.0)
            spec_i = torch.clamp(spec_i, min=0.0, max=xmax_i)
            transition_frames.append(spec_i)

    # Phase 3: Pure B (continues from where transition's B-side left off)
    print("Fase 3: decodificando instrumento B...")
    phase3_start = min(n_transition, Z_b.shape[0] - 1)
    phase3_end = min(n_transition + n_b, Z_b.shape[0])
    actual_n_b = phase3_end - phase3_start
    if actual_n_b < n_b:
        # Pad with last available frame if audio B is too short
        Z_b_phase3 = torch.cat(
            [
                Z_b[phase3_start:phase3_end],
                Z_b[-1:].expand(n_b - actual_n_b, -1),
            ],
            dim=0,
        )
    else:
        Z_b_phase3 = Z_b[phase3_start:phase3_end]
    with torch.no_grad():
        raw_b = model_b.decoder(Z_b_phase3)
        print(
            f"  [diag] Phase 3 decoder out: mean={raw_b.mean():.4f}, min={raw_b.min():.4f}, "
            f"max={raw_b.max():.4f}, nans={torch.isnan(raw_b).sum().item()}"
        )
        spec_b = raw_b * Xmax_b
        spec_b = torch.clamp(
            torch.nan_to_num(spec_b, nan=0.0, posinf=Xmax_b, neginf=0.0),
            min=0.0,
            max=Xmax_b,
        )

    # Concatenate and reconstruct audio
    print(f"Reconstruyendo audio con {phase_reconstruction}...")
    specgram = torch.cat([spec_a] + transition_frames + [spec_b], dim=0)

    try:
        audio = predict_audio(
            predicted_specgram=specgram,
            hps=hps_a,
            phase_option=phase_reconstruction,
            frames=total_frames,
            return_audio=True,
        )
    except Exception as err:
        print(f"[WARN] {phase_reconstruction} fallo ({err}). Reintentando con phase='random'.")
        audio = predict_audio(
            predicted_specgram=specgram,
            hps=hps_a,
            phase_option="random",
            frames=total_frames,
            return_audio=True,
        )

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
    audio = _autogain(audio)

    wav_name = f"morphing_{instrument_a}_to_{instrument_b}.wav"
    wav_path = output_path / wav_name
    sf.write(str(wav_path), audio, hps_a["target_sampling_rate"])
    print(f"Audio guardado en: {wav_path}")

    metadata = {
        "instrument_a": instrument_a,
        "instrument_b": instrument_b,
        "audio_a": str(audio_a),
        "audio_b": str(audio_b),
        "duration_a_requested": duration_a,
        "duration_transition_requested": duration_transition,
        "duration_b_requested": duration_b,
        "frames_a": n_a,
        "frames_transition": n_transition,
        "frames_b": n_b,
        "total_frames": total_frames,
        "total_duration": float(total_frames / fps),
        "Xmax_a": float(Xmax_a),
        "Xmax_b": float(Xmax_b),
        "source": source,
        "beta": beta,
        "interpolation_mode": interpolation_mode,
        "phase_reconstruction": phase_reconstruction,
    }
    meta_path = output_path / f"morphing_{instrument_a}_to_{instrument_b}_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f"Metadata guardada en: {meta_path}")


if __name__ == "__main__":
    main()

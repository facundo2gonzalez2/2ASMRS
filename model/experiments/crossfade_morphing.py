from pathlib import Path
import sys
import yaml
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audio_utils import get_specgram, get_waveform
from experiments.audio_morphing import (
    _autogain,
    _build_morphing_specgram,
    _encode_audio_frames,
    _frame_counts,
    _load_instrument_model,
    _specgram_to_audio,
)


def decode_specgram(model, z, xmax):
    with torch.no_grad():
        spec = model.decoder(z) * xmax
    return torch.clamp(torch.nan_to_num(spec, nan=0.0, posinf=xmax, neginf=0.0), min=0.0, max=xmax)


def crossfade_specgrams(spec_a, spec_b, alpha, db_min_norm, domain="linear"):
    """
    Cross-fade entre espectrogramas en la escala del decoder (dB por encima de db_min_norm).

    domain="linear": mezcla las magnitudes lineales, equivale a sumar las dos señales (cross-fade clásico).
    domain="db": promedia directamente la representación del modelo (media geométrica de magnitudes).
    """
    if domain == "db":
        return (1.0 - alpha) * spec_a + alpha * spec_b
    if domain != "linear":
        raise ValueError(f"crossfade_domain inválido: {domain}. Usa 'linear' o 'db'.")

    mag_a = torch.pow(10.0, (spec_a + db_min_norm) / 20.0)
    mag_b = torch.pow(10.0, (spec_b + db_min_norm) / 20.0)
    mag = (1.0 - alpha) * mag_a + alpha * mag_b
    return torch.clamp(20.0 * torch.log10(mag) - db_min_norm, min=0.0)


def _fit_length(audio, n_samples):
    if audio.shape[0] >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - audio.shape[0]))


def _crossfade_waveforms(audio_a, audio_b, sr, hop_length, n_a, n_transition, n_b):
    # Mismo timeline que el morphing: A suena desde t=0 y B entra al comenzar la transición.
    total_samples = (n_a + n_transition + n_b) * hop_length
    start = n_a * hop_length
    wave_a = get_waveform(audio_a, sr).numpy()
    wave_b = get_waveform(audio_b, sr).numpy()
    wave_a = _fit_length(wave_a / np.max(np.abs(wave_a)), total_samples)
    wave_b = np.concatenate([np.zeros(start), _fit_length(wave_b / np.max(np.abs(wave_b)), total_samples - start)])
    gain_b = np.clip((np.arange(total_samples) - start) / max(n_transition * hop_length - 1, 1), 0.0, 1.0)
    return (1.0 - gain_b) * wave_a + gain_b * wave_b


def _plot_spectrograms(audios, hps, transition_bounds, max_freq, title, output_path):
    hop, win, sr = hps["hop_length"], hps["win_length"], hps["target_sampling_rate"]
    fig, axes = plt.subplots(len(audios), 1, figsize=(12, 3 * len(audios)), sharex=True, squeeze=False)

    for axis, (label, audio) in zip(axes[:, 0], audios.items()):
        _, S = get_specgram(torch.as_tensor(audio, dtype=torch.float32), win, hop)
        S = S.clip(S.max() + hps["db_min_norm"], None).numpy()
        extent = (0.0, S.shape[0] * hop / sr, 0.0, sr / 2)
        axis.imshow(S.T, origin="lower", aspect="auto", extent=extent, cmap="magma")
        for t in transition_bounds:
            axis.axvline(t, color="white", linestyle="--", linewidth=1.2)
        axis.set_ylim(0, max_freq)
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title(label)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    # ── Config ──────────────────────────────────────────
    instrument_a = "voice"
    instrument_b = "guitar"
    # audio_a = MODEL_DIR / "data_audio_morphing/piano_new2_trimmed.wav"
    # audio_a = MODEL_DIR / "data_audio_morphing/freesound_community-piano-chord-45626_trimmed.mp3"
    # audio_b = MODEL_DIR / "data_audio_morphing/guitar_new2_trimmed.wav"
    audio_a = MODEL_DIR / "data_instruments/voice/f1_arpeggios_belt_c_a.wav"
    audio_b = MODEL_DIR / "data_instruments/guitar/00_BN1-129-Eb_solo_hex.wav"
    # audio_b = MODEL_DIR / "data_audio_morphing/guitar.mp3"
    duration_a = 2  # seconds
    duration_transition = 8
    duration_b = 2
    source = "checkpoint"  # "checkpoint" or "scratch"
    beta = "beta_0.001"  # "beta_0.001" or "no_beta"
    interpolation_mode = "slerp"  # "linear" or "slerp"
    phase_reconstruction = "pghi"  # "griffinlim", "pghi", or "random"
    crossfade_domain = "linear"  # "linear" (mezcla de magnitudes) o "db" (promedio de la representación del modelo)
    max_freq_plot = 4000  # Hz
    output_dir = f"crossfade_morphing_output_{phase_reconstruction}_2"
    # ────────────────────────────────────────────────────

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

    Z_a, Xmax_a = _encode_audio_frames(model_a, hps_a, audio_a)
    Z_b, Xmax_b = _encode_audio_frames(model_b, hps_b, audio_b)

    fps, n_a, n_transition, n_b = _frame_counts(hps_a, Z_a, Z_b, duration_a, duration_transition, duration_b)
    total_frames = n_a + n_transition + n_b
    print(f"Frames: {n_a} (A) + {n_transition} (transicion) + {n_b} (B) = {total_frames} " f"({total_frames/fps:.1f}s)")

    # 1) Nuestro método: exactamente el mismo procedimiento que audio_morphing.py
    morph_spec = _build_morphing_specgram(
        model_a, model_b, hps_a, Z_a, Z_b, Xmax_a, Xmax_b, n_a, n_transition, n_b, interpolation_mode
    )

    # 2) Cross-fade de las reconstrucciones: fases A y B idénticas al morphing, solo cambia la transición.
    #    En cada frame se mezclan las salidas de los modelos puros en lugar de interpolar Z y pesos.
    print(f"Cross-fade de espectrogramas (dominio {crossfade_domain})...")
    idx_a = torch.clamp(torch.arange(n_a, n_a + n_transition), max=Z_a.shape[0] - 1)
    idx_b = torch.clamp(torch.arange(n_transition), max=Z_b.shape[0] - 1)
    alphas = (torch.arange(n_transition, dtype=torch.float32) / max(n_transition - 1, 1))[:, None]
    transition = crossfade_specgrams(
        decode_specgram(model_a, Z_a[idx_a], Xmax_a),
        decode_specgram(model_b, Z_b[idx_b], Xmax_b),
        alphas,
        hps_a["db_min_norm"],
        crossfade_domain,
    )
    crossfade_spec = torch.cat([morph_spec[:n_a], transition, morph_spec[n_a + n_transition :]], dim=0)

    audios = {
        "crossfade_original": _autogain(
            _crossfade_waveforms(
                audio_a, audio_b, hps_a["target_sampling_rate"], hps_a["hop_length"], n_a, n_transition, n_b
            )
        ),
        "crossfade": _specgram_to_audio(crossfade_spec, hps_a, phase_reconstruction),
        "morphing": _specgram_to_audio(morph_spec, hps_a, phase_reconstruction),
    }

    for name, audio in audios.items():
        wav_path = output_path / f"{name}_{instrument_a}_to_{instrument_b}.wav"
        sf.write(str(wav_path), audio, hps_a["target_sampling_rate"])
        print(f"Audio guardado en: {wav_path}")

    plot_path = output_path / f"spectrograms_{instrument_a}_to_{instrument_b}.png"
    _plot_spectrograms(
        {
            "Waveform cross-fade (original audio)": audios["crossfade_original"],
            f"Spectrogram cross-fade of VAE reconstructions ({crossfade_domain})": audios["crossfade"],
            f"VAE morphing (ours, {interpolation_mode})": audios["morphing"],
        },
        hps_a,
        (n_a / fps, (n_a + n_transition) / fps),
        max_freq_plot,
        f"{instrument_a} → {instrument_b}",
        plot_path,
    )
    print(f"Espectrogramas guardados en: {plot_path}")

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
        "crossfade_domain": crossfade_domain,
    }
    meta_path = output_path / f"crossfade_morphing_{instrument_a}_to_{instrument_b}_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f"Metadata guardada en: {meta_path}")


if __name__ == "__main__":
    main()

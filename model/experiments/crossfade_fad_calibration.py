import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audio_utils import get_waveform  # noqa: E402
from experiments.crossfade_fad import read_rows, write_rows  # noqa: E402
from experiments.interpolate import FAMILY_STYLES, collect_audio_files  # noqa: E402

# MERT normaliza cada clip (do_normalize), así que el nivel absoluto no afecta la métrica. Igualamos el RMS de
# los segmentos para que la mezcla esté balanceada y fijamos -20 dBFS para que los WAV sean cómodos de escuchar.
TARGET_RMS = 0.1

INSTRUMENT_LABELS = {"piano": "piano", "guitar": "guitarra", "voice": "voz", "bass": "bajo"}


def _sample_segment(files, sr, segment_samples, rng):
    """Elige un archivo al azar y una ventana de segment_samples muestras, descartando silencios."""
    for _ in range(50):
        path = files[int(rng.integers(len(files)))]
        wave = get_waveform(path, sr, remove_all_silence=True).numpy()
        if wave.shape[0] < segment_samples:
            continue
        start = int(rng.integers(wave.shape[0] - segment_samples + 1))
        segment = wave[start : start + segment_samples]
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0:
            return segment * (TARGET_RMS / rms), path, start
    raise ValueError(f"No se encontraron audios con al menos {segment_samples} muestras en {files[0].parent}")


def _write_wav(wave, sr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sr, subtype="FLOAT")  # FLOAT para no recortar picos por encima de 1
    return path


def compute_fad_rows(pairs, num_samples, segment_samples, alphas, sr, data_root, output_root, seed):
    from audio_comparator import get_audio_similarity_fad, get_matrix_embedding

    rng = np.random.default_rng(seed)
    rows = []

    for instrument_a, instrument_b, instrument_w in pairs:
        print(f"\n══════ {instrument_a} → {instrument_b} (testigo: {instrument_w}) ══════")
        instruments = {"source": instrument_a, "target": instrument_b, "witness": instrument_w}
        files = {inst: collect_audio_files(data_root, inst) for inst in instruments.values()}
        for inst, inst_files in files.items():
            if not inst_files:
                raise FileNotFoundError(f"No hay audios en {data_root / inst}")

        pair_dir = output_root / f"{instrument_a}_to_{instrument_b}"
        for s in range(num_samples):
            sample_dir = pair_dir / f"sample_{s:02d}"
            print(f"\n── Sample {s + 1}/{num_samples} ──")

            # Referencias: los segmentos originales, sin pasar por ningún modelo.
            segments, waves, ref_mats = {}, {}, {}
            for family, inst in instruments.items():
                waves[family], path, start = _sample_segment(files[inst], sr, segment_samples, rng)
                print(f"  {family:8s} {inst:6s} {path.name} (muestra {start})")
                segments[family] = (path, start)
                ref_wav = _write_wav(waves[family], sr, sample_dir / f"ref_{family}.wav")
                ref_mats[family] = get_matrix_embedding(str(ref_wav))

            for alpha in alphas:
                mix = (1.0 - alpha) * waves["source"] + alpha * waves["target"]
                wav = _write_wav(mix, sr, sample_dir / f"crossfade_alpha_{alpha:.2f}.wav")
                mat = get_matrix_embedding(str(wav))
                row = {
                    "source_instrument": instrument_a,
                    "target_instrument": instrument_b,
                    "witness_instrument": instrument_w,
                    "sample_idx": s,
                    "alpha": alpha,
                    "audio": str(wav),
                }
                for family, (path, start) in segments.items():
                    row[f"fad_{family}"] = float(get_audio_similarity_fad(ref_mats[family], mat))
                    row[f"{family}_audio"] = str(path)
                    row[f"{family}_start_sample"] = start
                rows.append(row)
                print(
                    f"  α={alpha:.2f} fad_source={row['fad_source']:.4f} "
                    f"fad_target={row['fad_target']:.4f} fad_witness={row['fad_witness']:.4f}"
                )

    return rows


def _pair_keys(rows):
    return list(dict.fromkeys((r["source_instrument"], r["target_instrument"], r["witness_instrument"]) for r in rows))


def _pair_curves(rows, pair):
    """Devuelve los α y, por familia, una matriz (muestras × α) de similitudes FAD."""
    pair_rows = [r for r in rows if (r["source_instrument"], r["target_instrument"], r["witness_instrument"]) == pair]
    alphas = sorted({r["alpha"] for r in pair_rows})
    samples = sorted({r["sample_idx"] for r in pair_rows})
    values = {(r["sample_idx"], r["alpha"]): r for r in pair_rows}
    curves = {
        family: np.array([[values[(s, a)][f"fad_{family}"] for a in alphas] for s in samples])
        for family, _ in FAMILY_STYLES
    }
    return alphas, curves


def print_summary(rows):
    for pair in _pair_keys(rows):
        alphas, curves = _pair_curves(rows, pair)
        print(f"\n{pair[0]} → {pair[1]} (testigo: {pair[2]}), media ± std sobre {len(curves['source'])} muestras")
        print("  α      " + "".join(f"{family:>16s}" for family, _ in FAMILY_STYLES))
        for i, alpha in enumerate(alphas):
            cells = "".join(f"{curves[f][:, i].mean():9.3f} ± {curves[f][:, i].std():.3f}" for f, _ in FAMILY_STYLES)
            print(f"  {alpha:.2f} {cells}")


def plot_fad_curves(rows, output_root, show_sample_curves):
    pairs = _pair_keys(rows)
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 4.5), sharey=True, squeeze=False)

    for axis, pair in zip(axes[0], pairs):
        alphas, curves = _pair_curves(rows, pair)
        name_a, name_b, name_w = (INSTRUMENT_LABELS.get(inst, inst) for inst in pair)
        labels = {"source": name_a, "target": name_b, "witness": f"{name_w} (testigo)"}

        for family, color in FAMILY_STYLES:
            if show_sample_curves:
                for curve in curves[family]:
                    axis.plot(alphas, curve, color=color, linewidth=0.8, alpha=0.15)
            mean, std = curves[family].mean(axis=0), curves[family].std(axis=0)
            axis.plot(alphas, mean, color=color, marker="o", linewidth=2, label=labels[family])
            axis.fill_between(alphas, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

        axis.set_title(f"{name_a} → {name_b} ({len(curves['source'])} pares de muestras)")
        axis.set_xlabel("α")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[0][0].set_ylabel("Similitud (FAD)")
    fig.suptitle("Similitud (FAD) | cross-fade de audios originales (1 − α)·A + α·B", fontsize=13)
    fig.tight_layout()

    plot_path = output_root / "plots" / "fad_crossfade_calibration.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nGráfico guardado en: {plot_path}")


def main():
    # ── Config ──────────────────────────────────────────
    pairs = [  # (origen A, destino B, testigo)
        ("piano", "guitar", "voice"),
        ("voice", "piano", "guitar"),
    ]
    num_samples = 20  # pares de segmentos (A, B) por par de instrumentos
    num_frames = 64  # ~1.5 s, igual que en los experimentos de interpolación
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_rate = 22050  # misma frecuencia y hop que los modelos, para que la duración coincida
    hop_length = 512
    show_sample_curves = True  # dibuja cada par de segmentos además de la media ± std
    plot_only = False  # True: regenera el gráfico y el resumen desde el CSV sin recalcular
    data_root = MODEL_DIR / "data_instruments"
    output_root = MODEL_DIR / "crossfade_fad_calibration_output"
    seed = 0
    # ────────────────────────────────────────────────────

    raw_csv = output_root / "crossfade_fad_calibration_raw_metrics.csv"
    if plot_only:
        rows = read_rows(raw_csv)
    else:
        rows = compute_fad_rows(
            pairs=pairs,
            num_samples=num_samples,
            segment_samples=num_frames * hop_length,
            alphas=alphas,
            sr=sample_rate,
            data_root=data_root,
            output_root=output_root,
            seed=seed,
        )
        write_rows(rows, raw_csv)
        print(f"\nMétricas guardadas en: {raw_csv}")

    print_summary(rows)
    plot_fad_curves(rows, output_root, show_sample_curves)


if __name__ == "__main__":
    main()

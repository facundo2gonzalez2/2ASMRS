import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audio_utils import get_spectrograms_from_audios  # noqa: E402
from experiments.audio_morphing import _load_instrument_model  # noqa: E402
from experiments.crossfade_morphing import crossfade_specgrams, decode_specgram  # noqa: E402
from experiments.interpolate import FAMILY_STYLES, collect_audio_files, interpolar_vae  # noqa: E402
from scripts.vae_predict import predict_audio  # noqa: E402

METHOD_TITLES = {
    "crossfade": "Spectrogram cross-fade (baseline)",
    "morphing": "VAE morphing (ours)",
}


def _sample_segment(files, hps, num_frames, rng):
    """Elige un archivo al azar y una ventana de num_frames frames, descartando silencios."""
    for _ in range(50):
        path = files[int(rng.integers(len(files)))]
        X, _, _, _ = get_spectrograms_from_audios(
            [path],
            hps["target_sampling_rate"],
            hps["win_length"],
            hps["hop_length"],
            db_min_norm=hps["db_min_norm"],
            spec_in_db=hps["spec_in_db"],
            normalize_each_audio=hps["normalize_each_audio"],
            remove_all_silence=True,
        )
        if X.shape[0] >= num_frames:
            start = int(rng.integers(X.shape[0] - num_frames + 1))
            return X[start : start + num_frames], path, start
    raise ValueError(f"No se encontraron audios con al menos {num_frames} frames en {files[0].parent}")


def _encode(model, X):
    with torch.no_grad():
        mu, _ = model.encoder(X)
    return mu


def _write_wav(spec, hps, phase_reconstruction, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = predict_audio(
        predicted_specgram=spec,
        hps=hps,
        phase_option=phase_reconstruction,
        frames=spec.shape[0],
        return_audio=True,
    )
    sf.write(str(path), np.nan_to_num(audio), hps["target_sampling_rate"])  # type: ignore
    return path


def compute_fad_rows(
    pairs,
    num_samples,
    num_frames,
    alphas,
    source,
    beta,
    interpolation_mode,
    phase_reconstruction,
    crossfade_domain,
    include_morphing,
    data_root,
    output_root,
    seed,
):
    from audio_comparator import get_audio_similarity_fad, get_matrix_embedding

    rng = np.random.default_rng(seed)
    rows = []

    for instrument_a, instrument_b, instrument_w in pairs:
        print(f"\n══════ {instrument_a} → {instrument_b} (testigo: {instrument_w}) ══════")
        instruments = {"source": instrument_a, "target": instrument_b, "witness": instrument_w}
        models = {inst: _load_instrument_model(inst, source, beta) for inst in instruments.values()}
        files = {inst: collect_audio_files(data_root, inst) for inst in instruments.values()}
        (model_a, hps_a), (model_b, hps_b) = models[instrument_a], models[instrument_b]
        xmax_a, xmax_b = hps_a["Xmax"], hps_b["Xmax"]

        interpolated = {}
        if include_morphing:
            for alpha in alphas:
                interpolated[alpha] = interpolar_vae(
                    model_a,
                    model_b,
                    alpha,
                    encoder_layers=hps_a["encoder_layers"],
                    decoder_layers=hps_a["decoder_layers"],
                    latent_dim=hps_a["latent_dim"],
                    interpolation_mode=interpolation_mode,
                ).eval()

        pair_dir = output_root / f"{instrument_a}_to_{instrument_b}"
        for s in range(num_samples):
            sample_dir = pair_dir / f"sample_{s:02d}"
            print(f"\n── Sample {s + 1}/{num_samples} ──")

            # Referencias: reconstrucción de cada segmento con su modelo puro. Así α=0 y α=1 coinciden
            # exactamente con la referencia en ambos métodos y solo se compara el camino intermedio.
            segments, zs, ref_specs, ref_mats = {}, {}, {}, {}
            for family, inst in instruments.items():
                model, hps = models[inst]
                X, path, start = _sample_segment(files[inst], hps, num_frames, rng)
                print(f"  {family:8s} {inst:6s} {path.name} (frame {start})")
                segments[family] = (path, start)
                zs[family] = _encode(model, X)
                ref_specs[family] = decode_specgram(model, zs[family], hps["Xmax"])
                ref_wav = _write_wav(ref_specs[family], hps, phase_reconstruction, sample_dir / f"ref_{family}.wav")
                ref_mats[family] = get_matrix_embedding(str(ref_wav))

            for alpha in alphas:
                outputs = {
                    "crossfade": crossfade_specgrams(
                        ref_specs["source"], ref_specs["target"], alpha, hps_a["db_min_norm"], crossfade_domain
                    )
                }
                if include_morphing:
                    z = (1.0 - alpha) * zs["source"] + alpha * zs["target"]
                    outputs["morphing"] = decode_specgram(
                        interpolated[alpha], z, (1.0 - alpha) * xmax_a + alpha * xmax_b
                    )

                for method, spec in outputs.items():
                    wav = _write_wav(spec, hps_a, phase_reconstruction, sample_dir / f"{method}_alpha_{alpha:.2f}.wav")
                    mat = get_matrix_embedding(str(wav))
                    row = {
                        "source_instrument": instrument_a,
                        "target_instrument": instrument_b,
                        "witness_instrument": instrument_w,
                        "method": method,
                        "sample_idx": s,
                        "alpha": alpha,
                        "audio": str(wav),
                    }
                    for family, (path, start) in segments.items():
                        row[f"fad_{family}"] = float(get_audio_similarity_fad(ref_mats[family], mat))
                        row[f"{family}_audio"] = str(path)
                        row[f"{family}_start_frame"] = start
                    rows.append(row)
                    print(
                        f"  α={alpha:.2f} {method:9s} fad_source={row['fad_source']:.4f} "
                        f"fad_target={row['fad_target']:.4f} fad_witness={row['fad_witness']:.4f}"
                    )

    return rows


def write_rows(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(input_csv):
    with open(input_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["alpha"] = float(row["alpha"])
        row["sample_idx"] = int(row["sample_idx"])
        for family, _ in FAMILY_STYLES:
            row[f"fad_{family}"] = float(row[f"fad_{family}"])
    return rows


def plot_fad_curves(rows, output_root, show_sample_curves):
    pair_keys = sorted({(r["source_instrument"], r["target_instrument"], r["witness_instrument"]) for r in rows})

    for instrument_a, instrument_b, instrument_w in pair_keys:
        pair_rows = [
            r for r in rows if (r["source_instrument"], r["target_instrument"]) == (instrument_a, instrument_b)
        ]
        methods = [m for m in METHOD_TITLES if any(r["method"] == m for r in pair_rows)]
        labels = {"source": instrument_a, "target": instrument_b, "witness": f"{instrument_w} (witness)"}

        fig, axes = plt.subplots(1, len(methods), figsize=(6.5 * len(methods), 4.5), sharey=True, squeeze=False)
        for axis, method in zip(axes[0], methods):
            method_rows = [r for r in pair_rows if r["method"] == method]
            alphas = sorted({r["alpha"] for r in method_rows})
            samples = sorted({r["sample_idx"] for r in method_rows})
            values = {(r["sample_idx"], r["alpha"]): r for r in method_rows}

            for family, color in FAMILY_STYLES:
                curves = np.array([[values[(s, a)][f"fad_{family}"] for a in alphas] for s in samples])
                if show_sample_curves:
                    for curve in curves:
                        axis.plot(alphas, curve, color=color, linewidth=0.8, alpha=0.15)
                mean, std = curves.mean(axis=0), curves.std(axis=0)
                axis.plot(alphas, mean, color=color, marker="o", linewidth=2, label=labels[family])
                axis.fill_between(alphas, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

            axis.set_title(METHOD_TITLES[method])
            axis.set_xlabel("α")
            axis.grid(True, alpha=0.3)
            axis.legend()

        axes[0][0].set_ylabel("FAD similarity")
        fig.suptitle(f"FAD similarity | {instrument_a} → {instrument_b} ({len(samples)} sample pairs)", fontsize=13)
        fig.tight_layout()

        plot_path = output_root / "plots" / f"fad_crossfade_{instrument_a}_to_{instrument_b}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Gráfico guardado en: {plot_path}")


def main():
    # ── Config ──────────────────────────────────────────
    pairs = [  # (origen A, destino B, testigo)
        ("piano", "guitar", "voice"),
        ("voice", "piano", "guitar"),
    ]
    num_samples = 20  # pares de segmentos (A, B) por par de instrumentos
    num_frames = 64  # ~1.5 s, igual que en los experimentos de interpolación
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    source = "checkpoint"  # "checkpoint" or "scratch"
    beta = "beta_0.001"  # "beta_0.001" or "no_beta"
    interpolation_mode = "slerp"  # "linear" or "slerp"
    phase_reconstruction = "pghi"  # "griffinlim", "pghi", or "random"
    crossfade_domain = "linear"  # "linear" (mezcla de magnitudes) o "db" (promedio de la representación del modelo)
    include_morphing = True  # False: solo el baseline de cross-fade
    show_sample_curves = True  # dibuja cada par de segmentos además de la media ± std
    plot_only = False  # True: regenera los gráficos desde el CSV sin recalcular
    data_root = MODEL_DIR / "data_instruments"
    output_root = MODEL_DIR / f"crossfade_fad_output_{source}_{beta}"
    seed = 0
    # ────────────────────────────────────────────────────

    raw_csv = output_root / "crossfade_fad_raw_metrics.csv"
    if plot_only:
        rows = read_rows(raw_csv)
    else:
        rows = compute_fad_rows(
            pairs=pairs,
            num_samples=num_samples,
            num_frames=num_frames,
            alphas=alphas,
            source=source,
            beta=beta,
            interpolation_mode=interpolation_mode,
            phase_reconstruction=phase_reconstruction,
            crossfade_domain=crossfade_domain,
            include_morphing=include_morphing,
            data_root=data_root,
            output_root=output_root,
            seed=seed,
        )
        write_rows(rows, raw_csv)
        print(f"\nMétricas guardadas en: {raw_csv}")

    plot_fad_curves(rows, output_root, show_sample_curves)


if __name__ == "__main__":
    main()

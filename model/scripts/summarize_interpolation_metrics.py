"""Números de la tabla del paper LAMIR, a partir de resultados ya calculados (no entrena ni genera audio).

1. MSE de reconstrucción en validación (mejor época) de cada modelo, leído de los logs de TensorBoard.
2. Indicadores de la interpolación cruzada leídos del CSV de experiments/interpolate.py:
   - ρ de Spearman entre α y la similitud FAD a fuente / destino, por muestra de Z (media ± std).
   - rango (max − min) de la similitud al testigo a lo largo de α, por muestra (media ± std).
   - α de cruce entre las curvas medias de fuente y destino (interpolación lineal).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

REGIMES = {  # régimen del CSV → (directorio de modelos, sufijo)
    "checkpoint_beta": ("instruments_from_checkpoint", "from_checkpoint_beta_0.001"),
    "checkpoint_no_beta": ("instruments_from_checkpoint", "from_checkpoint_no_beta"),
    "scratch_beta": ("instruments_from_scratch", "from_scratch_beta_0.001"),
    "scratch_no_beta": ("instruments_from_scratch", "from_scratch_no_beta"),
}
INSTRUMENTS = ["bass", "guitar", "piano", "voice"]


def best_val_recon(log_dir):
    event_file = next(log_dir.glob("version_0/events.out.tfevents.*"))
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    return min(e.value for e in acc.Scalars("val_recon"))


def reconstruction_table(models_root):
    rows = []
    for regime, (subdir, suffix) in REGIMES.items():
        values = {inst: best_val_recon(models_root / subdir / f"{inst}_{suffix}") for inst in INSTRUMENTS}
        rows.append({"regime": regime, **values, "mean": np.mean(list(values.values()))})
    for name in ["base_model_beta_0.001", "base_model_no_beta"]:
        rows.append({"regime": name, "mean": best_val_recon(models_root / "base_model" / name)})
    return pd.DataFrame(rows)


def crossover_alpha(alphas, source_mean, target_mean):
    diff = target_mean - source_mean
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    return alphas[i] + (alphas[i + 1] - alphas[i]) * (-diff[i]) / (diff[i + 1] - diff[i])


def interpolation_table(raw_csv):
    df = pd.read_csv(raw_csv)
    rows = []
    for (regime, source, target), group in df.groupby(["regime", "source_instrument", "target_instrument"]):
        rho_source, rho_target, witness_range = [], [], []
        for _, sample in group.groupby("sample_idx"):
            sample = sample.sort_values("alpha")
            rho_source.append(spearmanr(sample.alpha, sample.fad_source).statistic)
            rho_target.append(spearmanr(sample.alpha, sample.fad_target).statistic)
            witness_range.append(sample.fad_witness.max() - sample.fad_witness.min())
        means = group.groupby("alpha")[["fad_source", "fad_target"]].mean().sort_index()
        rows.append(
            {
                "regime": regime,
                "pair": f"{source}->{target}",
                "rho_source": f"{np.mean(rho_source):+.2f}±{np.std(rho_source):.2f}",
                "rho_target": f"{np.mean(rho_target):+.2f}±{np.std(rho_target):.2f}",
                "witness_range": f"{np.mean(witness_range):.3f}±{np.std(witness_range):.3f}",
                "crossover_alpha": round(
                    crossover_alpha(means.index.values, means.fad_source.values, means.fad_target.values), 2
                ),
                "n_samples": group.sample_idx.nunique(),
            }
        )
    return pd.DataFrame(rows)


def main():
    # ── Config ──────────────────────────────────────────
    models_root = MODEL_DIR / "inference_models"
    raw_csv = MODEL_DIR / "interpolation_random_25_samples_ckpt_vs_scratch_gaussian_latest2" / "interpolation_raw_metrics.csv"
    # ────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    print("── Validation reconstruction MSE (best epoch) ──")
    print(reconstruction_table(models_root).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n── Cross-interpolation indicators (FAD similarity) ──")
    print(interpolation_table(raw_csv).to_string(index=False))


if __name__ == "__main__":
    main()

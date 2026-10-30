"""Figura de transición cruzada (paper LAMIR) en PDF vectorial a ancho de columna.

Lee el CSV resumen de experiments/interpolate.py (no recalcula métricas ni carga MERT) y dibuja una grilla 2×2:
filas = inicialización (Checkpoint / Scratch), columnas = regularización (β-VAE / AE), con eje y compartido.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

GRID = [  # (fila, columna) → régimen
    [("checkpoint_beta", "Checkpoint"), ("checkpoint_no_beta", None)],
    [("scratch_beta", "Scratch"), ("scratch_no_beta", None)],
]
COLUMN_TITLES = [r"$\beta$-VAE ($\beta=0.001$)", r"AE ($\beta=0$)"]
FAMILIES = [  # (familia, color, sufijo de leyenda)
    ("source", "tab:blue", "source A"),
    ("target", "tab:orange", "target B"),
    ("witness", "tab:green", "witness"),
]


def read_summary(path, source, target, metric):
    curves = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["source_instrument"] != source or row["target_instrument"] != target:
                continue
            curves.setdefault(row["regime"], []).append(row)
    out = {}
    for regime, rows in curves.items():
        rows.sort(key=lambda r: float(r["alpha"]))
        out[regime] = {
            "alpha": np.array([float(r["alpha"]) for r in rows]),
            "witness_instrument": rows[0]["witness_instrument"],
            **{
                fam: (
                    np.array([float(r[f"{metric}_{fam}_mean"]) for r in rows]),
                    np.array([float(r[f"{metric}_{fam}_std"]) for r in rows]),
                )
                for fam, _, _ in FAMILIES
            },
        }
    return out


def plot(curves, source, target, output_path, width_in, height_in):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(width_in, height_in))
    witness = next(iter(curves.values()))["witness_instrument"]
    names = {"source": source, "target": target, "witness": witness}

    for i, row in enumerate(GRID):
        for j, (regime, row_label) in enumerate(row):
            ax = axes[i][j]
            c = curves[regime]
            for fam, color, _ in FAMILIES:
                mean, std = c[fam]
                ax.plot(c["alpha"], mean, color=color, marker="o", markersize=2, linewidth=0.9)
                ax.fill_between(c["alpha"], mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
            ax.grid(True, alpha=0.3, linewidth=0.4)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
            if i == 0:
                ax.set_title(COLUMN_TITLES[j], pad=3)
            if i == 1:
                ax.set_xlabel(r"$\alpha$", labelpad=1)
            if j == 0:
                ax.set_ylabel(f"{row_label}\nFAD similarity", labelpad=2)

    handles = [plt.Line2D([], [], color=color, marker="o", markersize=2, linewidth=0.9) for _, color, _ in FAMILIES]
    labels = [f"{names[fam]} ({suffix})" for fam, _, suffix in FAMILIES]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.5)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4, rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura guardada en {output_path}")


def main():
    # ── Config ──────────────────────────────────────────
    summary_csv = MODEL_DIR / "interpolation_random_25_samples_ckpt_vs_scratch_gaussian_latest2" / "interpolation_summary_metrics.csv"
    source, target = "piano", "guitar"
    metric = "fad"
    output_path = MODEL_DIR / "docs" / "Paper_Template_for_LAMIR_2024" / "figs" / "cross_transition.pdf"
    width_in, height_in = 3.23, 2.6  # ancho de columna del template ISMIR/LAMIR (82 mm)
    # ────────────────────────────────────────────────────
    curves = read_summary(summary_csv, source, target, metric)
    plot(curves, source, target, output_path, width_in, height_in)


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default="imgs",
    help="Directorio de salida para los gráficos",
)
args = parser.parse_args()
output_dir = args.output_dir

# Crear carpeta de imágenes si no existe
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================

latentdim_experiments = [
    ("vae_latentdim_2", 2),
    ("vae_latentdim_3", 3),
    ("vae_latentdim_4", 4),
    ("vae_latentdim_6", 6),
    ("vae_latentdim_8", 8),
]

beta_latentdim_experiments = [
    ("beta_vae_latentdim_2", 2),
    ("beta_vae_latentdim_3", 3),
    ("beta_vae_latentdim_4", 4),
    ("beta_vae_latentdim_6", 6),
    ("beta_vae_latentdim_8", 8),
]

experiments = [
    ("experiments_models/experiment_latent_dim_voice", "Voice"),
    ("experiments_models/experiment_latent_dim_piano", "Piano"),
    ("experiments_models/experiment_latent_dim_bass", "Bass"),
    ("experiments_models/experiment_latent_dim_guitar", "Guitar"),
]

beta_experiments = [
    ("experiments_models/experiment_latent_dim_voice", "Voice"),
    ("experiments_models/experiment_latent_dim_piano", "Piano"),
    ("experiments_models/experiment_latent_dim_bass", "Bass"),
    ("experiments_models/experiment_latent_dim_guitar", "Guitar"),
]

BASE_MODEL_EXPERIMENT = "experiments_models/experiment_latent_dim_base_model"

versions = [f"version_{i}" for i in range(5)]
beta_versions = [f"version_{i}" for i in range(5)]


def _aggregate_dim_metrics(experiment_path, latentdim_experiments, versions, metric_cols):
    """Promedia métricas a lo largo de las versiones para cada (run, dim).
    Devuelve dict {dim: {col: mean_val}}."""
    agg = {}
    for latentdim_experiment, dim in latentdim_experiments:
        samples = {col: [] for col in metric_cols}
        for version in versions:
            path = f"{experiment_path}/{latentdim_experiment}/{version}/metrics_history_vae.csv"
            if not os.path.exists(path):
                print(f"Advertencia: No se encontró {path}")
                continue
            metrics = pd.read_csv(path)
            for col in metric_cols:
                samples[col].append(metrics[col].min())
        if samples[metric_cols[0]]:
            agg[dim] = {col: sum(vals) / len(vals) for col, vals in samples.items()}
    return agg


# --- Carga VAE Standard ---
records = {}
for experiment, instrument in experiments:
    dim_metrics = _aggregate_dim_metrics(
        experiment, latentdim_experiments, versions, ["val_recon"]
    )
    for dim, m in dim_metrics.items():
        records[(instrument, dim)] = {
            "Instrumento": instrument,
            "Espacio Latente": dim,
            "Error Validación": m["val_recon"],
        }

df = pd.DataFrame(records.values())
df = df.sort_values(by=["Instrumento", "Espacio Latente"])

# --- Carga Beta VAE ---
records_beta = {}
for experiment, instrument in beta_experiments:
    dim_metrics = _aggregate_dim_metrics(
        experiment, beta_latentdim_experiments, beta_versions, ["val_recon", "val_bkl"]
    )
    for dim, m in dim_metrics.items():
        records_beta[(instrument, dim)] = {
            "Instrumento": instrument,
            "Espacio Latente": dim,
            "Error Validación": m["val_recon"],
            "Val BKL": m["val_bkl"],
        }

df_beta = pd.DataFrame(records_beta.values())
df_beta = df_beta.sort_values(by=["Instrumento", "Espacio Latente"])

# --- Carga Modelo Base ---
base_vae_metrics = _aggregate_dim_metrics(
    BASE_MODEL_EXPERIMENT, latentdim_experiments, versions, ["val_recon"]
)
df_base_vae = pd.DataFrame(
    [
        {"Espacio Latente": dim, "Error Validación": m["val_recon"]}
        for dim, m in base_vae_metrics.items()
    ],
    columns=["Espacio Latente", "Error Validación"],
).sort_values(by="Espacio Latente")

base_beta_metrics = _aggregate_dim_metrics(
    BASE_MODEL_EXPERIMENT,
    beta_latentdim_experiments,
    beta_versions,
    ["val_recon", "val_bkl"],
)
df_base_beta = pd.DataFrame(
    [
        {
            "Espacio Latente": dim,
            "Error Validación": m["val_recon"],
            "Val BKL": m["val_bkl"],
        }
        for dim, m in base_beta_metrics.items()
    ],
    columns=["Espacio Latente", "Error Validación", "Val BKL"],
).sort_values(by="Espacio Latente")

# ==========================================
# 2. GENERACIÓN DE GRÁFICOS
# ==========================================

# Estilo general
plt.style.use("seaborn-v0_8-whitegrid")  # O usa 'ggplot' si prefieres
colors_vae = "#1f77b4"  # Azul
colors_beta = "#ff7f0e"  # Naranja


def _plot_arch_panel(ax, title, subset_vae, subset_beta, all_dims):
    ax.plot(
        subset_vae["Espacio Latente"],
        subset_vae["Error Validación"],
        marker="o",
        linestyle="-",
        label="VAE (Sin Beta)",
        color=colors_vae,
    )
    ax.plot(
        subset_beta["Espacio Latente"],
        subset_beta["Error Validación"],
        marker="s",
        linestyle="--",
        label="Beta-VAE",
        color=colors_beta,
    )
    ax.set_title(title)
    ax.set_xlabel("Dimensión Latente")
    ax.set_ylabel("Error Validación (Reconstrucción)")
    ax.set_xticks(all_dims)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------
# Gráfico 1: Val Loss vs Latent Dim por Instrumento + Base Model
# ---------------------------------------------------------
instruments = df["Instrumento"].unique()
all_dims = sorted(df["Espacio Latente"].unique())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, instrument in enumerate(instruments):
    _plot_arch_panel(
        axes[idx],
        instrument,
        df[df["Instrumento"] == instrument],
        df_beta[df_beta["Instrumento"] == instrument],
        all_dims,
    )

_plot_arch_panel(axes[4], "Base Model", df_base_vae, df_base_beta, all_dims)
axes[5].set_visible(False)

plt.suptitle("Comparación de Error de Reconstrucción: VAE vs Beta-VAE", fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/1_comparacion_por_instrumento.png")
plt.close()

# ---------------------------------------------------------
# Gráfico 2: Val Loss Mean & Std incluyendo Base Model (5 sujetos)
# ---------------------------------------------------------
df_vae_full = pd.concat(
    [df, df_base_vae.assign(Instrumento="Base Model")], ignore_index=True
)
df_beta_full = pd.concat(
    [df_beta, df_base_beta.assign(Instrumento="Base Model")], ignore_index=True
)

stats_vae_full = (
    df_vae_full.groupby("Espacio Latente")["Error Validación"]
    .agg(["mean", "std"])
    .reset_index()
)
stats_beta_full = (
    df_beta_full.groupby("Espacio Latente")["Error Validación"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(10, 6))

plt.errorbar(
    stats_vae_full["Espacio Latente"],
    stats_vae_full["mean"],
    yerr=stats_vae_full["std"],
    fmt="-o",
    capsize=5,
    label="Promedio VAE",
    color=colors_vae,
    alpha=0.8,
)

plt.errorbar(
    stats_beta_full["Espacio Latente"],
    stats_beta_full["mean"],
    yerr=stats_beta_full["std"],
    fmt="--s",
    capsize=5,
    label="Promedio Beta-VAE",
    color=colors_beta,
    alpha=0.8,
)

plt.xlabel("Espacio Latente")
plt.ylabel("Error de Validación Promedio (Recon)")
plt.title("Rendimiento Promedio Global (incluye Base Model): VAE vs Beta-VAE")
plt.legend()
plt.xticks(sorted(stats_vae_full["Espacio Latente"].unique()))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/2_val_loss_mean_std_with_base.png")
plt.close()

# ---------------------------------------------------------
# Gráfico 3: Mejora Porcentual (Rendimientos Decrecientes)
# ---------------------------------------------------------
# Calcular mejora porcentual VAE
vae_mean = df.groupby("Espacio Latente")["Error Validación"].mean().sort_index()
vae_imp = -vae_mean.pct_change() * 100
vae_imp = vae_imp.dropna()

# Calcular mejora porcentual Beta VAE
beta_mean = df_beta.groupby("Espacio Latente")["Error Validación"].mean().sort_index()
beta_imp = -beta_mean.pct_change() * 100
beta_imp = beta_imp.dropna()

labels = [
    f"{prev}->{curr}" for prev, curr in zip(vae_mean.index[:-1], vae_mean.index[1:])
]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
rects1 = plt.bar(
    x - width / 2, vae_imp, width, label="VAE", color=colors_vae, alpha=0.8
)
rects2 = plt.bar(
    x + width / 2, beta_imp, width, label="Beta-VAE", color=colors_beta, alpha=0.8
)


# Función para poner etiquetas
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


autolabel(rects1)
autolabel(rects2)

plt.xlabel("Incremento en Espacio Latente")
plt.ylabel("Reducción del Error (%)")
plt.title("Mejora Porcentual (Rendimientos Decrecientes)")
plt.xticks(x, labels)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/3_improvement_percentage_compare.png")
plt.close()

# ---------------------------------------------------------
# Gráfico 4: Evolución del Error KL (Solo Beta-VAE) + Base Model
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for instrument in df_beta["Instrumento"].unique():
    subset = df_beta[df_beta["Instrumento"] == instrument]
    plt.plot(subset["Espacio Latente"], subset["Val BKL"], marker="o", label=instrument)

if not df_base_beta.empty:
    plt.plot(
        df_base_beta["Espacio Latente"],
        df_base_beta["Val BKL"],
        marker="D",
        linestyle="--",
        linewidth=2,
        color="black",
        label="Base Model",
    )

plt.xlabel("Espacio Latente")
plt.ylabel("Divergencia KL (Val BKL)")
plt.title("Evolución de la Divergencia KL en Beta-VAE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(sorted(df_beta["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig(f"{output_dir}/4_beta_kl_evolution.png")
plt.close()

# ---------------------------------------------------------
# Gráfico 5: Evolución del Error de Reconstrucción (VAE, β=0) + Base Model
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for instrument in df["Instrumento"].unique():
    subset = df[df["Instrumento"] == instrument]
    plt.plot(
        subset["Espacio Latente"],
        subset["Error Validación"],
        marker="o",
        label=instrument,
    )

if not df_base_vae.empty:
    plt.plot(
        df_base_vae["Espacio Latente"],
        df_base_vae["Error Validación"],
        marker="D",
        linestyle="--",
        linewidth=2,
        color="black",
        label="Base Model",
    )

plt.xlabel("Espacio Latente")
plt.ylabel("Error Validación (Reconstrucción)")
plt.title("Evolución del Error de Reconstrucción (VAE, β=0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(sorted(df["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig(f"{output_dir}/5_recon_evolution_no_beta.png")
plt.close()

print(f"¡Gráficos generados exitosamente en la carpeta '{output_dir}'!")

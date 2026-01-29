import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta de imágenes si no existe
os.makedirs("imgs", exist_ok=True)

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
    ("experiment_latent_dim_voice_small", "Voice"),
    ("experiment_latent_dim_piano_small", "Piano"),
    ("experiment_latent_dim_bass_small", "Bass"),
    ("experiment_latent_dim_guitar_small", "Guitar"),
]

beta_experiments = [
    ("experiment_latent_dim_beta_voice_small", "Voice"),
    ("experiment_latent_dim_beta_piano_small", "Piano"),
    ("experiment_latent_dim_beta_bass_small", "Bass"),
    ("experiment_latent_dim_beta_guitar_small", "Guitar"),
]

versions = [f"version_{i}" for i in range(6)]
beta_versions = [f"version_{i}" for i in range(5)]

# --- Carga VAE Standard ---
records = {}
for experiment, instrument in experiments:
    for latentdim_experiment, dim in latentdim_experiments:
        val_losses = []
        train_losses = []

        for version in versions:
            # Nota: Ajusta la ruta si es necesario
            path = (
                f"{experiment}/{latentdim_experiment}/{version}/metrics_history_vae.csv"
            )
            if not os.path.exists(path):
                print(f"Advertencia: No se encontró {path}")
                continue

            metrics = pd.read_csv(path)

            # Usamos val_recon para que sea comparable con el Beta VAE en términos de calidad de audio
            val_losses.append(metrics["val_recon"].min())
            train_losses.append(metrics["train_recon"].min())

        if val_losses:
            records[(instrument, dim)] = {
                "Instrumento": instrument,
                "Espacio Latente": dim,
                "Error Validación": sum(val_losses) / len(val_losses),
                "Error Entrenamiento": sum(train_losses) / len(train_losses),
            }

df = pd.DataFrame(records.values())
df = df.sort_values(by=["Instrumento", "Espacio Latente"])

# --- Carga Beta VAE ---
records_beta = {}
for experiment, instrument in beta_experiments:
    for latentdim_experiment, dim in beta_latentdim_experiments:
        val_losses = []
        val_bkl = []  # Para el gráfico de KL

        for version in beta_versions:
            path = (
                f"{experiment}/{latentdim_experiment}/{version}/metrics_history_vae.csv"
            )
            if not os.path.exists(path):
                print(f"Advertencia: No se encontró {path}")
                continue

            metrics = pd.read_csv(path)

            # En Beta VAE, 'val_recon' es la reconstrucción pura, comparable con VAE
            val_losses.append(metrics["val_recon"].min())
            val_bkl.append(metrics["val_bkl"].min())

        if val_losses:
            records_beta[(instrument, dim)] = {
                "Instrumento": instrument,
                "Espacio Latente": dim,
                "Error Validación": sum(val_losses) / len(val_losses),
                "Val BKL": sum(val_bkl) / len(val_bkl),
            }

df_beta = pd.DataFrame(records_beta.values())
df_beta = df_beta.sort_values(by=["Instrumento", "Espacio Latente"])

# ==========================================
# 2. GENERACIÓN DE GRÁFICOS
# ==========================================

# Estilo general
plt.style.use("seaborn-v0_8-whitegrid")  # O usa 'ggplot' si prefieres
colors_vae = "#1f77b4"  # Azul
colors_beta = "#ff7f0e"  # Naranja

# ---------------------------------------------------------
# Gráfico 1: Val Loss vs Latent Dim por Instrumento (Comparativo)
# ---------------------------------------------------------
instruments = df["Instrumento"].unique()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, instrument in enumerate(instruments):
    ax = axes[idx]

    # Datos VAE
    subset_vae = df[df["Instrumento"] == instrument]
    ax.plot(
        subset_vae["Espacio Latente"],
        subset_vae["Error Validación"],
        marker="o",
        linestyle="-",
        label="VAE (Sin Beta)",
        color=colors_vae,
    )

    # Datos Beta VAE
    subset_beta = df_beta[df_beta["Instrumento"] == instrument]
    ax.plot(
        subset_beta["Espacio Latente"],
        subset_beta["Error Validación"],
        marker="s",
        linestyle="--",
        label="Beta-VAE",
        color=colors_beta,
    )

    ax.set_title(f"{instrument}")
    ax.set_xlabel("Dimensión Latente")
    ax.set_ylabel("Error Validación (Reconstrucción)")
    ax.set_xticks(sorted(df["Espacio Latente"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Comparación de Error de Reconstrucción: VAE vs Beta-VAE", fontsize=16)
plt.tight_layout()
plt.savefig("imgs/1_comparacion_por_instrumento.png")
plt.close()

# ---------------------------------------------------------
# Gráfico 2: Val Loss Mean & Std (Comparativo Global)
# ---------------------------------------------------------
# Agrupar VAE
stats_vae = (
    df.groupby("Espacio Latente")["Error Validación"].agg(["mean", "std"]).reset_index()
)
# Agrupar Beta VAE
stats_beta = (
    df_beta.groupby("Espacio Latente")["Error Validación"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(10, 6))

# VAE Plot
plt.errorbar(
    stats_vae["Espacio Latente"],
    stats_vae["mean"],
    yerr=stats_vae["std"],
    fmt="-o",
    capsize=5,
    label="Promedio VAE",
    color=colors_vae,
    alpha=0.8,
)

# Beta VAE Plot
plt.errorbar(
    stats_beta["Espacio Latente"],
    stats_beta["mean"],
    yerr=stats_beta["std"],
    fmt="--s",
    capsize=5,
    label="Promedio Beta-VAE",
    color=colors_beta,
    alpha=0.8,
)

plt.xlabel("Espacio Latente")
plt.ylabel("Error de Validación Promedio (Recon)")
plt.title("Rendimiento Promedio Global: VAE vs Beta-VAE")
plt.legend()
plt.xticks(sorted(stats_vae["Espacio Latente"].unique()))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("imgs/2_val_loss_mean_std_compare.png")
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
plt.savefig("imgs/3_improvement_percentage_compare.png")
plt.close()

# ---------------------------------------------------------
# Gráfico 4: Evolución del Error KL (Solo Beta-VAE)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for instrument in df_beta["Instrumento"].unique():
    subset = df_beta[df_beta["Instrumento"] == instrument]
    plt.plot(subset["Espacio Latente"], subset["Val BKL"], marker="o", label=instrument)

plt.xlabel("Espacio Latente")
plt.ylabel("Divergencia KL (Val BKL)")
plt.title("Evolución de la Divergencia KL en Beta-VAE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(sorted(df_beta["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig("imgs/4_beta_kl_evolution.png")
plt.close()

print("¡Gráficos generados exitosamente en la carpeta 'imgs'!")

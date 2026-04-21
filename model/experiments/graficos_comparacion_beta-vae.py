import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

model_base = "experiments_models/base_model_beta_variation"

beta_dirs = {
    "base_model_beta_0.01": 0.01,
    "base_model_beta_0.001": 0.001,
    "base_model_beta_0.0001": 0.0001,
    "base_model_beta_1e-05": 0.00001,
    "base_model_beta_0": 0,
}

plt.style.use("seaborn-v0_8-whitegrid")

fig_recon, ax_recon = plt.subplots(1, 2, figsize=(16, 6))
fig_kl, ax_kl = plt.subplots(1, 2, figsize=(16, 6))
fig_kl_zoom, ax_kl_zoom = plt.subplots(1, 2, figsize=(16, 6))

fig_recon.suptitle("Evolución del Error de Reconstrucción (Train vs Val)", fontsize=16)
fig_kl.suptitle("Evolución de la Divergencia KL (Train vs Val) - Escala Amplia", fontsize=16)
fig_kl_zoom.suptitle("Evolución de la Divergencia KL (Train vs Val) - Zoom In", fontsize=16)

ax_recon[0].set_title("Train Reconstruction Error")
ax_recon[0].set_xlabel("Paso / Etapa")
ax_recon[0].set_ylabel("Error")

ax_recon[1].set_title("Validation Reconstruction Error")
ax_recon[1].set_xlabel("Paso / Etapa")
ax_recon[1].set_ylabel("Error")

ax_kl[0].set_title("Train KL Divergence")
ax_kl[0].set_xlabel("Paso / Etapa")
ax_kl[0].set_ylabel("KL")

ax_kl[1].set_title("Validation KL Divergence")
ax_kl[1].set_xlabel("Paso / Etapa")
ax_kl[1].set_ylabel("KL")

ax_kl_zoom[0].set_title("Train KL Divergence")
ax_kl_zoom[0].set_xlabel("Paso / Etapa")
ax_kl_zoom[0].set_ylabel("KL")

ax_kl_zoom[1].set_title("Validation KL Divergence")
ax_kl_zoom[1].set_xlabel("Paso / Etapa")
ax_kl_zoom[1].set_ylabel("KL")

for dir_name, beta in beta_dirs.items():
    csv_path = os.path.join(model_base, dir_name, "version_0", "metrics_history_vae.csv")
    if not os.path.exists(csv_path):
        print(f"Advertencia: No se encontró {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Train Recon
    if "train_recon" in df.columns:
        train_recon_q = df["train_recon"].dropna()
        # Ignoramos el paso 0 que suele tener un error gigante (outlier inicial)
        train_recon_q = train_recon_q.iloc[1:]
        ax_recon[0].plot(train_recon_q.index, train_recon_q.values, label=f"Beta {beta}")

    # Val Recon
    if "val_recon" in df.columns:
        val_recon_q = df["val_recon"].dropna()
        val_recon_q = val_recon_q.iloc[1:]
        ax_recon[1].plot(val_recon_q.index, val_recon_q.values, label=f"Beta {beta}")

    # Train KL
    if "train_kl" in df.columns:
        train_kl_q = df["train_kl"].dropna()
    elif "train_bkl" in df.columns:
        train_kl_q = df["train_bkl"].dropna()
    else:
        train_kl_q = pd.Series(dtype=float)

    if not train_kl_q.empty:
        train_kl_q = train_kl_q.iloc[1:]
        ax_kl[0].plot(train_kl_q.index, train_kl_q.values, label=f"Beta {beta}")
        ax_kl_zoom[0].plot(train_kl_q.index, train_kl_q.values, label=f"Beta {beta}")

    # Val KL
    if "val_kl" in df.columns:
        val_kl_q = df["val_kl"].dropna()
    elif "val_bkl" in df.columns:
        val_kl_q = df["val_bkl"].dropna()
    else:
        val_kl_q = pd.Series(dtype=float)

    if not val_kl_q.empty:
        val_kl_q = val_kl_q.iloc[1:]
        ax_kl[1].plot(val_kl_q.index, val_kl_q.values, label=f"Beta {beta}")
        ax_kl_zoom[1].plot(val_kl_q.index, val_kl_q.values, label=f"Beta {beta}")

for ax in ax_recon:
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    # Limitar el top a un valor representativo obviando picos.
    current_top = ax.get_ylim()[1]
    ax.set_ylim(top=min(0.01, current_top))

for ax in ax_kl:
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    current_top = ax.get_ylim()[1]
    ax.set_ylim(top=min(20, current_top))

for ax in ax_kl_zoom:
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    current_top = ax.get_ylim()[1]
    ax.set_ylim(top=min(5.5, current_top))

fig_recon.tight_layout()
fig_kl.tight_layout()
fig_kl_zoom.tight_layout()

recon_path = os.path.join(output_dir, "recon_evolution.png")
kl_path = os.path.join(output_dir, "kl_evolution_0.2.png")
kl_zoom_path = os.path.join(output_dir, "kl_evolution_0.002.png")

fig_recon.savefig(recon_path)
fig_kl.savefig(kl_path)
fig_kl_zoom.savefig(kl_zoom_path)

plt.close(fig_recon)
plt.close(fig_kl)
plt.close(fig_kl_zoom)

print(f"¡Gráficos de Beta-VAE generados exitosamente en la carpeta '{output_dir}'!")

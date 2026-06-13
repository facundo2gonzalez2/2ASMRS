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

pareto_data = []
window_size = 30

# Crear las tres figuras
fig_val, ax_val = plt.subplots(1, 2, figsize=(16, 6))
fig_pareto_lin, ax_pareto_lin = plt.subplots(figsize=(8, 6))
fig_pareto_log, ax_pareto_log = plt.subplots(figsize=(8, 6))

fig_val.suptitle("Dinámicas de Validación Suavizadas (Media Móvil)", fontsize=16, fontweight="bold")
fig_pareto_lin.suptitle(
    "Trade-off: Error vs Divergencia KL\n(Escala Lineal, excluyendo $\\beta=0$)", fontsize=16, fontweight="bold"
)
fig_pareto_log.suptitle(
    "Trade-off: Error vs Divergencia KL\n(Escala Log-Log, incluyendo $\\beta=0$)", fontsize=16, fontweight="bold"
)

ax_val[0].set_title("Validation Reconstruction Error")
ax_val[0].set_xlabel("Paso / Etapa")
ax_val[0].set_ylabel("Error")

ax_val[1].set_title("Validation KL Divergence")
ax_val[1].set_xlabel("Paso / Etapa")
ax_val[1].set_ylabel("KL")

for dir_name, beta in beta_dirs.items():
    csv_path = os.path.join(model_base, dir_name, "version_0", "metrics_history_vae.csv")
    if not os.path.exists(csv_path):
        print(f"Advertencia: No se encontró {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    val_recon_col = "val_recon" if "val_recon" in df.columns else None
    val_kl_col = "val_kl"

    if val_recon_col and val_kl_col:
        val_recon = df[val_recon_col].dropna().iloc[1:]
        val_kl = df[val_kl_col].dropna().iloc[1:]

        # 1. Series temporales suavizadas
        recon_smooth = val_recon.rolling(window=window_size, min_periods=1).mean()
        kl_smooth = val_kl.rolling(window=window_size, min_periods=1).mean()

        linewidth = 1.2
        alpha = 0.7

        ax_val[0].plot(recon_smooth.index, recon_smooth.values, label=f"Beta {beta}", linewidth=linewidth, alpha=alpha)
        ax_val[1].plot(kl_smooth.index, kl_smooth.values, label=f"Beta {beta}", linewidth=linewidth, alpha=alpha)

        # 2. Datos para el Pareto (último 10% del entrenamiento)
        n_tail = max(10, int(len(val_recon) * 0.1))
        avg_recon = val_recon.tail(n_tail).mean()
        avg_kl = val_kl.tail(n_tail).mean()

        pareto_data.append({"beta": beta, "recon": avg_recon, "kl": avg_kl})

# --- Preparar DataFrames para Pareto ---
pareto_df = pd.DataFrame(pareto_data).sort_values(by="kl")
pareto_df_lin = pareto_df[pareto_df["beta"] != 0]

# --- 1. Dibujar Gráfico de Pareto (Lineal, sin beta=0) ---
ax_pareto_lin.plot(
    pareto_df_lin["kl"], pareto_df_lin["recon"], marker="", linestyle="--", color="gray", alpha=0.5, zorder=1
)

for _, row in pareto_df_lin.iterrows():
    color = "blue"
    size = 50
    ax_pareto_lin.scatter(row["kl"], row["recon"], color=color, s=size, zorder=5)

    label = f"$\\beta$={row['beta']}"

    ax_pareto_lin.annotate(
        label,
        (row["kl"], row["recon"]),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=10,
        fontweight="normal",
    )

ax_pareto_lin.set_xlabel("Divergencia KL")
ax_pareto_lin.set_ylabel("Error de Reconstrucción")
ax_pareto_lin.grid(True, alpha=0.3)

# --- 2. Dibujar Gráfico de Pareto (Logarítmico, con beta=0) ---
ax_pareto_log.plot(pareto_df["kl"], pareto_df["recon"], marker="", linestyle="--", color="gray", alpha=0.5, zorder=1)

for _, row in pareto_df.iterrows():
    color = "blue"
    size = 50
    ax_pareto_log.scatter(row["kl"], row["recon"], color=color, s=size, zorder=5)

    label = f"$\\beta$={row['beta']}"

    xytext_offset = (10, 5) if row["beta"] != 0 else (-40, -15)

    ax_pareto_log.annotate(
        label,
        (row["kl"], row["recon"]),
        xytext=xytext_offset,
        textcoords="offset points",
        fontsize=10,
        fontweight="normal",
    )

ax_pareto_log.set_xscale("log")
ax_pareto_log.set_yscale("log")
ax_pareto_log.set_xlabel("Divergencia KL (Log Scale)")
ax_pareto_log.set_ylabel("Error de Reconstrucción (Log Scale)")
ax_pareto_log.grid(True, alpha=0.3, which="both", ls="--")

# --- Ajustes visuales finales para Validación ---
ax_val[0].legend()
ax_val[0].grid(True, alpha=0.3)
ax_val[0].set_ylim(bottom=0.002, top=0.0085)

ax_val[1].legend()
ax_val[1].grid(True, alpha=0.3)
ax_val[1].set_ylim(bottom=0, top=6)

fig_val.tight_layout()
fig_pareto_lin.tight_layout()
fig_pareto_log.tight_layout()

val_path = os.path.join(output_dir, "validation_smoothed_evolution.png")
pareto_lin_path = os.path.join(output_dir, "pareto_tradeoff_linear.png")
pareto_log_path = os.path.join(output_dir, "pareto_tradeoff_loglog.png")

fig_val.savefig(val_path, dpi=300)
fig_pareto_lin.savefig(pareto_lin_path, dpi=300)
fig_pareto_log.savefig(pareto_log_path, dpi=300)

plt.close("all")

print(f"¡Los tres gráficos fueron generados exitosamente en la carpeta '{output_dir}'!")

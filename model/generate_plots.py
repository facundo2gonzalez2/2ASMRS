import pandas as pd
import re
import matplotlib.pyplot as plt

latentdim_experiments = [
    "vae_latentdim_2",
    "vae_latentdim_3",
    "vae_latentdim_4",
    "vae_latentdim_6",
    "vae_latentdim_8",
]

experiments = [
    "experiment_latent_dim_voice_small",
    "experiment_latent_dim_piano_small",
    "experiment_latent_dim_bass_small",
    "experiment_latent_dim_guitar_small",
]

versions = [f"version_{i}" for i in range(6)]

data_str = ""

# 1. Generación del String de Datos
for experiment in experiments:
    for latentdim_experiment in latentdim_experiments:
        val_losses = []
        train_losses = []
        val_epochs = []
        train_epochs = []

        for version in versions:
            metrics = pd.read_csv(
                f"{experiment}/{latentdim_experiment}/{version}/metrics_history_vae.csv"
            )

            # per-version minima
            min_val_loss = metrics["val_loss"].min()
            min_val_loss_epoch = metrics["val_loss"].idxmin()
            min_train_loss = metrics["train_loss"].min()
            min_train_loss_epoch = metrics["train_loss"].idxmin()

            val_losses.append(min_val_loss)
            val_epochs.append(min_val_loss_epoch)
            train_losses.append(min_train_loss)
            train_epochs.append(min_train_loss_epoch)

        # average across versions
        avg_min_val_loss = sum(val_losses) / len(val_losses)
        avg_min_train_loss = sum(train_losses) / len(train_losses)
        avg_min_val_loss_epoch = sum(val_epochs) / len(val_epochs)
        avg_min_train_loss_epoch = sum(train_epochs) / len(train_epochs)

        data_str += (
            f"Experiment: {experiment}, Latent Dim: {latentdim_experiment}, "
            f"Min Val Loss: {avg_min_val_loss}, Min Val Loss Epoch: {avg_min_val_loss_epoch}\n"
        )
        data_str += (
            f"Experiment: {experiment}, Latent Dim: {latentdim_experiment}, "
            f"Min Train Loss: {avg_min_train_loss}, Min Train Loss Epoch: {avg_min_train_loss_epoch}\n"
        )


# 2. Parseo Unificado (Una sola vez para todo)
records = {}
# Actualizamos el regex para capturar también la Época al final de la línea
pattern = r"Experiment: experiment_latent_dim_([a-z]+)_small, Latent Dim: vae_latentdim_(\d+), Min (Val|Train) Loss: ([\d\.]+), Min (?:Val|Train) Loss Epoch: ([\d\.]+)"

for line in data_str.strip().split("\n"):
    match = re.search(pattern, line)
    if match:
        inst = match.group(1).capitalize()
        dim = int(match.group(2))
        loss_type = match.group(3)
        loss_val = float(match.group(4))
        epoch_val = float(match.group(5))  # Capturamos la época

        key = (inst, dim)
        if key not in records:
            records[key] = {"Instrumento": inst, "Espacio Latente": dim}

        if loss_type == "Val":
            records[key]["Error Validación"] = loss_val
            records[key]["Min Val Loss Epoch"] = epoch_val  # Guardamos época val

        elif loss_type == "Train":
            records[key]["Error Entrenamiento"] = loss_val
            # Opcional: records[key]["Min Train Loss Epoch"] = epoch_val

df = pd.DataFrame(records.values())
df = df.sort_values(by=["Instrumento", "Espacio Latente"])

# ---------------------------------------------------------
# Plot 1: Validación por Instrumento
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

plt.xlabel("Espacio Latente")
plt.ylabel("Error Validación")
plt.title("Error de Validación vs Espacio Latente por Instrumento")
plt.legend()
plt.grid(True)
plt.xticks(sorted(df["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig("imgs/val_loss_vs_latent_dim.png")

# ---------------------------------------------------------
# Plot 2: Promedio General con Desviación Estándar
# ---------------------------------------------------------
stats = (
    df.groupby("Espacio Latente")["Error Validación"].agg(["mean", "std"]).reset_index()
)

plt.figure(figsize=(10, 6))
plt.errorbar(
    stats["Espacio Latente"],
    stats["mean"],
    yerr=stats["std"],
    fmt="-o",
    capsize=5,
    label="Promedio general",
)

plt.xlabel("Espacio Latente")
plt.ylabel("Error de Validación Promedio")
plt.title("Disminución del Error de Validación Promedio")
plt.grid(True)
plt.xticks(sorted(stats["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig("imgs/val_loss_mean_std.png")

# ---------------------------------------------------------
# Plot 3: Mejora Porcentual (Rendimientos Decrecientes)
# ---------------------------------------------------------
stats_mean = df.groupby("Espacio Latente")["Error Validación"].mean().sort_index()
improvements = -stats_mean.pct_change() * 100
improvements = improvements.dropna()

labels = [
    f"{prev}->{curr}" for prev, curr in zip(stats_mean.index[:-1], stats_mean.index[1:])
]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, improvements, color="#4c72b0", zorder=3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.1,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.title("Mejora Porcentual del Error al Aumentar Dimensiones")
plt.xlabel("Incremento en Espacio Latente")
plt.ylabel("Reducción del Error (%)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("imgs/improvement_percentage.png")

# ---------------------------------------------------------
# Plot 4: Gap de Generalización (Train vs Val)
# ---------------------------------------------------------
stats_train = df.groupby("Espacio Latente")["Error Entrenamiento"].mean()
stats_val = df.groupby("Espacio Latente")["Error Validación"].mean()

plt.figure(figsize=(10, 6))
plt.plot(stats_val.index, stats_val.values, marker="o", label="Validación", linewidth=2)
plt.plot(
    stats_train.index,
    stats_train.values,
    marker="s",
    label="Entrenamiento",
    linestyle="--",
    alpha=0.7,
)

plt.fill_between(
    stats_val.index, stats_train.values, stats_val.values, alpha=0.1, color="gray"
)

plt.xlabel("Espacio Latente")
plt.ylabel("Loss Promedio")
plt.title("Gap de Generalización: Entrenamiento vs Validación")
plt.legend()
plt.grid(True)
plt.xticks(sorted(df["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig("imgs/train_vs_val_gap.png")

# ---------------------------------------------------------
# Plot 5: Velocidad de Convergencia
# ---------------------------------------------------------
# Ahora esto funcionará porque parseamos 'Min Val Loss Epoch' arriba
stats_epochs = df.groupby("Espacio Latente")["Min Val Loss Epoch"].mean()

plt.figure(figsize=(10, 6))
plt.plot(
    stats_epochs.index, stats_epochs.values, marker="D", color="green", linestyle="-."
)

plt.xlabel("Espacio Latente")
plt.ylabel("Época del Mínimo Error (Validación)")
plt.title("Velocidad de Convergencia Promedio")
plt.grid(True)
plt.xticks(sorted(df["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig("imgs/convergence_speed.png")

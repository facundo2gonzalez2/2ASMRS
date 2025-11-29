import pandas as pd
import re
import matplotlib.pyplot as plt

# latentdim_experiments = [
#     "vae_latentdim_2",
#     "vae_latentdim_3",
#     "vae_latentdim_4",
#     "vae_latentdim_6",
#     "vae_latentdim_8",
# ]

# experiments = [
#     "experiment_latent_dim_voice_small",
#     "experiment_latent_dim_piano_small",
#     "experiment_latent_dim_bass_small",
#     "experiment_latent_dim_guitar_small",
# ]

# for experiment in experiments:
#     for latentdim_experiment in latentdim_experiments:
#         metrics = pd.read_csv(
#             f"{experiment}/{latentdim_experiment}/version_0/metrics_history_vae.csv"
#         )
#         min_val_loss = metrics["val_loss"].min()
#         min_val_loss_epoch = metrics["val_loss"].idxmin()
#         print(
#             f"Experiment: {experiment}, Latent Dim: {latentdim_experiment}, Min Val Loss: {min_val_loss}, Min Val Loss Epoch: {min_val_loss_epoch}"
#         )

#         min_train_loss = metrics["train_loss"].min()
#         min_train_loss_epoch = metrics["train_loss"].idxmin()
#         print(
#             f"Experiment: {experiment}, Latent Dim: {latentdim_experiment}, Min Train Loss: {min_train_loss}, Min Train Loss Epoch: {min_train_loss_epoch}"
#         )

data_str = """
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_2, Min Val Loss: 0.0036795954219996, Min Val Loss Epoch: 896
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_2, Min Train Loss: 0.0031399470753967, Min Train Loss Epoch: 971
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_3, Min Val Loss: 0.0028366423211991, Min Val Loss Epoch: 845
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_3, Min Train Loss: 0.0023210193030536, Min Train Loss Epoch: 811
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_4, Min Val Loss: 0.0022639378439635, Min Val Loss Epoch: 824
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_4, Min Train Loss: 0.00181851524394, Min Train Loss Epoch: 865
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_6, Min Val Loss: 0.0019274718360975, Min Val Loss Epoch: 998
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_6, Min Train Loss: 0.0016014863504096, Min Train Loss Epoch: 1000
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_8, Min Val Loss: 0.0017863225657492, Min Val Loss Epoch: 961
Experiment: experiment_latent_dim_voice_small, Latent Dim: vae_latentdim_8, Min Train Loss: 0.0015274350298568, Min Train Loss Epoch: 958
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_2, Min Val Loss: 0.0022211386822164, Min Val Loss Epoch: 691
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_2, Min Train Loss: 0.0019498858600854, Min Train Loss Epoch: 783
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_3, Min Val Loss: 0.0022101034410297, Min Val Loss Epoch: 712
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_3, Min Train Loss: 0.0018312627216801, Min Train Loss Epoch: 741
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_4, Min Val Loss: 0.0014656913699582, Min Val Loss Epoch: 719
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_4, Min Train Loss: 0.0012296291533857, Min Train Loss Epoch: 809
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_6, Min Val Loss: 0.0013164352858439, Min Val Loss Epoch: 840
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_6, Min Train Loss: 0.0011030378518626, Min Train Loss Epoch: 919
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_8, Min Val Loss: 0.0012609069235622, Min Val Loss Epoch: 731
Experiment: experiment_latent_dim_piano_small, Latent Dim: vae_latentdim_8, Min Train Loss: 0.0010808920487761, Min Train Loss Epoch: 706
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_2, Min Val Loss: 0.0021817458327859, Min Val Loss Epoch: 533
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_2, Min Train Loss: 0.0020681202877312, Min Train Loss Epoch: 602
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_3, Min Val Loss: 0.0029844546224921, Min Val Loss Epoch: 47
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_3, Min Train Loss: 0.0023435608018189, Min Train Loss Epoch: 121
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_4, Min Val Loss: 0.0019637839868664, Min Val Loss Epoch: 374
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_4, Min Train Loss: 0.0017949879402294, Min Train Loss Epoch: 365
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_6, Min Val Loss: 0.0018852731445804, Min Val Loss Epoch: 256
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_6, Min Train Loss: 0.0017070503672584, Min Train Loss Epoch: 328
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_8, Min Val Loss: 0.0018377560190856, Min Val Loss Epoch: 198
Experiment: experiment_latent_dim_bass_small, Latent Dim: vae_latentdim_8, Min Train Loss: 0.0016324598109349, Min Train Loss Epoch: 279
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_2, Min Val Loss: 0.0022435039281845, Min Val Loss Epoch: 533
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_2, Min Train Loss: 0.0020896804053336, Min Train Loss Epoch: 549
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_3, Min Val Loss: 0.0017770656850188, Min Val Loss Epoch: 634
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_3, Min Train Loss: 0.0014612325467169, Min Train Loss Epoch: 683
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_4, Min Val Loss: 0.0016552835004404, Min Val Loss Epoch: 593
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_4, Min Train Loss: 0.0012582292547449, Min Train Loss Epoch: 662
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_6, Min Val Loss: 0.0015107116196304, Min Val Loss Epoch: 780
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_6, Min Train Loss: 0.0011776937171816, Min Train Loss Epoch: 771
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_8, Min Val Loss: 0.0013368392828851, Min Val Loss Epoch: 709
Experiment: experiment_latent_dim_guitar_small, Latent Dim: vae_latentdim_8, Min Train Loss: 0.0010340777225792, Min Train Loss Epoch: 704
"""

records = {}
pattern = r"Experiment: experiment_latent_dim_([a-z]+)_small, Latent Dim: vae_latentdim_(\d+), Min (Val|Train) Loss: ([\d\.]+)"

for line in data_str.strip().split("\n"):
    match = re.search(pattern, line)
    if match:
        inst = match.group(1).capitalize()
        dim = int(match.group(2))
        loss_type = match.group(3)
        loss_val = float(match.group(4))

        key = (inst, dim)
        if key not in records:
            records[key] = {"Instrumento": inst, "Espacio Latente": dim}

        if loss_type == "Val":
            records[key]["Error Validación"] = loss_val

df = pd.DataFrame(records.values())
df = df.sort_values(by=["Instrumento", "Espacio Latente"])

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
plt.savefig("val_loss_vs_latent_dim.png")


records = {}
pattern = r"Experiment: experiment_latent_dim_([a-z]+)_small, Latent Dim: vae_latentdim_(\d+), Min (Val|Train) Loss: ([\d\.]+)"

for line in data_str.strip().split("\n"):
    match = re.search(pattern, line)
    if match:
        inst = match.group(1).capitalize()
        dim = int(match.group(2))
        loss_type = match.group(3)
        loss_val = float(match.group(4))

        key = (inst, dim)
        if key not in records:
            records[key] = {"Instrumento": inst, "Espacio Latente": dim}

        if loss_type == "Val":
            records[key]["Error Validación"] = loss_val

df = pd.DataFrame(records.values())

# Aggregate by Latent Dimension
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

plt.xlabel("Espacio Latente (Dimensionalidad)")
plt.ylabel("Error de Validación Promedio")
plt.title("Disminución del Error de Validación Promedio vs Espacio Latente")
plt.grid(True)
plt.xticks(sorted(stats["Espacio Latente"].unique()))
plt.tight_layout()
plt.savefig("val_loss_mean_std.png")

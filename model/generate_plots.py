import pandas as pd

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

for experiment in experiments:
    for latentdim_experiment in latentdim_experiments:
        metrics = pd.read_csv(
            f"{experiment}/{latentdim_experiment}/version_0/metrics_history_vae.csv"
        )
        min_val_loss = metrics["val_loss"].min()
        min_val_loss_epoch = metrics["val_loss"].idxmin()
        print(
            f"Experiment: {experiment}, Latent Dim: {latentdim_experiment}, Min Val Loss: {min_val_loss}, Min Val Loss Epoch: {min_val_loss_epoch}"
        )

        min_train_loss = metrics["train_loss"].min()
        min_train_loss_epoch = metrics["train_loss"].idxmin()
        print(
            f"Experiment: {experiment}, Latent Dim: {latentdim_experiment}, Min Train Loss: {min_train_loss}, Min Train Loss Epoch: {min_train_loss_epoch}"
        )

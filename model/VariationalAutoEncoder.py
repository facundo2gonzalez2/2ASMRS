import json
from pathlib import Path
import numpy as np
from audio_utils import spectrogram2audio
from utils import train_val_split_no_overlap  # , eps
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
from utils import MetricTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from torch.nn.utils import weight_norm


def normalization(module: nn.Module, mode: str = "weight_norm"):
    if mode == "identity":
        return module
    elif mode == "weight_norm":
        return weight_norm(module)
    else:
        raise Exception(f"Normalization mode {mode} not supported")


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAEEncoder(nn.Module):
    def __init__(self, layers_size, latent_dim):
        super().__init__()
        self.shared = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(layers_size[i], layers_size[i + 1]),
                    nn.BatchNorm1d(layers_size[i + 1]),
                    nn.ELU(),
                )
                for i in range(len(layers_size) - 1)
            ]
        )
        self.fc_mu = nn.Linear(layers_size[-1], latent_dim)
        self.fc_logvar = nn.Linear(layers_size[-1], latent_dim)

    def forward(self, x):
        h = self.shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, layers_size):
        super().__init__()
        layers = []
        for i in range(len(layers_size) - 2):
            layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))
            layers.append(nn.BatchNorm1d(layers_size[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(layers_size[-2], layers_size[-1]))
        layers.append(nn.BatchNorm1d(layers_size[-1]))
        # layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(
        self, encoder_layers, decoder_layers, latent_dim, checkpoint_path=None
    ):
        super().__init__()
        self.encoder = VAEEncoder(encoder_layers, latent_dim)
        self.decoder = Decoder((latent_dim,) + tuple(decoder_layers))
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss_fn(self, x, x_hat, mu, logvar):
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - x_hat) ** 2)
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    # def loss_fn2(self, x, x_hat, mu, logvar):
    #     # Supongamos que x y x_hat son espectrogramas de magnitud
    #     recon_loss = torch.mean(torch.abs(x - x_hat))  # L1 suele sonar mejor que MSE
    #
    #     kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    #     return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def load_checkpoint(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        self.checkpoint_path = path

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss, recon_loss, kl_loss = self.loss_fn(x, x_hat, mu, logvar)
        self.log("train_loss", loss)
        self.log("train_recon", recon_loss)
        self.log("train_kl", kl_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss, recon_loss, kl_loss = self.loss_fn(x, x_hat, mu, logvar)
        self.log("val_loss", loss)
        self.log("val_recon", recon_loss)
        self.log("val_kl", kl_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def load_data(self, X, y, Xmax, db_min_norm, spec_in_db):
        self.X = X
        self.y = y
        self.Xmax = Xmax
        self.db_min_norm = db_min_norm
        self.spec_in_db = spec_in_db

    def split_data(
        self, validation_size=0.05, val_consecutive_rows=None, random_seed=42
    ):
        if val_consecutive_rows is not None:
            self.X_train, self.X_val, self.y_train, self.y_val = (
                train_val_split_no_overlap(
                    self.X,
                    self.y,
                    val_consecutive_rows,
                    validation_size,
                    random_seed=random_seed,
                )
            )
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X,
                self.y,
                test_size=validation_size,
                random_state=random_seed,
            )

        print("Train and val shapes", self.X_train.shape, self.X_val.shape)

    def predict(self, specgram):
        self.eval()
        with torch.no_grad():
            x_hat, mu, logvar = self(specgram)
        return x_hat

    def log_hyperparameters(self, **kwargs):
        print(kwargs)

        for k, v in kwargs.items():
            self.hparams[k] = v
        self.hparams["parameters"] = sum(p.numel() for p in self.parameters())
        self.save_hyperparameters(ignore=["checkpoint_path"])

    def train_model(
        self,
        learning_rate,
        beta,
        epochs=120,
        batch_size=512,
        log_path="tb_logs_vae",
        run_name="example_vae",
        accelerator="cpu",
        use_early_stopping=True,
    ):
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.beta = beta

        # DataLoaders
        dataset = TensorDataset(
            torch.tensor(self.X_train).float(), torch.tensor(self.X_train).float()
        )
        train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

        dataset = TensorDataset(
            torch.tensor(self.X_val).float(), torch.tensor(self.X_val).float()
        )
        val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

        logger = TensorBoardLogger(log_path, name=run_name)
        metric_tracker = MetricTracker()

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-6, patience=100, verbose=True, mode="min"
        )

        callbacks = [metric_tracker]
        if use_early_stopping:
            callbacks.append(early_stop_callback)

        trainer = pl.Trainer(
            max_epochs=epochs,
            enable_model_summary=False,
            logger=logger,
            callbacks=callbacks,
            accelerator=accelerator,
        )

        trainer.fit(
            self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )
        return trainer, metric_tracker

    def export_decoder(self):
        hps = self.hparams

        with torch.no_grad():
            mu, logvar = self.encoder(self.X)
            Z = mu.cpu().numpy()

        rangeZ = np.ceil(np.abs(Z).max(0))
        decoder = self.decoder

        output_path = Path("exported_models_vae", f"{self.run_name}.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        decoder.to("cpu").eval()
        example = torch.zeros(1, hps["encoder_layers"][-1])
        print("Tracing decoder")
        traced_script_module = torch.jit.trace(decoder, example)
        traced_script_module.save(output_path)  # type: ignore

        data = {
            "parameters": {
                "latent_dim": hps["encoder_layers"][-1],
                "sClip": hps["db_min_norm"],
                "sr": hps["target_sampling_rate"],
                "win_length": hps["win_length"],
                "xMax": hps["Xmax"],
                "zRange": [{"max": int(v), "min": -int(v)} for v in rangeZ],
            },
            "model_name": f"{self.run_name}.pt",
            "ztrack": Z.tolist(),
        }

        output_path = Path("exported_models_vae", f"{self.run_name}.json")

        with open(output_path, "w") as fp:
            json.dump(data, fp)

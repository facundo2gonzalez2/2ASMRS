from fire import Fire
from pathlib import Path
import yaml
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

from audio_utils import (
    get_spectrograms_from_audios,
)
from VariationalAutoEncoder import VariationalAutoEncoder
from multidimensional_wilcoxon_mann_whitney import multidimensional_ranksum


def get_latent_vectors(model_path, audio_paths, db_min_norm=-60, spec_in_db=True):
    """
    Carga un modelo y devuelve los vectores latentes (mu) para una lista de audios.
    """
    try:
        with open(Path(model_path, "hparams.yaml")) as file:
            hps = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: No se encontró hparams.yaml en {model_path}")
        return None

    checkpoints = list(Path(model_path, "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        print(f"Error: No hay checkpoints en {model_path}")
        return None
    checkpoint_path = checkpoints[0]

    print(f"--- Procesando audios para modelo: {model_path} ---")
    X, _, _, y = get_spectrograms_from_audios(
        audio_paths,
        hps["target_sampling_rate"],
        hps["win_length"],
        hps["hop_length"],
        db_min_norm=db_min_norm,
        spec_in_db=spec_in_db,
    )

    vae = VariationalAutoEncoder(
        encoder_layers=hps["encoder_layers"],
        decoder_layers=hps["decoder_layers"],
        latent_dim=hps["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    vae.eval()

    with torch.no_grad():
        mu, _ = vae.encoder(X)
        Z = mu.cpu().numpy()

    return Z, hps["latent_dim"], y.cpu().numpy(), hps


def get_latent_projection(model_path, audio_path, db_min_norm=-60, spec_in_db=True):
    """
    Carga un modelo, procesa el audio y devuelve la proyección 2D UMAP de los vectores latentes.
    """
    # 1. Cargar configuración
    res = get_latent_vectors(
        model_path,
        [audio_path],
        db_min_norm=db_min_norm,
        spec_in_db=spec_in_db,
    )
    if res is None:
        return None
    Z, latent_dim, _, _ = res

    # 5. UMAP
    # n_neighbors: controla cómo UMAP balancea estructura local vs global.
    # n_neighbors más bajo (ej 15) preserva estructura local (bueno para trayectorias).
    print(f"Ejecutando UMAP (Dim original: {Z.shape[1]} -> 2)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(Z)

    return embedding, latent_dim


def run_comparison(
    audio_path="umap_experiment/fur_elise_piano_cut.mp3",
    model_path_1="experiment_latent_dim_beta_piano_small/beta_vae_latentdim_8/version_0",
    model_path_2="experiment_latent_dim_piano_small/vae_latentdim_3/version_0",
    output_img_path="umap_experiment/comparison_trajectory_piano.png",
):
    # Obtener proyecciones
    res1 = get_latent_projection(model_path_1, audio_path)
    res2 = get_latent_projection(model_path_2, audio_path)

    if res1 is None or res2 is None:
        return

    emb1, dim1 = res1
    emb2, dim2 = res2

    # Configurar Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Datos para graficar
    experiments = [
        (axes[0], emb1, f"Model 1: Beta-VAE (Latent Dim {dim1})"),
        (axes[1], emb2, f"Model 2: Standard VAE (Latent Dim {dim2})"),
    ]

    for ax, emb, title in experiments:
        n_frames = len(emb)
        time_colors = np.arange(n_frames)

        # 1. TRAYECTORIA (Línea)
        # Graficamos una línea gris fina conectando los puntos secuencialmente.
        # Esto permite ver el "movimiento" de la canción.
        ax.plot(emb[:, 0], emb[:, 1], color="gray", alpha=0.4, linewidth=0.8, zorder=1)

        # 2. ESTADOS (Scatter)
        # Graficamos los puntos encima con el mapa de color temporal
        sc = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=time_colors,
            cmap="viridis",
            s=10,
            alpha=0.8,
            zorder=2,
        )

        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)

    # Barra de color compartida
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label="Tiempo (frames)")

    plt.suptitle(
        f"Comparación de Trayectorias Latentes: {Path(audio_path).name}", fontsize=14
    )

    # Guardar
    Path(output_img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path)
    print(f"Gráfico comparativo guardado en {output_img_path}")


def run_model_base_comparison():
    piano_audio = "umap_experiment/fur_elise_piano_cut.mp3"
    guitar_audio = "data_instruments/guitar/00_Jazz1-130-D_comp_hex.wav"
    voice_audio = "data_instruments/voice/m1_arpeggios_straight_o.wav"
    bass_audio = "umap_experiment/bass_cut.mp3"

    base_model_path = "base_model/base_model_no_beta/version_0"
    output_img_path = "umap_experiment/base_model_instruments_umap.png"

    instrument_audios = {
        "Piano": piano_audio,
        "Guitar": guitar_audio,
        "Voice": voice_audio,
        "Bass": bass_audio,
    }

    res = get_latent_vectors(base_model_path, list(instrument_audios.values()))
    if res is None:
        return
    Z, latent_dim, y, _ = res

    print(f"Ejecutando UMAP (Dim original: {latent_dim} -> 2)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(Z)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = {
        "Piano": "#1f77b4",
        "Guitar": "#ff7f0e",
        "Voice": "#2ca02c",
        "Bass": "#d62728",
    }

    instrument_names = list(instrument_audios.keys())

    print("--- Separación por pares (Avg U) ---")
    for i in range(len(instrument_names)):
        for j in range(i + 1, len(instrument_names)):
            mask = (y == i) | (y == j)
            Z_pair = Z[mask]
            y_pair = y[mask]
            U = multidimensional_ranksum(Z_pair, y_pair)
            avg_u = float(np.mean(U))
            print(
                f"{instrument_names[i]} vs {instrument_names[j]}: Avg U = {avg_u:.4f}"
            )

    for idx, instrument in enumerate(instrument_names):
        emb = embedding[y == idx]

        ax.plot(emb[:, 0], emb[:, 1], color=colors[instrument], alpha=0.5, linewidth=1)
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            color=colors[instrument],
            s=8,
            alpha=0.8,
            label=instrument,
        )

    ax.set_title("Base Model: Trayectorias Latentes por Instrumento")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    Path(output_img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path, dpi=200, bbox_inches="tight")
    print(f"Gráfico guardado en {output_img_path}")


if __name__ == "__main__":
    Fire(run_model_base_comparison)

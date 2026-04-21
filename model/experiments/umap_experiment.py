from fire import Fire
from pathlib import Path
import yaml
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
import sys

# Allow running this script directly via `python experiments/interpolate.py`
# by making the parent `model/` directory importable.
MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audio_utils import (
    get_spectrograms_from_audios,
)
from VariationalAutoEncoder import VariationalAutoEncoder
from multidimensional_wilcoxon_mann_whitney import multidimensional_ranksum


def get_latent_vectors(
    model_path,
    audio_paths,
    db_min_norm=-60,
    spec_in_db=True,
    trim_silence=False,
    remove_all_silence=False,
    silence_top_db=35,
    silence_frame_length=2048,
    silence_hop_length=512,
    encode_batch_size=4096,
):
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

    vae = VariationalAutoEncoder(
        encoder_layers=hps["encoder_layers"],
        decoder_layers=hps["decoder_layers"],
        latent_dim=hps["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    vae.eval()

    latent_batches = []
    audio_y_batches = []

    with torch.no_grad():
        for audio_idx, audio_path in enumerate(audio_paths):
            X, _, _, _ = get_spectrograms_from_audios(
                [audio_path],
                hps["target_sampling_rate"],
                hps["win_length"],
                hps["hop_length"],
                db_min_norm=db_min_norm,
                spec_in_db=spec_in_db,
                trim_silence=trim_silence,
                remove_all_silence=remove_all_silence,
                silence_top_db=silence_top_db,
                silence_frame_length=silence_frame_length,
                silence_hop_length=silence_hop_length,
            )

            if X is None or X.shape[0] == 0:
                continue

            n_frames = X.shape[0]
            for start_idx in range(0, n_frames, encode_batch_size):
                end_idx = min(start_idx + encode_batch_size, n_frames)
                X_chunk = X[start_idx:end_idx]
                mu, _ = vae.encoder(X_chunk)
                latent_batches.append(mu.cpu().numpy())

            audio_y_batches.append(np.full(n_frames, audio_idx, dtype=np.int32))
            del X

    if not latent_batches:
        print("Error: No se pudieron extraer vectores latentes de los audios.")
        return None

    Z = np.concatenate(latent_batches, axis=0)
    audio_y = np.concatenate(audio_y_batches, axis=0)

    return Z, hps["latent_dim"], audio_y, hps


def sample_latents_by_class(
    Z,
    y,
    max_frames_per_class=None,
    max_total_frames=None,
    random_seed=42,
):
    if max_frames_per_class is None and max_total_frames is None:
        return Z, y

    rng = np.random.default_rng(random_seed)
    selected_indices = []

    classes = np.unique(y)
    for class_id in classes:
        class_indices = np.flatnonzero(y == class_id)
        if max_frames_per_class is not None and len(class_indices) > max_frames_per_class:
            class_indices = rng.choice(class_indices, size=max_frames_per_class, replace=False)
        selected_indices.append(class_indices)

    if not selected_indices:
        return Z, y

    selected_indices = np.concatenate(selected_indices)

    if max_total_frames is not None and len(selected_indices) > max_total_frames:
        selected_indices = rng.choice(selected_indices, size=max_total_frames, replace=False)

    selected_indices = np.sort(selected_indices)
    return Z[selected_indices], y[selected_indices]


def get_latent_projection(
    model_path,
    audio_path,
    db_min_norm=-60,
    spec_in_db=True,
    trim_silence=False,
    remove_all_silence=False,
    silence_top_db=35,
    silence_frame_length=2048,
    silence_hop_length=512,
    encode_batch_size=4096,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_random_state=42,
):
    """
    Carga un modelo, procesa el audio y devuelve la proyección 2D UMAP de los vectores latentes.
    """
    # 1. Cargar configuración
    res = get_latent_vectors(
        model_path,
        [audio_path],
        db_min_norm=db_min_norm,
        spec_in_db=spec_in_db,
        trim_silence=trim_silence,
        remove_all_silence=remove_all_silence,
        silence_top_db=silence_top_db,
        silence_frame_length=silence_frame_length,
        silence_hop_length=silence_hop_length,
        encode_batch_size=encode_batch_size,
    )
    if res is None:
        return None
    Z, latent_dim, _, _ = res

    # 5. UMAP
    # n_neighbors: controla cómo UMAP balancea estructura local vs global.
    # n_neighbors más bajo (ej 15) preserva estructura local (bueno para trayectorias).
    print(f"Ejecutando UMAP (Dim original: {Z.shape[1]} -> 2)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
    )
    embedding = np.asarray(reducer.fit_transform(Z), dtype=np.float32)

    return embedding, latent_dim


def run_comparison(
    audio_path="umap_experiment/fur_elise_piano_cut.mp3",
    model_path_1="experiments_models/experiment_latent_dim_piano/beta_vae_latentdim_8/version_0",
    model_path_2="experiments_models/experiment_latent_dim_piano/vae_latentdim_3/version_0",
    output_img_path="umap_experiment/comparison_trajectory_piano.png",
    trim_silence=False,
    remove_all_silence=False,
    silence_top_db=35,
    silence_frame_length=2048,
    silence_hop_length=512,
    encode_batch_size=4096,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_random_state=42,
):
    # Obtener proyecciones
    res1 = get_latent_projection(
        model_path_1,
        audio_path,
        trim_silence=trim_silence,
        remove_all_silence=remove_all_silence,
        silence_top_db=silence_top_db,
        silence_frame_length=silence_frame_length,
        silence_hop_length=silence_hop_length,
        encode_batch_size=encode_batch_size,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
    )
    res2 = get_latent_projection(
        model_path_2,
        audio_path,
        trim_silence=trim_silence,
        remove_all_silence=remove_all_silence,
        silence_top_db=silence_top_db,
        silence_frame_length=silence_frame_length,
        silence_hop_length=silence_hop_length,
        encode_batch_size=encode_batch_size,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
    )

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
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Tiempo (frames)")

    plt.suptitle(f"Comparación de Trayectorias Latentes: {Path(audio_path).name}", fontsize=14)

    # Guardar
    Path(output_img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path)
    print(f"Gráfico comparativo guardado en {output_img_path}")


def run_model_base_comparison(
    base_model_path="inference_models/base_model/base_model_beta_0.001/version_0",
    output_img_path="umap_experiment/base_model_instruments_umap_new_beta_test.png",
    run_label="Base Model",
    scatter_alpha=0.4,
    trim_silence=True,
    remove_all_silence=True,
    silence_top_db=35,
    silence_frame_length=2048,
    silence_hop_length=512,
    encode_batch_size=4096,
    umap_n_neighbors=20,
    umap_min_dist=0.1,
    umap_random_state=42,
    max_frames_per_class=20000,
    max_total_frames=60000,
    max_stats_frames_per_pair=10000,
):
    instrument_folders = {
        "Piano": "data_instruments_small/piano",
        "Guitar": "data_instruments_small/guitar",
        "Voice": "data_instruments_small/voice",
        "Bass": "data_instruments_small/bass",
    }

    all_audio_paths = []
    instrument_map = {}
    audio_idx = 0

    for instr_idx, (instrument, folder) in enumerate(instrument_folders.items()):
        paths = list(Path(folder).rglob("*.wav")) + list(Path(folder).rglob("*.mp3"))
        paths = sorted(paths)
        if not paths:
            print(f"Warning: No audio files found in {folder}")
        for p in paths:
            all_audio_paths.append(str(p))
            instrument_map[audio_idx] = instr_idx
            audio_idx += 1

    if not all_audio_paths:
        print("Error: No audio files found in any specified instrument folder.")
        return

    res = get_latent_vectors(
        base_model_path,
        all_audio_paths,
        trim_silence=trim_silence,
        remove_all_silence=remove_all_silence,
        silence_top_db=silence_top_db,
        silence_frame_length=silence_frame_length,
        silence_hop_length=silence_hop_length,
        encode_batch_size=encode_batch_size,
    )
    if res is None:
        return
    Z, latent_dim, audio_y, _ = res
    y = np.array([instrument_map[int(val)] for val in audio_y])

    Z, y = sample_latents_by_class(
        Z,
        y,
        max_frames_per_class=max_frames_per_class,
        max_total_frames=max_total_frames,
        random_seed=42,
    )

    print(f"Ejecutando UMAP (Dim original: {latent_dim} -> 2)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
    )
    embedding = np.asarray(reducer.fit_transform(Z), dtype=np.float32)

    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = {
        "Piano": "#1f77b4",
        "Guitar": "#ff7f0e",
        "Voice": "#2ca02c",
        "Bass": "#d62728",
    }

    instrument_names = list(instrument_folders.keys())

    print("--- Separación por pares (Avg U) ---")
    n_instr = len(instrument_names)
    separation_matrix = np.zeros((n_instr, n_instr), dtype=np.float32)
    rng = np.random.default_rng(42)
    for i in range(n_instr):
        for j in range(i + 1, n_instr):
            mask = (y == i) | (y == j)
            Z_pair = Z[mask]
            y_pair = y[mask]
            if len(Z_pair) > max_stats_frames_per_pair:
                sampled_indices = rng.choice(len(Z_pair), size=max_stats_frames_per_pair, replace=False)
                Z_pair = Z_pair[sampled_indices]
                y_pair = y_pair[sampled_indices]
            U = multidimensional_ranksum(Z_pair, y_pair)
            avg_u = float(np.mean(U))
            separation_matrix[i, j] = avg_u
            separation_matrix[j, i] = avg_u
            print(f"{instrument_names[i]} vs {instrument_names[j]}: Avg U = {avg_u:.4f}")

    for idx, instrument in enumerate(instrument_names):
        emb = np.asarray(embedding[y == idx], dtype=np.float32)

        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            color=colors[instrument],
            s=8,
            alpha=scatter_alpha,
            label=instrument,
            edgecolors="none",
        )

    ax.set_title(f"{run_label}: Trayectorias Latentes por Instrumento")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)
    legend = ax.legend()
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)  # type: ignore

    Path(output_img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path, dpi=200, bbox_inches="tight")
    print(f"Gráfico guardado en {output_img_path}")

    # Matriz de separación por pares (Avg U)
    fig_mat, ax_mat = plt.subplots(figsize=(7, 6))
    im = ax_mat.imshow(separation_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax_mat.set_xticks(range(n_instr))
    ax_mat.set_yticks(range(n_instr))
    ax_mat.set_xticklabels(instrument_names)
    ax_mat.set_yticklabels(instrument_names)
    for i in range(n_instr):
        for j in range(n_instr):
            val = separation_matrix[i, j]
            text_color = "white" if val < 0.6 else "black"
            ax_mat.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)
    fig_mat.colorbar(im, ax=ax_mat, label="Avg U (1 = mayor separación)")
    ax_mat.set_title(f"{run_label}: Matriz de Separación por Pares (Avg U)")

    matrix_path = Path(output_img_path).with_name(f"{Path(output_img_path).stem}_separation_matrix.png")
    plt.savefig(matrix_path, dpi=200, bbox_inches="tight")
    print(f"Matriz de separación guardada en {matrix_path}")
    plt.close(fig_mat)


if __name__ == "__main__":
    Fire()

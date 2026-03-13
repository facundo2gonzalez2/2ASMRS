import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from scipy.spatial.distance import cosine
import numpy as np
from scipy import linalg

# --- Configuración ---
# Usamos el modelo MERT-v1-95M (más ligero) o MERT-v1-330M (más preciso)
MODEL_ID = "m-a-p/MERT-v1-95M"
TARGET_SAMPLE_RATE = 24000  # MERT fue entrenado con audios a 24k
FAD_SIMILARITY_K = 585.0

print(f"Cargando modelo {MODEL_ID}...")
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)


def load_and_preprocess_audio(path):
    """
    Carga el audio, lo re-muestrea a 24kHz y lo prepara para el modelo.
    """
    waveform, sample_rate = torchaudio.load(path)

    # 1. Convertir a Mono si es estéreo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 2. Re-muestrear si es necesario (MERT necesita 24kHz)
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    # 3. Procesar con el feature extractor
    inputs = processor(
        waveform.squeeze(),  # type: ignore
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def get_matrix_embedding(path):
    """
    Pasa el audio por MERT y obtiene la matriz de embeddings temporales.
    """
    inputs = load_and_preprocess_audio(path)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Devolemos la matriz con forma (Tiempo, Dimensiones)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.squeeze().numpy()


def get_embedding(path):
    """
    Pasa el audio por MERT y obtiene el vector promedio.
    """
    inputs = load_and_preprocess_audio(path)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # MERT tiene muchas capas. La última capa (last_hidden_state) suele ser buena,
    # pero a veces un promedio de las últimas capas captura mejor la tímbrica.
    # Aquí usamos la última capa promediada en el tiempo.

    # shape: [batch, time_steps, hidden_size] -> [batch, hidden_size]
    last_hidden_state = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_state, dim=1)

    return embedding.squeeze().numpy()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calcula la Frechet Distance entre dos distribuciones multivariadas.
    Basado en la ecuación (1) del paper [2]:
    FAD = ||mu_r - mu_t||^2 + tr(Sigma_r + Sigma_t - 2*sqrt(Sigma_r * Sigma_t))
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    # Forzamos simetría para reducir inestabilidad numérica en sqrtm.
    sigma1 = (sigma1 + sigma1.T) / 2.0
    sigma2 = (sigma2 + sigma2.T) / 2.0

    # 1. Distancia euclidiana al cuadrado entre las medias
    diff = mu1 - mu2
    mean_term = np.dot(diff, diff)

    # 2. Término de la traza (Covarianza)
    # Con pocos frames frente a muchas features, las covarianzas suelen ser
    # singulares. Regularizamos progresivamente hasta obtener una sqrtm usable.
    covmean = None
    identity = np.eye(sigma1.shape[0], dtype=np.float64)
    for attempt in range(6):
        regularization = eps * (10**attempt)
        cov_prod = (sigma1 + identity * regularization).dot(
            sigma2 + identity * regularization
        )
        candidate, _ = linalg.sqrtm(cov_prod, disp=False)

        if not np.isfinite(candidate).all():
            continue

        if np.iscomplexobj(candidate):
            max_imag = float(np.max(np.abs(candidate.imag)))
            max_real = float(np.max(np.abs(candidate.real)))
            tolerance = max(1e-3, max_real * 1e-3)
            if max_imag > tolerance:
                continue
            candidate = candidate.real

        covmean = candidate
        break

    if covmean is None:
        raise ValueError(
            "No se pudo estabilizar el cálculo de FAD para las matrices de covarianza."
        )

    tr_covmean = np.trace(covmean)  # type: ignore

    # Fórmula completa
    return mean_term + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_audio_similarity_fad(embeddings_a, embeddings_b):
    """
    Calcula una similitud basada en FAD entre dos pistas de audio.
    Se transforma la distancia con: similitud = exp(-distancia / k).

    Args:
        embeddings_a (np.array): Matriz de forma (N_frames, Dimensiones) para el Audio A.
        embeddings_b (np.array): Matriz de forma (M_frames, Dimensiones) para el Audio B.

    Returns:
        float: Similitud en rango (0, 1], donde mayor es más similar.
    """
    # Validaciones previas
    if embeddings_a.ndim != 2 or embeddings_b.ndim != 2:
        raise ValueError(
            "Los embeddings deben ser matrices 2D (Frames x Features). No uses el vector promediado."
        )

    # Calcular estadísticas (Mu y Sigma) para el Audio A
    mu_a = np.mean(embeddings_a, axis=0)
    sigma_a = np.cov(embeddings_a, rowvar=False)

    # Calcular estadísticas (Mu y Sigma) para el Audio B
    mu_b = np.mean(embeddings_b, axis=0)
    sigma_b = np.cov(embeddings_b, rowvar=False)

    # Calcular distancia FAD y transformarla a similitud
    fad_distance = calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
    fad_similarity = float(np.exp(-fad_distance / FAD_SIMILARITY_K))

    return fad_similarity


def get_audio_similarit_fad(embeddings_a, embeddings_b):
    """
    Alias compatible con typo histórico del nombre de la función.
    """
    return get_audio_similarity_fad(embeddings_a, embeddings_b)


def get_fad_distance_between_files(file_a, file_b):
    emb_a = get_matrix_embedding(file_a)
    emb_b = get_matrix_embedding(file_b)
    return get_audio_similarity_fad(emb_a, emb_b)


def test_fad_score():
    # --- Ejemplo de uso simulado ---
    # Supongamos que usas MERT y obtienes una matriz de (Tiempo, 768)
    # Ejemplo: Audio A dura 10 seg, Audio B dura 12 seg.
    # Dimensiones de MERT suelen ser 768 [5] o 1024 dependiendo del modelo.
    # Simulación de embeddings (reemplaza esto con tu función de extracción MERT)
    reference_file = "./outputs/reference_tracks/c-major-scale-90710.mp3"
    instrument_file = (
        "./outputs/interpolate_fine_tuned/exp_0.0_guitar_c-major-scale-90710.wav"
    )

    instrument_file = (
        "./outputs/interpolate_fine_tuned/exp_1.0_guitar_c-major-scale-90710.wav"
    )

    emb_audio_1 = get_matrix_embedding(reference_file)
    emb_audio_2 = get_matrix_embedding(instrument_file)

    score = get_audio_similarity_fad(emb_audio_1, emb_audio_2)
    print(f"Similitud FAD: {score}")


def get_cosine_similarity(path_1, path_2):
    emb1 = get_embedding(path_1)
    emb2 = get_embedding(path_2)
    similarity = 1 - cosine(emb1, emb2)
    return similarity


def evaluate_interpolation_reconstruction(
    reference_file: str,
    reconstruced_per_alpha: list[str],
):
    embeddings = [get_embedding(f) for f in reconstruced_per_alpha]
    reference_embedding = get_embedding(reference_file)

    print("--- Comparación de interpolación por alfa ---")
    print("Instrumento A vs Referencia:")
    for i, emb in enumerate(embeddings):
        similarity = 1 - cosine(emb, reference_embedding)
        fad_distance = get_fad_distance_between_files(
            reference_file, reconstruced_per_alpha[i]
        )
        print(
            f" Alpha {i / (len(embeddings) - 1):.2f}: Similitud = {similarity:.4f}, FAD = {fad_distance:.4f}"
        )


def get_embedding_similarity_between(file_1: str, file_2: str):
    # 1. Obtener embeddings
    print("Procesando audios...")
    # Nota: Asegúrate de tener archivos reales o el script fallará al cargar
    try:
        emb1 = get_embedding(file_1)
        emb2 = get_embedding(file_2)
        # emb3 = get_embedding(file_3) # Descomentar para probar con voz

        # 2. Calcular Similitud del Coseno
        # La distancia coseno es 1 - similitud.
        # Queremos similitud, así que hacemos 1 - distancia.
        similarity = 1 - cosine(emb1, emb2)

        print("--- Resultados ---")
        print(f"Similitud entre {file_1} y {file_2}: {similarity:.4f}")

        # Interpretación
        if similarity > 0.8:
            print(">> Muy similares (probablemente mismo instrumento/estilo)")
        elif similarity > 0.5:
            print(">> Algo similares (comparten características musicales)")
        else:
            print(">> Diferentes")

    except Exception as e:
        print(f"Error (probablemente no encontraste los archivos de audio): {e}")


if __name__ == "__main__":
    # Ejemplo de comparación entre dos audios
    # compare_two_audios(
    #     "./outputs/reference_tracks/c-major-scale-90710.mp3",
    #     "./outputs/interpolate_fine_tuned/exp_0.0_guitar_c-major-scale-90710.wav",
    # )

    reference_file = "./outputs/reference_tracks/c-major-scale-child-102262.mp3"

    reconstruced_per_alpha = [
        "./outputs/interpolate_fine_tuned/exp_0.0_voice_c-major-scale-child-102262.wav",
        "./outputs/interpolate_fine_tuned/exp_0.25_voice_c-major-scale-child-102262.wav",
        "./outputs/interpolate_fine_tuned/exp_0.5_voice_c-major-scale-child-102262.wav",
        "./outputs/interpolate_fine_tuned/exp_0.75_voice_c-major-scale-child-102262.wav",
        "./outputs/interpolate_fine_tuned/exp_1.0_voice_c-major-scale-child-102262.wav",
    ]

    evaluate_interpolation_reconstruction(reference_file, reconstruced_per_alpha)

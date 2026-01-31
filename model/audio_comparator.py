import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from scipy.spatial.distance import cosine

# --- Configuración ---
# Usamos el modelo MERT-v1-95M (más ligero) o MERT-v1-330M (más preciso)
MODEL_ID = "m-a-p/MERT-v1-95M"
TARGET_SAMPLE_RATE = 24000  # MERT fue entrenado con audios a 24k

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


def get_cosine_similarity(path_1, path_2):
    emb1 = get_embedding(path_1)
    emb2 = get_embedding(path_2)
    similarity = 1 - cosine(emb1, emb2)
    return similarity


def compare_interpolation(
    reference_file: str,
    files_instrument_a: list[str],
    files_instrument_b: list[str],
    witness_file: str,
):
    """
    Asumiento que files_instrument_a y files_instrument_b son listas de archivos de tipo:
    ["guitar_output_alpha_0.0.wav", "guitar_output_alpha_0.25.wav", ..., "guitar_output_alpha_1.0.wav"]
    ["piano_output_alpha_0.0.wav", "piano_output_alpha_0.25.wav", ..., "piano_output_alpha_1.0.wav"]
    Y witness_file es un archivo de audio real de otro instrumento diferente.
    """
    embeddings_a = [get_embedding(f) for f in files_instrument_a]
    embeddings_b = [get_embedding(f) for f in files_instrument_b]
    witness_embedding = get_embedding(witness_file)
    reference_embedding = get_embedding(reference_file)

    print("--- Comparación de interpolaciones ---")
    print("Instrumento A vs Referencia:")
    for i, emb in enumerate(embeddings_a):
        similarity = 1 - cosine(emb, reference_embedding)
        print(f" Alpha {i / (len(embeddings_a) - 1):.2f}: Similitud = {similarity:.4f}")

    print("\nInstrumento B vs Referencia:")
    for i, emb in enumerate(embeddings_b):
        similarity = 1 - cosine(emb, reference_embedding)
        print(f" Alpha {i / (len(embeddings_b) - 1):.2f}: Similitud = {similarity:.4f}")

    print("\nTestigo vs Referencia:")
    similarity = 1 - cosine(witness_embedding, reference_embedding)
    print(f"Similitud = {similarity:.4f}")


def compare_two_audios(file_1: str, file_2: str):
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

    reference_file = "./outputs/reference_tracks/c-major-scale-90710.mp3"

    files_instrument_a = [
        "./outputs/interpolate_fine_tuned/exp_0.0_guitar_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_0.25_guitar_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_0.5_guitar_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_0.75_guitar_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_1.0_guitar_c-major-scale-90710.wav",
    ]

    files_instrument_b = [
        "./outputs/interpolate_fine_tuned/exp_0.0_piano_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_0.25_piano_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_0.5_piano_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_0.75_piano_c-major-scale-90710.wav",
        "./outputs/interpolate_fine_tuned/exp_1.0_piano_c-major-scale-90710.wav",
    ]

    witness_file = "./outputs/witness_tracks/violin_c-major-scale-90710.wav"

    compare_interpolation(
        reference_file, files_instrument_a, files_instrument_b, witness_file
    )

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from torch import nn
import numpy as np
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
        waveform.squeeze(),
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


# --- Ejecución ---

# Rutas a tus archivos
file_1 = "./outputs/reference_tracks/c-major-scale-90710.mp3 "
file_2 = "./outputs/interpolate_fine_tuned/exp_0.0_guitar_c-major-scale-90710.wav"
file_3 = "voz_humana.wav"  # Cambia esto por tu archivo

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

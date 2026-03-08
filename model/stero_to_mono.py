import os
import librosa
import soundfile as sf


def convertir_carpeta_a_mono(carpeta_entrada, carpeta_salida):
    # Crea la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Recorrer todos los archivos de la carpeta
    for archivo in os.listdir(carpeta_entrada):
        if archivo.lower().endswith(".wav"):
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            ruta_salida = os.path.join(carpeta_salida, archivo)

            try:
                # librosa.load con mono=True convierte automáticamente estéreo a mono
                # sr=None mantiene la frecuencia de muestreo original del archivo
                audio, sample_rate = librosa.load(ruta_entrada, sr=None, mono=True)

                # Guarda el nuevo archivo en la carpeta de salida
                sf.write(ruta_salida, audio, sample_rate)
                print(f"✅ Convertido: {archivo}")

            except Exception as e:
                print(f"❌ Error procesando {archivo}: {e}")

    print("¡Proceso terminado!")


# --- Configuración ---
# Cambia estas rutas por las tuyas
CARPETA_INPUT = "data_instruments/guitar"
CARPETA_OUTPUT = "data_instruments/guitar_mono"

convertir_carpeta_a_mono(CARPETA_INPUT, CARPETA_OUTPUT)

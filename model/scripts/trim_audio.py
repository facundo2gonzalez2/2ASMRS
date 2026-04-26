import os
import sys
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def trim_silence(file_path, silence_threshold=-50, min_silence_len=100):
    """
    Corta los silencios al principio y al final de un archivo de audio.

    :param file_path: Ruta del archivo de audio original.
    *param silence_threshold: Umbral en dBFS por debajo del cual se considera silencio.
    :param min_silence_len: Longitud mínima en milisegundos para ser considerado silencio.
    """
    print(f"Procesando: {file_path}...")

    try:
        # Cargar el archivo de audio
        audio = AudioSegment.from_file(file_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'.")
        return
    except Exception as e:
        print(f"Error al cargar el audio. ¿Tienes FFmpeg instalado? Detalles: {e}")
        return

    # Detectar las partes que NO son silencio
    # detect_nonsilent devuelve una lista de listas: [[start1, end1], [start2, end2], ...]
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold)

    if not nonsilent_ranges:
        print("El audio parece estar completamente en silencio o el umbral es muy alto.")
        return

    # Tomar el inicio del primer fragmento con sonido y el final del último
    start_trim = nonsilent_ranges[0][0]
    end_trim = nonsilent_ranges[-1][1]

    # Cortar el audio original usando los milisegundos obtenidos
    trimmed_audio = audio[start_trim:end_trim]

    # Generar el nuevo nombre de archivo
    # os.path.splitext separa el nombre de la extensión (ej. "audio", ".mp3")
    file_name, file_extension = os.path.splitext(file_path)
    output_path = f"{file_name}_trimmed{file_extension}"

    # Exportar el audio recortado
    # Pydub requiere el formato sin el punto, así que quitamos el primer caracter de la extensión
    export_format = file_extension.replace(".", "")
    trimmed_audio.export(output_path, format=export_format)

    print(f"¡Listo! Archivo guardado como: {output_path}")
    print(f"Tiempo original: {len(audio)/1000:.2f}s | Tiempo final: {len(trimmed_audio)/1000:.2f}s")


if __name__ == "__main__":
    # Verificar que se haya pasado un argumento (el archivo de audio)
    if len(sys.argv) < 2:
        print("Uso correcto: python trimmer.py <ruta_al_archivo_de_audio>")
        sys.exit(1)

    input_file = sys.argv[1]
    trim_silence(input_file)

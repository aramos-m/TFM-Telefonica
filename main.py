import sys
import os
from pathlib import Path
from subtitulacion.utilidades import convertir_mp4_a_wav
from subtitulacion.transcribir import transcribir_audio_a_srt

def procesar_directorio(directorio):
    for archivo in Path(directorio).glob("*.mp4"):
        ruta_wav = convertir_mp4_a_wav(str(archivo))
        transcribir_audio_a_srt(ruta_wav)

def procesar_archivo(ruta_mp4):
    ruta_wav = convertir_mp4_a_wav(ruta_mp4)
    transcribir_audio_a_srt(ruta_wav)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <archivo_video.mp4> [<archivo2.mp4> ...] o <directorio>")
        sys.exit(1)

    rutas = sys.argv[1:]

    for ruta in rutas:
        if os.path.isdir(ruta):
            procesar_directorio(ruta)
        elif os.path.isfile(ruta) and ruta.endswith(".mp4"):
            procesar_archivo(ruta)
        else:
            print(f"Se omite '{ruta}': no es un archivo .mp4 válido ni un directorio.")

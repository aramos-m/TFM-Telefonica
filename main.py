import os
from pathlib import Path
from subtitulacion.utilidades import convertir_mp4_a_wav
from subtitulacion.transcribir import transcribir_audio_a_srt
from diarizacion.diarizacion_voces_caras import generate_face_detections

def procesar_directorio(directorio):
    archivos_mp4 = list(Path(directorio).glob("*.mp4"))
    if not archivos_mp4:
        return False  # No hay archivos mp4
    for archivo in archivos_mp4:
        ruta_wav = convertir_mp4_a_wav(str(archivo))
        #transcribir_audio_a_srt(ruta_wav)
        generate_face_detections(str(archivo))
    return True  # Se procesaron archivos

if __name__ == "__main__":
    os.environ["ORT_LOG_LEVEL"] = "ERROR"            # alternativa 1
    os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3" 
    ruta = "./data"
    if os.path.isdir(ruta):
        hay_mp4 = procesar_directorio(ruta)
        if not hay_mp4:
            print("No hay archivos .mp4 en 'data'.")
    else:
        print("Error al leer del directorio 'data'.")
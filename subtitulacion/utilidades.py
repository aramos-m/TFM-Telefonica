from moviepy.editor import VideoFileClip

def convertir_mp4_a_wav(ruta_video):
    """
    Convierte un archivo de video MP4 a un archivo de audio WAV.
    """
    video = VideoFileClip(ruta_video)
    audio = video.audio
    ruta_audio = ruta_video.replace(".mp4", ".wav")
    audio.write_audiofile(ruta_audio)
    video.close()
    return ruta_audio

def formato_srt(segundos):
    """
    Convierte segundos en formato SRT (HH:MM:SS,mmm)
    """
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    seg = int(segundos % 60)
    milisegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02}:{minutos:02}:{seg:02},{milisegundos:03}"

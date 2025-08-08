from moviepy import VideoFileClip

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

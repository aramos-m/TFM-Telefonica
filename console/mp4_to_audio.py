from moviepy.editor import VideoFileClip
from pathlib import Path

# Instalción: bibliote y dependencia
# pip install moviepy==1.0.3 numpy>=1.18.1 imageio>=2.5.0 decorator>=4.3.0 tqdm>=4.0.0 Pillow>=7.0.0 scipy>=1.3.0 pydub>=0.23.0 audiofile>=0.0.0 opencv-python>=4.5

class Mp4toAudio:
    
    def ExtraerAudio(self, miVideo, directorio_salida=None):
        
        archivo = Path(miVideo)
        # Cambiar la extensión de .mp4 a .wav
        archivo_wav = archivo.with_suffix(".wav")
        
        # Cargar el video
        video = VideoFileClip(miVideo)
        
        # Extraer el audio del video
        audio_file = video.audio          
        
        # Guardar el archivo de audio
        audio_file.write_audiofile( archivo_wav )       
        
        return archivo_wav

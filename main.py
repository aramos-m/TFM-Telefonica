from pathlib import Path
import audio_to_srt as atos
from moviepy.editor import VideoFileClip

ruta = './data/'

def main():
    carpeta = Path(ruta)

    for archivo in carpeta.iterdir():

        video = VideoFileClip(str(archivo.absolute()))
        audio_file = video.audio          
        audio_file.write_audiofile( "temp.wav" )     
        print(f"Transcribiendo: {archivo.absolute()}")
        atos.audio_to_srt("temp.wav")

main()
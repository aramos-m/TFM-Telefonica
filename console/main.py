import mp4_to_audio as mp
import audio_to_text_srt_progress as audtxt

# Extraer Subtitular vídeo
path = './data/'
fileName = "01.mp4"
file = path + fileName

mp4 = mp.Mp4toAudio()
audio_file_path = mp4.ExtraerAudio( file )

if audio_file_path != None:
    print("\nSe ha extraido el audio exitosamente\n")
    at = audtxt.AudioToTextStrProgress()
    at.save_as_srt(audio_file_path)

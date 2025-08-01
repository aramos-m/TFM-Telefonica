import warnings
import whisper
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm

class AudioToTextStrProgress:
    def __init__(self):
        warnings.simplefilter("ignore", category=FutureWarning)
        self.model = whisper.load_model("base")

    def _format_srt_time(self, seconds):
        """Convierte segundos a formato SRT hh:mm:ss,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def _transcribe_audio(self, audio_path):
        try:
            audio_path = Path(audio_path).resolve()
            if not audio_path.exists():
                print(f"Archivo de audio no encontrado: {audio_path}")
                return []

            print(f"Procesando: {audio_path}")

            # Cargar el audio completo
            audio_array, sr = librosa.load(str(audio_path), sr=16000)
            duration = librosa.get_duration(y=audio_array, sr=sr)

            # Dividir el audio en fragmentos de 30s para procesar poco a poco
            chunk_size = 30  # segundos
            total_chunks = int(np.ceil(duration / chunk_size))
            transcript_data = []

            # Procesar cada fragmento con barra de progreso
            with tqdm(total=total_chunks, desc="Transcribiendo", unit="fragmento") as pbar:
                for i in range(total_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, duration)
                    
                    # Extraer el fragmento del audio
                    chunk = audio_array[int(start * sr): int(end * sr)]

                    # Transcribir el fragmento
                    result = self.model.transcribe(chunk, fp16=False)
                    
                    # Guardar los segmentos con tiempos
                    for seg in result.get("segments", []):
                        start_time = self._format_srt_time(start + seg["start"])
                        end_time = self._format_srt_time(start + seg["end"])
                        text = seg["text"]
                        transcript_data.append((start_time, end_time, text))

                    pbar.update(1)  # Avanzar la barra de progreso

            return transcript_data

        except Exception as e:
            print(f"Error al transcribir el audio {audio_path}: {e}")
            return []

    def save_as_srt(self, audio_file_path):
        print(f"Generando archivo SRT desde: {audio_file_path}")

        transcript_data = self._transcribe_audio(audio_file_path)

        if transcript_data:
            srt_file_path = str(Path(audio_file_path).with_suffix('')) + '.srt'
            
            with open(srt_file_path, 'w', encoding='utf-8') as file:
                for i, (start, end, text) in enumerate(transcript_data, start=1):
                    file.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                    
            print(f"Archivo SRT guardado en: {srt_file_path}")
            return srt_file_path
        else:
            print("No se pudo transcribir el audio.")
            return None

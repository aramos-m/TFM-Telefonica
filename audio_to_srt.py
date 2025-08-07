import whisper
from datetime import timedelta
from pathlib import Path

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds))) + "," + str(int((seconds % 1) * 1000)).zfill(3)

def export_srt(segments, filename="output.srt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"Subtítulos exportados a {filename}")

def audio_to_srt(audio_file_path):
    audio_file_path = str(audio_file_path)  # Asegura que sea str, no Path

    print("Cargando modelo Whisper...")
    model = whisper.load_model("small")

    print("Transcribiendo audio...")
    result = model.transcribe(
        audio_file_path,
        language="es",
        fp16=False,
        verbose=True,
        condition_on_previous_text=True,
        temperature=0.2,
        best_of=5,
        initial_prompt="este audio menciona la frase hecha en español 'en fin Serafín'"
    )

    # Exporta los resultados
    export_srt(result["segments"], filename="transcripcion.srt")
    print("Transcripción y subtítulos completados.")

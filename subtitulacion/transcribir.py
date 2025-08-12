import whisper
from subtitulacion.utilidades import formato_srt

def transcribir_audio_a_srt(ruta_audio):
    """
    Transcribe un archivo de audio y guarda la transcripción en formato SRT usando Whisper.
    """
    # whisper_models = ["tiny", "base", "small", "medium", "large"]
    print("Cargando modelo Whisper...")
    modelo = whisper.load_model("medium")

    print("Transcribiendo audio...")
    resultado = modelo.transcribe(
        ruta_audio,
        language="es",
        fp16=False, 
        verbose=True,
        condition_on_previous_text=True,
        temperature=0.0,
        best_of=5,
        initial_prompt="este audio menciona la frase hecha en español 'en fin Serafín'"
    )

    ruta_srt = ruta_audio.replace(".wav", ".srt")
    with open(ruta_srt, "w", encoding="utf-8") as archivo_srt:
        for i, segmento in enumerate(resultado["segments"]):
            inicio = formato_srt(segmento["start"])
            fin = formato_srt(segmento["end"])
            archivo_srt.write(f"{i+1}\n{inicio} --> {fin}\n{segmento['text'].strip()}\n\n")

    print(f"Transcripción guardada en '{ruta_srt}'")

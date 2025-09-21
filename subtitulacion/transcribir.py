import whisper
from subtitulacion.utilidades import formato_srt
from pathlib import Path

def transcribir_audio_a_srt(ruta_audio):
    """
    Transcribe un archivo de audio y guarda la transcripción en formato SRT usando Whisper.
    """
    # whisper_models = ["tiny", "base", "small", "medium", "large"]
    print("Cargando modelo Whisper...")
    modelo = whisper.load_model("large", "cpu")

    print("Transcribiendo audio...")
    resultado = modelo.transcribe(
        ruta_audio,
        language="es",
        fp16=False,
        verbose=True,
        condition_on_previous_text=False,
        temperature=0,
        no_speech_threshold=0.9,           # más agresivo detectando silencio
        logprob_threshold=-0.5            # descarta hipótesis flojas
    )
    out_dir = Path(ruta_audio).parent.parent / "outdir"
    out_dir.mkdir(exist_ok=True)
    ruta_srt = out_dir/ (Path(ruta_audio).stem + "_es.srt")
    with open(ruta_srt, "w", encoding="utf-8") as archivo_srt:
        for i, segmento in enumerate(resultado["segments"]):
            inicio = formato_srt(segmento["start"])
            fin = formato_srt(segmento["end"])
            archivo_srt.write(f"{i+1}\n{inicio} --> {fin}\n{segmento['text'].strip()}\n\n")

    print(f"Transcripción guardada en '{ruta_srt}'")

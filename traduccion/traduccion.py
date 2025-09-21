# ==== SRT Translator (CLI/función) — ES->EN y EN->XX ====
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path

# Mapas de modelos Marian (en ambos pasos usamos MarianMT)
ES_EN_MODEL = "Helsinki-NLP/opus-mt-es-en"
EN_TO_MODEL: Dict[str, str] = {
    "de": "Helsinki-NLP/opus-mt-en-de",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
    "sv": "Helsinki-NLP/opus-mt-en-sv",
    "da": "Helsinki-NLP/opus-mt-en-da",
    "cs": "Helsinki-NLP/opus-mt-en-cs",
    "pl": "gsarti/opus-mt-tc-en-pl",
    "uk": "Helsinki-NLP/opus-mt-en-uk",
    "el": "Helsinki-NLP/opus-mt-en-el",
    "he": "Helsinki-NLP/opus-mt-en-he",
    "tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "ro": "Helsinki-NLP/opus-mt-en-ro",
    "zh-hans": "Helsinki-NLP/opus-mt-en-zh",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "ja": "Helsinki-NLP/opus-mt-en-ja",
    "ko": "Helsinki-NLP/opus-mt-tc-big-en-ko"
}

# -------- Utilidades SRT --------
def parse_srt(text: str) -> List[Dict[str, str]]:
    """Parseo sencillo de SRT en bloques con 'times' y 'text'."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = [b for b in text.split("\n\n") if b.strip()]
    out = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 2:
            continue
        times = lines[1].strip()
        txt = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        out.append({"times": times, "text": txt})
    return out

def write_srt(entries: List[Dict[str, str]]) -> str:
    """Reconstruye texto SRT renumerando 1..N."""
    out = []
    for i, e in enumerate(entries, 1):
        out += [str(i), e["times"], e["text"], ""]
    return "\n".join(out).rstrip() + "\n"

def build_translator(model_name: str, device: int):
    """
    Carga SIEMPRE en formato safetensors para evitar torch.load y el requisito de torch>=2.6.
    Si el repositorio no tiene safetensors, lanza un error claro para que cambies de modelo.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=True,   # <-- clave
            torch_dtype=None        # CPU
        )
    except Exception as e:
        msg = str(e)
        raise RuntimeError(
            f"El modelo '{model_name}' no se pudo cargar en formato safetensors.\n"
            f"Asegúrate de que ese repo tenga pesos .safetensors o usa un modelo alternativo.\n"
            f"Error original: {msg}"
        )
    return pipeline("translation", model=mdl, tokenizer=tok, device=device)

def translate_entries(entries_src: List[Dict[str, str]], translator, batch_size=16, print_lines=False, src_lang="SRC", tgt_code="TGT"):
    """Traduce entries línea a línea pero procesando por lotes (solo líneas traducibles)."""
    total = len(entries_src)
    out_entries = []
    for i, e in enumerate(entries_src, start=1):
        src_lines = e["text"].split("\n") if e["text"] else [""]
        tgt_lines = [""] * len(src_lines)

        idxs, batch = [], []
        for j, line in enumerate(src_lines):
            s = line.strip()
            if s:
                idxs.append(j); batch.append(s)

        if batch:
            outs = translator(batch, max_length=512, batch_size=batch_size)
            outs_text = [o["translation_text"] for o in outs]
            for k, j in enumerate(idxs):
                tgt_lines[j] = outs_text[k]
                if print_lines:
                    es = src_lines[j]; en = tgt_lines[j]
                    es_s = es if len(es) <= 80 else es[:77] + "..."
                    en_s = en if len(en) <= 80 else en[:77] + "..."
                    print(f"[{i}/{total}] línea {j+1}/{len(src_lines)}\n  {src_lang}: {es_s}\n  {tgt_code}: {en_s}\n")

        # Copiar tal cual líneas vacías / meta
        for j, line in enumerate(src_lines):
            if line.strip() == "":
                tgt_lines[j] = line

        out_entries.append({"times": e["times"], "text": "\n".join(tgt_lines)})

        if i % 20 == 0 or i == total:
            print(f"Bloque {i}/{total}\n", flush=True)
    return out_entries

# -------- Función principal --------
def translate_srt_to(lang_target: str, video_path, out_dir) -> str:
    """
    Traduce ./srts/sub_es.srt al idioma 'lang_target' usando MarianMT.
    - Primero ES->EN; si lang_target != 'en', luego EN->lang_target.
    - Guarda ./srts/sub_<lang_target>.srt y devuelve su ruta.
    Parámetros:
      lang_target: código destino (ej. 'en', 'de', 'fr', 'pt', 'zh-hans', ...).
    """
    import os, io, torch

    # --- Configuración fija ---
    srt_in = out_dir / (Path(video_path).stem + "_es.srt")
    srt_out = out_dir / (Path(video_path).stem + "_" + lang_target + ".srt")
    batch = 16
    print_lines = True
    keep_only = True
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(srt_in):
        raise FileNotFoundError(f"No se encontró el archivo SRT de entrada: {srt_in}")

    # 1) Cargar SRT ES
    with io.open(srt_in, "r", encoding="utf-8") as f:
        es_text = f.read()
    entries_es = parse_srt(es_text)
    print(f"Cargados {len(entries_es)} bloques desde {srt_in}")

    device = -1

    # 2) ES->EN
    print("Traduciendo ES->EN ...")
    translator_es_en = build_translator(ES_EN_MODEL, device)
    entries_en = translate_entries(entries_es, translator_es_en, batch_size=batch,
                                   print_lines=print_lines, src_lang="ES", tgt_code="EN")
    with io.open(srt_out, "w", encoding="utf-8") as f:
        f.write(write_srt(entries_en))
    print(f"Guardado Inglés: {srt_out}")

    # 3) Si el destino es EN, terminamos
    lang_target = lang_target.lower()
    if lang_target == "en":
        return srt_out

    # 4) EN->destino
    if lang_target not in EN_TO_MODEL:
        supported = ", ".join(sorted(["en"] + list(EN_TO_MODEL.keys())))
        raise ValueError(f"Idioma destino '{lang_target}' no soportado. Soportados: {supported}")

    print(f"Traduciendo EN->{lang_target.upper()} ...")
    translator_en_xx = build_translator(EN_TO_MODEL[lang_target], device)
    entries_xx = translate_entries(entries_en, translator_en_xx, batch_size=batch,
                                   print_lines=print_lines, src_lang="EN", tgt_code=lang_target.upper())
    with io.open(srt_out, "w", encoding="utf-8") as f:
        f.write(write_srt(entries_xx))
    print(f"Guardado {lang_target.upper()}: {srt_out}")

    return srt_out

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import re, csv, bisect
from collections import defaultdict

# ============================================================
# ---------------------- HELPERS merge (ms) ------------------
# ============================================================
SRT_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}),(?P<ms2>\d{3})"
)

@dataclass
class Cue:
    idx: int
    start_ms: int
    end_ms: int
    text: str

def parse_time_to_ms(t: str) -> int:
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

def ms_to_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_srt(path: str) -> List[Cue]:
    cues: List[Cue] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    idx_counter = 1
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        time_line = None
        line_offset = 0
        if re.fullmatch(r"\d+", lines[0].strip()):
            line_offset = 1
        if line_offset < len(lines):
            cand = lines[line_offset].strip()
            if "-->" in cand:
                time_line = cand
        if not time_line:
            continue
        m = SRT_TIME_RE.search(time_line)
        if not m:
            continue
        start_ms = parse_time_to_ms(f"{m['h']}:{m['m']}:{m['s']},{m['ms']}")
        end_ms = parse_time_to_ms(f"{m['h2']}:{m['m2']}:{m['s2']},{m['ms2']}")
        text_lines = lines[line_offset + 1:]
        txt = "\n".join(text_lines).strip()
        cues.append(Cue(idx=idx_counter, start_ms=start_ms, end_ms=end_ms, text=txt))
        idx_counter += 1
    return cues

def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def single_line(text: str) -> str:
    parts = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return re.sub(r"\s+", " ", " ".join(parts))

# ============================================================
# ----------------- HELPERS diarización (seg) ----------------
# ============================================================
TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
FACE_LINE_RE = re.compile(r"(?:^|\s)(FACE[_ ]\d+)\b", flags=re.IGNORECASE)
SPEAKER_PREFIX_RE = re.compile(r"^(SPEAKER[_ ]\d+)(:?\s*)", flags=re.IGNORECASE)
FACE_PREFIX_RE    = re.compile(r"^(FACE[_ ]\d+)(:?\s*)",    flags=re.IGNORECASE)

def parse_time_srt(t: str) -> float:
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(",")
    return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0

def fmt_time_srt(s: float) -> str:
    ms_total = int(round(s * 1000))
    hh, rem = divmod(ms_total, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def parse_srt_blocks(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    cues = []
    for b in blocks:
        lines = [ln.rstrip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        t_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if t_idx is None:
            continue
        m = TIME_RE.search(lines[t_idx])
        if not m:
            continue
        start = parse_time_srt(m.group(1))
        end   = parse_time_srt(m.group(2))
        text_lines = lines[t_idx+1:] if t_idx+1 < len(lines) else []
        cues.append((start, end, text_lines))
    return cues

def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def load_faces(faces_srt: Path):
    face_cues = parse_srt_blocks(faces_srt)
    faces = []
    for s, e, txt_lines in face_cues:
        if not txt_lines:
            continue
        m = FACE_LINE_RE.search(txt_lines[0].strip())
        if not m:
            continue
        label = m.group(1).upper().replace(" ", "_")
        faces.append((s, e, label))
    return sorted(faces, key=lambda x: x[0])

def assign_face_to_subs(subs, faces):
    if not faces:
        return [(s, e, lines, None) for (s, e, lines) in subs]

    faces_by_start = sorted(faces, key=lambda x: x[0])
    faces_by_end   = sorted(faces, key=lambda x: x[1])
    face_starts    = [fs for fs, _, _ in faces_by_start]
    face_ends      = [fe for _, fe, _ in faces_by_end]

    assigned = []
    for s, e, lines in subs:
        best_label, best_ov = None, 0.0
        for fs, fe, flab in faces:
            ov = overlap((s, e), (fs, fe))
            if ov > best_ov:
                best_ov, best_label = ov, flab
        if best_label is None:
            idx_prev = bisect.bisect_right(face_ends, s) - 1
            if idx_prev >= 0:
                best_label = faces_by_end[idx_prev][2]
            else:
                idx_next = bisect.bisect_left(face_starts, e)
                if idx_next < len(faces_by_start):
                    best_label = faces_by_start[idx_next][2]
        assigned.append((s, e, lines, best_label))
    return assigned

def apply_speaker_labels(assigned, label_position="prefix", mapping_csv: Path | None = None):
    face2spk, next_id = {}, 1
    out_blocks = []
    for s, e, lines, face_label in assigned:
        new_lines = list(lines) if lines else []
        if new_lines:
            first = new_lines[0].strip()
            first = SPEAKER_PREFIX_RE.sub("", first)
            first = FACE_PREFIX_RE.sub("", first)
            first = first.lstrip(": ").lstrip()
            new_lines[0] = first
        if face_label:
            if face_label not in face2spk:
                face2spk[face_label] = f"SPEAKER_{next_id}"
                next_id += 1
            spk = face2spk[face_label]
            if new_lines:
                new_lines[0] = (f"{spk}: {new_lines[0]}" if label_position == "prefix"
                                else f"{new_lines[0]} ({spk})")
            else:
                new_lines = [spk]
        out_blocks.append((s, e, new_lines))
    if mapping_csv:
        with Path(mapping_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["FACE_label", "SPEAKER_label"])
            for face, spk in face2spk.items():
                w.writerow([face, spk])
    return out_blocks

def write_srt(blocks, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, (s, e, lines) in enumerate(blocks, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time_srt(s)} --> {fmt_time_srt(e)}\n")
            for ln in lines:
                f.write(f"{ln}\n")
            f.write("\n")

# ============================================================
# ------------------- MERGE: ES + XX (dual) ------------------
# ============================================================
def merge_dual_srt(
    srt1_path: str,
    srt2_path: str,
    out_path: str,
    *,
    second_line_hex: str = "#FFFF00",
    collapse_lines: bool = True,
    min_overlap_ms: int = 500,
    min_overlap_ratio: float = 0.25,
    nearest_gap_ms: int = 1000
) -> None:
    cues1 = parse_srt(srt1_path)
    cues2 = parse_srt(srt2_path)

    out_lines = []
    for i, c1 in enumerate(cues1, 1):
        dur1 = max(1, c1.end_ms - c1.start_ms)
        threshold = max(min_overlap_ms, int(dur1 * min_overlap_ratio))

        overlapped: List[Cue] = []
        for c2 in cues2:
            ov = overlap_ms(c1.start_ms, c1.end_ms, c2.start_ms, c2.end_ms)
            if ov >= threshold:
                overlapped.append(c2)

        if not overlapped:
            nearest: Optional[Cue] = None
            best_gap = 10**9
            for c2 in cues2:
                if c2.end_ms < c1.start_ms:
                    gap = c1.start_ms - c2.end_ms
                elif c2.start_ms > c1.end_ms:
                    gap = c2.start_ms - c1.end_ms
                else:
                    gap = 0
                if gap < best_gap:
                    best_gap = gap
                    nearest = c2
            if nearest and best_gap <= nearest_gap_ms:
                overlapped = [nearest]

        line1 = single_line(c1.text) if collapse_lines else c1.text
        if overlapped:
            joined = " ".join([c2.text for c2 in overlapped]) if collapse_lines else "\n".join([c2.text for c2 in overlapped])
            line2_txt = single_line(joined) if collapse_lines else joined
        else:
            line2_txt = ""

        out_lines.append(str(i))
        out_lines.append(f"{ms_to_time(c1.start_ms)} --> {ms_to_time(c1.end_ms)}")
        out_lines.append(line1 if line1 else "")
        if line2_txt:
            out_lines.append(f'<font color="{second_line_hex}">{line2_txt}</font>')
        out_lines.append("")

    Path(out_path).write_text("\n".join(out_lines), encoding="utf-8")

# ============================================================
# ------------------- DIARIZACIÓN (sin split) ----------------
# ============================================================
def diarize_subs_path(
    subs_in: str | Path,
    faces_srt: str | Path,
    out_path: str | Path | None = None,
    *,
    label_position: str = "prefix",
    mapping_csv: str | Path | None = None,
) -> Path:
    """
    Asigna SPEAKER_n en base a talking_faces (sin dividir cues).
    """
    subs_in = Path(subs_in)
    faces_srt = Path(faces_srt)
    out_path = Path(out_path) if out_path else subs_in.with_name(subs_in.stem + "_diarizado.srt")

    subs     = parse_srt_blocks(subs_in)
    faces    = load_faces(faces_srt)
    assigned = assign_face_to_subs(subs, faces)
    blocks   = apply_speaker_labels(assigned, label_position=label_position, mapping_csv=Path(mapping_csv) if mapping_csv else None)
    write_srt(blocks, out_path)
    return out_path

# ============================================================
# --------- POST: split bilingüe guiado por puntuación -------
# ============================================================
PUNCT_CHARS = set(list(",.;:!?…"))

def _nearest_punct_cut(text: str, ratio_first: float, max_radius: int = 18):
    if not text or not text.strip():
        return None, None
    idx = int(round(len(text) * ratio_first))
    idx = max(1, min(len(text)-1, idx))
    for r in range(0, max_radius+1):
        for pos in (idx-r, idx+r):
            if 1 <= pos < len(text)-1 and text[pos] in PUNCT_CHARS:
                nxt = text[pos+1]
                if nxt == " " or nxt in ['"', '»', '”']:
                    return pos, text[pos]
    return None, None

def _capitalize_first_alpha(s: str) -> str:
    i = 0
    while i < len(s) and not s[i].isalpha():
        i += 1
    if i < len(s):
        return s[:i] + s[i].upper() + s[i+1:]
    return s

def _strip_font_tag(line: str):
    m = re.match(r'^<font\s+color="([^"]+)">(.*)</font>\s*$', line.strip(), flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(2), m.group(1)
    return line, None

def _wrap_font(text: str, color: str | None):
    return f'<font color="{color}">{text}</font>' if color else text

def _face_runs_within_gentle(cue_start, cue_end, faces, min_overlap_ratio=0.05, min_run_seconds=0.2):
    cue_dur = max(1e-6, cue_end - cue_start)
    runs = []
    for fs, fe, flab in faces:
        s = max(cue_start, fs); e = min(cue_end, fe)
        dur = e - s
        if dur <= 0:
            continue
        if dur < min_run_seconds and dur < cue_dur * min_overlap_ratio:
            continue
        runs.append((s, e, flab))
    if not runs:
        return []
    runs.sort(key=lambda x: x[0])
    merged = []
    for s,e,lab in runs:
        if not merged or lab != merged[-1][2] or s > merged[-1][1]:
            merged.append([s,e,lab])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s,e,lab) for s,e,lab in merged]

def _extract_speaker_from_line(line: str):
    m = SPEAKER_PREFIX_RE.match(line.strip())
    if m:
        return m.group(1).upper().replace(" ", "_")
    return None

def _build_face_to_speaker_map(cues, faces):
    votes = defaultdict(lambda: defaultdict(int))
    for c in cues:
        if not c:
            continue
        s, e, lines = c
        if not lines:
            continue
        spk = _extract_speaker_from_line(lines[0])
        if not spk:
            continue
        runs = _face_runs_within_gentle(s, e, faces)
        if not runs:
            continue
        by_lab = defaultdict(float)
        for rs, re_end, lab in runs:
            by_lab[lab] += (re_end - rs)
        lab, _ = max(by_lab.items(), key=lambda kv: kv[1])
        votes[lab][spk] += 1
    mapping = {}
    for face, d in votes.items():
        sp = max(d.items(), key=lambda kv: kv[1])[0]
        mapping[face] = sp
    return mapping

def bilingual_punct_split_inplace(subs_path: Path, faces_srt: Path,
                                  *, MIN_SEG_SEC: float = 0.8, PUNCT_WINDOW: int = 18) -> Path:
    """
    Reescribe el archivo `subs_path` dividiendo cues multi-orador SOLO si hay puntuación
    cerca del cambio de cara. Aplica el mismo corte a la línea en inglés.
    Si corta en coma: la convierte en punto y capitaliza el inicio del siguiente fragmento.
    """
    # Leer
    cues = parse_srt_blocks(subs_path)
    faces = load_faces(faces_srt)
    face2spk = _build_face_to_speaker_map(cues, faces)

    existing_spks = []
    for _, _, lines in cues:
        if not lines:
            existing_spks.append(None); continue
        existing_spks.append(_extract_speaker_from_line(lines[0]))

    max_id = 0
    for sp in existing_spks:
        if not sp: continue
        m = re.search(r'(\d+)$', sp)
        if m:
            max_id = max(max_id, int(m.group(1)))
    next_id = max_id + 1

    def prev_spk(i):
        for j in range(i-1, -1, -1):
            if existing_spks[j]:
                return existing_spks[j]
        return None

    def next_spk(i):
        for j in range(i+1, len(existing_spks)):
            if existing_spks[j]:
                return existing_spks[j]
        return None

    out_blocks = []
    for idx, (s, e, lines) in enumerate(cues):
        prv = prev_spk(idx)
        nxt = next_spk(idx)

        # preparar ES + EN
        primary = ""
        extras = []
        en_text = None
        en_color = None

        if lines:
            primary = SPEAKER_PREFIX_RE.sub("", lines[0].strip())
            primary = FACE_PREFIX_RE.sub("", primary).lstrip(": ").lstrip()
            if len(lines) > 1:
                en_text, en_color = _strip_font_tag(lines[1])
                extras = lines[2:] if len(lines) > 2 else []
            else:
                extras = []
        else:
            extras = []

        runs = _face_runs_within_gentle(s, e, faces)
        # agrupar contiguas del mismo FACE
        selected = []
        for rs, re_end, lab in runs:
            if not selected or lab != selected[-1][2]:
                selected.append((rs, re_end, lab))
            else:
                p = selected[-1]
                selected[-1] = (p[0], re_end, lab)

        did_split = False
        if len(selected) >= 2 and primary:
            rs1, re1, lab1 = selected[0]
            rs2, re2, lab2 = selected[1]
            if lab1 != lab2:
                total_d = e - s
                ratio_first = (re1 - s) / total_d if total_d > 0 else 0.5
                punct_idx_es, punct_char_es = _nearest_punct_cut(primary, ratio_first, max_radius=PUNCT_WINDOW)
                punct_idx_en, punct_char_en = (None, None)
                if en_text:
                    punct_idx_en, punct_char_en = _nearest_punct_cut(en_text, ratio_first, max_radius=PUNCT_WINDOW)

                if punct_idx_es is not None:
                    t_cut = s + (e - s) * (punct_idx_es / max(1, len(primary)))
                    if (t_cut - s) >= MIN_SEG_SEC and (e - t_cut) >= MIN_SEG_SEC:
                        left_es  = primary[:punct_idx_es+1].strip()
                        right_es = primary[punct_idx_es+1:].strip()
                        if punct_char_es == ",":
                            left_es  = left_es[:-1].rstrip() + "."
                            right_es = _capitalize_first_alpha(right_es)

                        if en_text:
                            if punct_idx_en is None:
                                prop = int(round(len(en_text) * (punct_idx_es / max(1, len(primary)))))
                                prop = max(1, min(len(en_text)-1, prop))
                                pi2, pc2 = _nearest_punct_cut(en_text, punct_idx_es/max(1, len(primary)), max_radius=10)
                                if pi2 is not None:
                                    punct_idx_en, punct_char_en = pi2, pc2
                                else:
                                    punct_idx_en, punct_char_en = prop-1, en_text[prop-1]
                            left_en  = en_text[:punct_idx_en+1].strip()
                            right_en = en_text[punct_idx_en+1:].strip()
                            if punct_char_en == ",":
                                left_en  = left_en[:-1].rstrip() + "."
                                right_en = _capitalize_first_alpha(right_en)

                        spk1 = face2spk.get(lab1) or prv or f"SPEAKER_{next_id}"
                        if spk1.startswith("SPEAKER_") and spk1 not in (existing_spks or []):
                            next_id += 1
                        spk2 = face2spk.get(lab2) or nxt or (spk1 if prv and not nxt else f"SPEAKER_{next_id}")
                        if spk2 == spk1 or spk2 is None:
                            spk2 = f"SPEAKER_{next_id}"; next_id += 1

                        lines1 = [f"{spk1}: {left_es}"]
                        if en_text is not None:
                            lines1.append(_wrap_font(left_en, en_color))
                        lines1 += extras

                        lines2 = [f"{spk2}: {right_es}"]
                        if en_text is not None:
                            lines2.append(_wrap_font(right_en, en_color))
                        lines2 += extras

                        out_blocks.append((s, t_cut, lines1))
                        out_blocks.append((t_cut, e, lines2))
                        did_split = True

        if not did_split:
            existing_spk = existing_spks[idx]
            if primary:
                first_line = (f"{existing_spk}: {primary}" if existing_spk else primary)
                lines_out = [first_line]
                if en_text is not None:
                    lines_out.append(_wrap_font(en_text, en_color))
                lines_out += extras
                out_blocks.append((s, e, lines_out))
            else:
                out_blocks.append((s, e, lines))

    # Reescribir en el mismo path
    write_srt(out_blocks, subs_path)
    return subs_path

# ============================================================
# ------ FUNCIÓN pública simplificada: SOLO video_path -------
# ============================================================
def merge_and_diarize_from_video(video_path: str, src_lang: str = "es") -> list[Path]:
    """
    A partir de `video_path`:
      - Busca en <project_root>/outdir:
          <base>_<src>.srt                   (p.ej. _es.srt)
          <base>_XX.srt                      (traducciones)
          <base>_talking_faces.srt
      - Para cada XX:
          1) merge_dual_srt -> <base>_<src>XX.srt
          2) diarize_subs_path -> <base>_<src>XX_diarizado.srt
          3) bilingual_punct_split_inplace -> reescribe <base>_<src>XX_diarizado.srt
    Devuelve la lista de rutas finales (_diarizado.srt tras el post-procesado).
    """
    vp = Path(video_path).resolve()

    # project_root: padre de 'data' si existe; si no, carpeta del vídeo
    project_root = None
    for parent in vp.parents:
        if parent.name.lower() == "data":
            project_root = parent.parent
            break
    if project_root is None:
        project_root = vp.parent

    outdir = project_root / "outdir"
    base = vp.stem

    es_main = outdir / f"{base}_{src_lang}.srt"
    faces   = outdir / f"{base}_talking_faces.srt"

    # candidatos XX (excepto _es, _talking_faces, _diarizado/_esXX)
    candidates = []
    for p in outdir.glob(f"{base}_*.srt"):
        low = p.name.lower()
        if low.endswith("_talking_faces.srt"):
            continue
        if f"_{src_lang}.srt" == low[-(len(src_lang)+5):]:
            continue
        if low.endswith("_diarizado.srt"):
            continue
        # evitar ya-duales tipo _esen.srt (dos códigos pegados tras _es)
        if f"_{src_lang}" in low and low.endswith(".srt") and len(low) > len(f"{base}_{src_lang}.srt"):
            continue
        candidates.append(p)

    outputs: list[Path] = []

    # Constantes internas
    SECOND_LINE_COLOR = "#FFFF00"
    COLLAPSE_LINES = True
    MIN_OVERLAP_MS = 500
    MIN_OVERLAP_RATIO = 0.25
    NEAREST_GAP_MS = 1000
    LABEL_POSITION = "prefix"

    for second_srt in sorted(candidates):
        suffix = second_srt.stem.split("_")[-1]  # xx
        dual_out = outdir / f"{base}_{src_lang}{suffix}.srt"
        diarized_out = outdir / f"{base}_{src_lang}{suffix}_diarizado.srt"

        # 1) merge dual
        merge_dual_srt(
            str(es_main), str(second_srt), str(dual_out),
            second_line_hex=SECOND_LINE_COLOR,
            collapse_lines=COLLAPSE_LINES,
            min_overlap_ms=MIN_OVERLAP_MS,
            min_overlap_ratio=MIN_OVERLAP_RATIO,
            nearest_gap_ms=NEAREST_GAP_MS
        )

        # 2) diarización (sin dividir)
        diarize_subs_path(
            subs_in=dual_out,
            faces_srt=faces,
            out_path=diarized_out,
            label_position=LABEL_POSITION,
            mapping_csv=None
        )

        # 3) split bilingüe con puntuación (REESCRIBE el _diarizado.srt)
        bilingual_punct_split_inplace(
            subs_path=diarized_out,
            faces_srt=faces,
            MIN_SEG_SEC=0.8,        # interno
            PUNCT_WINDOW=18         # interno
        )

        outputs.append(diarized_out)

    return outputs

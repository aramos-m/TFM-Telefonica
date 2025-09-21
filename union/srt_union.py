from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import re, csv, bisect

# ============================================================
# ---------------------- HELPERS SRT (ms) --------------------
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
# ----------- FUNCIONES de merge y diarización ---------------
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

def diarize_subs_path(
    subs_in: str | Path,
    faces_srt: str | Path,
    out_path: str | Path | None = None,
    *,
    label_position: str = "prefix",
    mapping_csv: str | Path | None = None,
) -> Path:
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
# -------- FUNCIÓN simplificada: SOLO video_path -------------
# ============================================================
def merge_and_diarize_from_video(video_path: str, src_lang: str = "es") -> list[Path]:
    """
    A partir de la ruta del vídeo, asume esta convención en la misma raíz del proyecto:

      outdir/
        <basename>_<src_lang>.srt                (SRT base, p.ej. _es.srt)
        <basename>_XX.srt                        (uno o varios segundos SRT; XX = en, fr, ... )
        <basename>_talking_faces.srt             (para diarización)
        -> genera:
           <basename>_<src_lang>XX.srt
           <basename>_<src_lang>XX_diarizado.srt

    No hace comprobaciones de existencia.
    Devuelve la lista de SRT diarizados generados.
    """
    vp = Path(video_path).resolve()
    # project_root: padre de 'data' si existe en la ruta; si no, carpeta del vídeo
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

    # Detectar todos los *_XX.srt (excepto es, talking_faces, diarizado, y ya-duales)
    candidates = []
    for p in outdir.glob(f"{base}_*.srt"):
        n = p.name
        low = n.lower()
        if low.endswith("_talking_faces.srt"):
            continue
        if low.endswith("_diarizado.srt"):
            continue
        if f"_{src_lang}." in low:  # exacto: _es.srt
            continue
        if f"_{src_lang}" in low and low.endswith(".srt") and len(low) > len(f"{base}_{src_lang}.srt"):
            # evita ya-duales tipo _esen.srt, _esfr.srt, etc.
            continue
        candidates.append(p)

    diarized_outputs: list[Path] = []

    # Constantes internas (no expuestas como argumentos)
    SECOND_LINE_COLOR = "#FFFF00"
    COLLAPSE_LINES = True
    MIN_OVERLAP_MS = 500
    MIN_OVERLAP_RATIO = 0.25
    NEAREST_GAP_MS = 1000
    LABEL_POSITION = "prefix"

    for second_srt in sorted(candidates):
        # sufijo XX
        suffix = second_srt.stem.split("_")[-1]  # basename_xx -> xx
        dual_out = outdir / f"{base}_{src_lang}{suffix}.srt"
        diarized_out = outdir / f"{base}_{src_lang}{suffix}_diarizado.srt"

        # 1) Merge dual es+xx -> _esxx.srt
        merge_dual_srt(
            str(es_main), str(second_srt), str(dual_out),
            second_line_hex=SECOND_LINE_COLOR,
            collapse_lines=COLLAPSE_LINES,
            min_overlap_ms=MIN_OVERLAP_MS,
            min_overlap_ratio=MIN_OVERLAP_RATIO,
            nearest_gap_ms=NEAREST_GAP_MS
        )

        # 2) Diarización con talking_faces -> _esxx_diarizado.srt
        out_path = diarize_subs_path(
            subs_in=dual_out,
            faces_srt=faces,
            out_path=diarized_out,
            label_position=LABEL_POSITION,
            mapping_csv=None
        )
        diarized_outputs.append(out_path)

    return diarized_outputs

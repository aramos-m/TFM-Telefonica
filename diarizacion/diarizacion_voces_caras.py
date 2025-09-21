# === TODO-EN-UNO: genera faces_detections.csv si falta + Talking-Face SRT/CSV ===

# -------------------------
# Imports
# -------------------------
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm
from insightface.app.face_analysis import FaceAnalysis
from sklearn.cluster import DBSCAN
import mediapipe as mp
import librosa

# ==========================
# Helpers (arriba del archivo)
# ==========================

def formato_srt(segundos):
    """
    Convierte segundos en formato SRT (HH:MM:SS,mmm)
    """
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    seg = int(segundos % 60)
    milisegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02}:{minutos:02}:{seg:02},{milisegundos:03}"

def build_segments(times_bool, gap=0.6):
    """Une marcas (t, bool) en segmentos [inicio, fin] fusionando huecos <= gap."""
    if not times_bool:
        return []
    times_bool.sort()
    active = [t for t, v in times_bool if v]
    if not active:
        return []
    segs, s, prev = [], active[0], active[0]
    for t in active[1:]:
        if t - prev > gap:
            segs.append((s, prev))
            s = t
        prev = t
    segs.append((s, prev))
    return segs


def is_voiced(t, talk_gate):
    """True si la energía de audio en el instante t supera el umbral."""
    if talk_gate is None:
        return True
    t_env, env, thr = talk_gate
    if len(t_env) < 2:
        return True
    dt = (t_env[1] - t_env[0]) or 1e-6
    idx = int(np.clip(round(t / dt), 0, len(env) - 1))
    return env[idx] >= thr


def mouth_open_ratio(img_bgr, box, mesh, UP_C=13, LO_C=14, LEFT=61, RIGHT=291):
    """Calcula (altura de boca / anchura de boca) en el recorte de la cara."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(np.clip(v, 0, max(1, lim - 1))) for v, lim in zip(box, [w, h, w, h])]
    if x2 <= x1 or y2 <= y1:
        return None
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark

    def pt(i):
        return np.array([lm[i].x * (x2 - x1), lm[i].y * (y2 - y1)], dtype=np.float32)

    up, lo, L, R = pt(UP_C), pt(LO_C), pt(LEFT), pt(RIGHT)
    mouth_h = np.linalg.norm(up - lo)
    mouth_w = np.linalg.norm(L - R) + 1e-6
    return float(mouth_h / mouth_w)


def audio_energy_envelope(video_path, sr=16000, hop_s=0.05):
    """
    Devuelve (t_env, env_rms). Puede fallar con .mp4 si el backend no soporta el contenedor;
    en ese caso, la función llamante debería capturar la excepción y seguir sin gate.
    """
    y, sr = librosa.load(video_path, sr=sr, mono=True)
    hop = max(1, int(sr * hop_s))
    frame_rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop, center=True).flatten()
    t = np.arange(len(frame_rms)) * (hop / sr)
    return t, frame_rms

# ==========================
# Función 1: detecciones de caras (InsightFace + DBSCAN)
# ==========================
def generate_face_detections(video_path):
    SAMPLE_FPS_DET = 5
    DBSCAN_EPS = 0.45
    MIN_SAMPLES = 3
    DET_SIZE = (640, 640)

    out_dir = Path(video_path).parent.parent / "outdir"
    detections_csv = out_dir / (Path(video_path).stem + "_faces_detections.csv")
    print(detections_csv)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / SAMPLE_FPS_DET)))

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=DET_SIZE)  # CPU

    embeds, times, bboxes = [], [], []
    f = 0
    ok, frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    with tqdm(total=total, desc="Detecciones+Embeddings") as pbar:
        while ok:
            if f % step == 0:
                t = f / fps
                faces = app.get(frame) or []
                for fa in faces:
                    emb = getattr(fa, "normed_embedding", None)
                    bbox = getattr(fa, "bbox", None)
                    if emb is None or bbox is None:
                        continue
                    embeds.append(emb)
                    times.append(t)
                    bboxes.append(np.asarray(bbox, dtype=int).tolist())
            ok, frame = cap.read()
            f += 1
            pbar.update(1)
    cap.release()

    if not embeds:
        raise SystemExit("No se detectaron caras. Revisa el vídeo o ajusta parámetros.")

    X = np.vstack(embeds)
    cl = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES, metric="cosine").fit(X)
    labels = cl.labels_

    det_df = pd.DataFrame({
        "time_s": times,
        "face_id": labels,
        "x1": [bb[0] for bb in bboxes],
        "y1": [bb[1] for bb in bboxes],
        "x2": [bb[2] for bb in bboxes],
        "y2": [bb[3] for bb in bboxes],
    })
    det_df.to_csv(detections_csv, index=False)
    print("Detecciones guardadas en:", detections_csv.resolve())
    detect_talking_faces(video_path, detections_csv, out_dir)

# ==========================
# Función 2: talking-face (labios + audio gate opcional) -> CSV + SRT
# ==========================
def detect_talking_faces(video_path, detections_csv, out_dir):
    MOUTH_THRESH_DELTA = 0.02
    GAP_MERGE_S = 0.5
    USE_AUDIO_GATE = True
    AUDIO_GATE_PCTL = 60

    csv_talking   = out_dir / f"{Path(video_path).resolve()}_talking_faces.csv"
    srt_out       = out_dir / f"{Path(video_path).resolve()}_talking_faces.srt"

    # --- cargar detecciones ---
    df = pd.read_csv(detections_csv)
    print("CARGANDO DETECCIONES DE CARAS")

    if "face_id" not in df.columns and "face_id_or_-1" in df.columns:
        df = df.rename(columns={"face_id_or_-1": "face_id"})
    if "time_s" not in df.columns:
        if "frame_idx" in df.columns:
            cap_tmp = cv2.VideoCapture(video_path)
            fps_tmp = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
            cap_tmp.release()
            df["time_s"] = df["frame_idx"].astype(int) / float(fps_tmp)
        else:
            raise SystemExit("El CSV debe tener 'time_s' o 'frame_idx'.")

    df = df[(df["face_id"] >= 0)].copy()
    for c in ["x1", "y1", "x2", "y2"]:
        if c not in df.columns:
            raise SystemExit("Faltan columnas de bbox x1,y1,x2,y2 en detecciones.")
    df["time_s"] = df["time_s"].astype(float)
    df["face_id"] = df["face_id"].astype(int)

    # acceso aleatorio al vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    df["frame_idx"] = (df["time_s"] * fps).round().astype(int)
    groups = df.groupby("frame_idx")

    # FaceMesh (creado aquí y pasado al helper)
    mp_face = mp.solutions.face_mesh
    mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # calcular ratio de boca por detección
    rat_s = pd.Series(index=df.index, dtype="float32")
    hit_s = pd.Series(False, index=df.index)
    for frame_idx, g in tqdm(groups, desc="FaceMesh labios"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        for ridx in g.index:
            r = df.loc[ridx]
            rati = mouth_open_ratio(frame, (r.x1, r.y1, r.x2, r.y2), mesh)
            if rati is not None:
                rat_s.at[ridx] = rati
                hit_s.at[ridx] = True

    df["mouth_ratio"] = rat_s.values
    df = df[hit_s.values].copy()
    if df.empty:
        raise SystemExit("No se pudieron estimar landmarks de labios en ninguna detección.")

    # umbral adaptativo por cara
    thr_by_face = (df.groupby("face_id")["mouth_ratio"].median()).to_dict()
    df["thr_face"] = df["face_id"].map(thr_by_face)
    df["is_talking_mouth"] = df["mouth_ratio"] > (df["thr_face"] + MOUTH_THRESH_DELTA)

    # gate de audio (opcional)
    talk_gate = None
    if USE_AUDIO_GATE:
        try:
            t_env, env = audio_energy_envelope(video_path, sr=16000, hop_s=0.05)
            thr = np.percentile(env, AUDIO_GATE_PCTL)
            talk_gate = (t_env, env, float(thr))
        except Exception as e:
            print("Aviso: no se pudo calcular energía de audio, continúo sin gate.", e)

    # combinar boca + audio
    df["is_talking"] = df.apply(lambda r: bool(r.is_talking_mouth and is_voiced(r.time_s, talk_gate)), axis=1)

    # construir segmentos por cara
    segments = []
    for fid, g in df.groupby("face_id"):
        tb = list(zip(g["time_s"].tolist(), g["is_talking"].tolist()))
        for s, e in build_segments(tb, gap=GAP_MERGE_S):
            if e <= s:
                e = s + 0.2
            segments.append((int(fid), float(s), float(e)))

    # guardar resultados
    segments.sort(key=lambda x: (x[1], x[2], x[0]))

    seg_df = pd.DataFrame(segments, columns=["face_id", "start_s", "end_s"])
    seg_df.to_csv(csv_talking, index=False)
    print(f"CSV 'talking faces' escrito: {csv_talking.resolve()} | Filas: {len(seg_df)}")

    with open(srt_out, "w", encoding="utf-8") as f:
        for i, (fid, s, e) in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{formato_srt(s)} --> {formato_srt(e)}\n")
            f.write(f"FACE_{fid:02d}\n\n") 

    print(f"SRT 'talking faces' escrito: {srt_out.resolve()} | Cues: {len(segments)}")

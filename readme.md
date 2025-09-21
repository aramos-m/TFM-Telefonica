# TFM – Subtitulación, Traducción y Diarización de Vídeos  

Este proyecto procesa vídeos en formato `.mp4` para:  
- Extraer y transcribir el audio con **Whisper**.  
- Detectar caras y segmentos de habla sincronizados.  
- Traduce automáticamente los subtítulos al inglés (y otros idiomas soportados).  
- Generar subtítulos duales (español + traducción) con **diarización por hablante**.  

Los subtítulos finales se guardan en la carpeta `outdir`.  

---

## 📦 Instalación

Clona el repositorio y asegúrate de tener **Python 3.9+**:  

```bash
git clone https://github.com/aramos-m/TFM-Telefonica.git
cd TFM-Telefonica
pip3 install -r requirements.txt
```

---

## 📂 Preparación de datos

Coloca en la carpeta `data` los vídeos en formato **`.mp4`** que quieras procesar:  

```
├── data/
│   ├── ejemplo1.mp4
│   └── ejemplo2.mp4
```

---

## ▶️ Ejecución

Lanza el script principal desde una terminal:  

```bash
python3 main.py
```

El pipeline automático hará lo siguiente para cada vídeo en `data/`:  
1. Convertir `.mp4` → `.wav`.  
2. Transcribir el audio a subtítulos en español (`*_es.srt`).  
3. Detectar caras y segmentos de habla (`*_faces_detections.csv`, `*_talking_faces.srt`).  
4. Traducir los subtítulos al inglés (`*_en.srt`).  
5. Fusionar subtítulos español+inglés y asignar hablantes (`*_esen_diarizado.srt`).  

---

## 📁 Resultados

Los archivos generados se guardan en `outdir/`, por ejemplo:  

```
outdir/
├── ejemplo1_es.srt              # subtítulos en español
├── ejemplo1_en.srt              # traducción al inglés
├── ejemplo1_esen.srt            # subtítulos duales (ES+EN)
├── ejemplo1_esen_diarizado.srt  # subtítulos duales con hablantes
├── ejemplo1_faces_detections.csv
├── ejemplo1_talking_faces.srt
```

---

## 🌍 Idiomas soportados

Actualmente se soporta traducción automática desde español a:  
- **en** (inglés, por defecto)  
- de, fr, it, pt, nl, sv, da, cs, pl, uk, el, he, tr, ro, zh-hans, ar, ru, ja, ko  

> Para cambiar el idioma destino, modifica el parámetro `lang_target` en la función [`translate_srt_to`](traduccion/traduccion.py).  

---



## ⚙️ Nota sobre recursos y rendimiento

Este proyecto está optimizado para equipos con VRAM limitada, permitiendo ejecutar el modelo Whisper-large usando CPU en lugar de GPU.

- La ejecución en CPU implica una velocidad de procesamiento menor, pero garantiza compatibilidad en ordenadores sin GPU potente.

- El rendimiento depende fuertemente de la memoria RAM disponible: se recomienda un mínimo de 8 GB.

- En equipos con solo 8 GB de RAM, es aconsejable cerrar otros procesos pesados durante la ejecución para evitar errores por falta de memoria.

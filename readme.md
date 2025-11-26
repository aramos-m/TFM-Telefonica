<table width="100%">
<tr>
<td align="left" valign="middle">
  <a href="https://www.ucm.es/">
    <img src=".images/logo-ucm.png" alt="Universidad Complutense de Madrid" height="80"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.unir.net/">
    <img src="https://www.unir.net/wp-content/uploads/2019/11/Unir_2021_logo.jpg" alt="UNIR" height="80"/>
  </a>
</td>
<td align="right" valign="middle">
  <a href="https://www.telefonica.com/">
    <img src=".images/logo-telefonica.png" alt="Telefónica" height="80"/>
  </a>
</td>
</tr>
</table>

# Sistema Integral de Accesibilidad Audiovisual basado en IA

Este repositorio contiene el código fuente del proyecto desarrollado como **Trabajo de Fin de Máster (TFM)** conjunto, en el marco del **Programa Tutoría de Telefónica**.

El proyecto es el resultado de la colaboración académica entre estudiantes del **Máster en Letras Digitales** de la Universidad Complutense de Madrid (UCM) y el **Máster en Inteligencia Artificial** de la Universidad Internacional de La Rioja (UNIR).

---

## 📄 Descripción del Proyecto

Este sistema implementa un pipeline automatizado diseñado para mejorar la accesibilidad de contenidos audiovisuales mediante el uso de Inteligencia Artificial Generativa y Visión Artificial. El software procesa vídeos (`.mp4`) para generar subtítulos enriquecidos que incluyen:

1.  **Transcripción automática** del habla (Speech-to-Text).
2.  **Diarización visual de hablantes**: Identificación de *quién* habla basándose en reconocimiento facial y sincronización labial (*lip-sync*), no solo en la voz.
3.  **Traducción automática neuronal** a múltiples idiomas.
4.  **Fusión multimodal**: Generación de archivos de subtítulos duales y etiquetados por hablante.

---

## 🛠️ Arquitectura Técnica

El sistema integra múltiples modelos de Inteligencia Artificial para procesar archivos `.mp4` de forma automática:

* **Transcripción de Alta Precisión:** Utiliza **OpenAI Whisper (Large)** para convertir audio a texto en español con marcas de tiempo precisas.
* **Diarización Visual (Talking Face Detection):**
    * Detección y *embedding* de rostros mediante **InsightFace**.
    * Clustering de identidades con **DBSCAN**.
    * Sincronización labial (*Active Speaker Detection*) usando **MediaPipe** para asociar el audio al rostro correcto en pantalla.
* [cite_start]**Traducción Neuronal en Cascada:** Implementa modelos **MarianMT** (Helsinki-NLP) para traducir subtítulos del español al inglés y, posteriormente, a una variedad de idiomas destino.
* **Fusión Multimodal:** Genera archivos `.srt` duales (idioma original + traducción) con identificación de hablantes (`SPEAKER_X`) basada en la detección visual.

---

## 📦 Requisitos e Instalación

Este proyecto ha sido optimizado para ejecutarse en entornos con recursos limitados (**CPU**), democratizando el acceso a herramientas de accesibilidad sin necesidad de GPUs dedicadas de alto rendimiento.

### Prerrequisitos
*   **Python 3.9+**
*   **FFmpeg** instalado en el sistema.
*   Se recomienda un mínimo de **8 GB de RAM** (16 GB recomendados para el modelo Large de Whisper).

### Instalación
Clona el repositorio e instala las dependencias necesarias:

```bash
git clone [https://github.com/aramos-m/TFM-Telefonica.git](https://github.com/aramos-m/TFM-Telefonica.git)
cd TFM-Telefonica

# Se recomienda crear un entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalación de librerías (Torch CPU por defecto)
pip3 install -r requirements.txt
```
---

## ⏯️ Ejecución

1.  **Preparación de datos:**
    Coloca los vídeos `.mp4` que deseas procesar en la carpeta `data/`.
    ```text
    ├── data/
    │   ├── ejemplo1.mp4
    │   └── ejemplo2.mp4
    ```

2.  **Ejecutar el pipeline:**
    ```bash
    python main.py
    ```

El pipeline automático hará lo siguiente para cada vídeo en `data/`:  
1. Convertir `.mp4` → `.wav`.  
2. Transcribir el audio a subtítulos en español (`*_es.srt`).  
3. Detectar caras y segmentos de habla (`*_faces_detections.csv`, `*_talking_faces.srt`).  
4. Traducir los subtítulos al inglés (`*_en.srt`).  
5. Fusionar subtítulos español+inglés y asignar hablantes (`*_esen_diarizado.srt`).  

---

## 📂 Resultados

Los archivos resultantes se guardan en el directorio `outdir/`. Para cada vídeo procesado, obtendrás:

| Archivo | Contenido |
| :--- | :--- |
| `*_es.srt` | Transcripción original en español. |
| `*_en.srt` | Traducción al inglés (o idioma seleccionado). |
| `*_talking_faces.srt` | Segmentos de tiempo con caras hablando detectadas. |
| `*_faces_detections.csv` | Datos técnicos de detección facial y embeddings. |
| **`*_esen_diarizado.srt`** | **Archivo Final:** Subtítulos bilingües con identificación de hablante (e.g., `SPEAKER_1`). |

---

## 🌍 Idiomas Soportados

El módulo de traducción utiliza **MarianMT** y soporta la traducción desde Español a una amplia variedad de idiomas, incluyendo:

*   **Inglés (en)** - *Por defecto*
*   Francés (fr), Alemán (de), Italiano (it), Portugués (pt)
*   Chino (zh-hans), Japonés (ja), Árabe (ar), Ruso (ru)
*   Y otros idiomas europeos (nl, sv, da, cs, pl, uk, el, tr, ro).

Para cambiar el idioma destino, modifica el primer parámetro en la función [`translate_srt_to`](main.py).

---

© 2024 - TFM Máster Letras Digitales (UCM) & Máster IA (UNIR) - Programa Tutoría Telefónica.

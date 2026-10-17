"""Catálogo de voces SUPERTONIC 3 (ONNX, 100 % local, CPU) para Jarvis.

Misma interfaz que `tts_piper.py` y `tts_eleven.py` (`catalogo` / `usar_voz` / `clave_de` /
`disponible` / `sintetizar` / `normalizar`) → `jarvis-tablet.py` no distingue motores y los
endpoints `/voces` y `/voz` siguen exactamente igual.

POR QUÉ EXISTE (10-oct-2026). Veredicto de Tavo sobre el catálogo Piper: *"supertonic son
buenísimas, las piper son basura"*. VITS (Piper) tiene techo: en español las voces femeninas de
calidad son un puñado de corpus viejos, y las `low` de MLS **se cortan** (inestables: la misma
frase les sale 15,7 / 19,0 / 9,7 s). Supertonic 3 es otra arquitectura: **44 100 Hz** (el doble
que Piper), 31 idiomas, ~99M de parámetros y corre en **CPU** — no compite por la VRAM, que es
donde vive el Whisper del dictado. Medido acá: **1,1-1,4 s** por frase corta y **sin cortes** con
737 caracteres (49-52 s de audio en 9-11 s de cómputo).

⚠️ SOLO VOCES FEMENINAS (regla de Tavo, 9-oct-2026). El modelo trae estilos F1-F5 y M1-M5; acá se
exponen **sólo las F**. El género no se asumió: se midió la F0 mediana por autocorrelación
(todas >165 Hz, ver `descripcion` de cada una).

⚠️ LA GANANCIA ES PROPIA DE ESTE MOTOR. Supertonic entrega ~**-25 dB** de media con picos de
-5 a -10 dB: unos **9 dB más bajo** que Piper. Se compensa con `JARVIS_VOLUMEN_DB_SUPERTONIC`
(medir con `ffmpeg -af volumedetect`; el limitador tapa los picos).

⚠️ LA LATENCIA SE PAGA POR ORACIÓN, no por turno: `jarvis-tablet.py` sintetiza oración por oración
mientras el LLM sigue escribiendo, así que el primer sonido sale en ~1-2 s aunque la respuesta
completa tarde más. Con textos largos (700+ caracteres) la síntesis entera tarda 9-11 s: si algún
día molesta, la palanca es `max_chunk_length`, no bajar la calidad.

## DE DÓNDE SALEN MÁS VOCES (ver también `references/motor-supertonic.md`)

Un estilo es un **JSON de ~290 KB con tensores float** (`style_ttl` [1,50,256] + `style_dp`
[1,8,16]); cualquiera que aparezca en `CUSTOM_DIRS` se suma solo al catálogo. Tres fuentes:

  1. **Mezclas de presets** — gratis, local, sin GPU: `tools/mezclar-estilos.py A.json B.json 0.7 salida.json`
     interpola los tensores y produce una voz nueva legítima (no clona a ninguna persona).
  2. **Comunidad** — `sam749/supertonic3-custom-styles` (estilos `F6-zh`, `F7-gb`, `F8-gb`).
  3. **Entrenar uno propio desde un audio de referencia** — `saurabhv749/supertonic3-voice-clone`
     (MIT, ~2,6 GB de VRAM → entra en la 3050; 6-30 min por voz). Es la vía para una chilena.

  ⛔ El **Voice Builder oficial está MUERTO**: su repo se archivó y el servicio dejó de estar
  accesible el **31-ago-2026** (aviso oficial del 23-jul-2026). Lo que ya está descargado sigue
  andando offline para siempre; lo que no, no se puede generar.
"""

import io
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import wave

import numpy as np

MODELO_DIR = pathlib.Path.home() / ".cache" / "supertonic3"
ONNX_ESPERADOS = ("onnx/duration_predictor.onnx", "onnx/text_encoder.onnx",
                  "onnx/vector_estimator.onnx", "onnx/vocoder.onnx")

# Directorios donde se buscan ESTILOS SUELTOS (voces nuevas sin tocar código). Un estilo es un
# JSON de ~290 KB con tensores float (`style_ttl` + `style_dp`): cualquiera que aparezca acá se
# suma solo al catálogo de la tablet (se rescannea al pedir /voces).
CUSTOM_DIRS = (
    MODELO_DIR / "custom_styles",                                   # el que usa el SDK
    pathlib.Path(__file__).resolve().parent / "voces-custom",      # el del proyecto (versionable)
)

# ── Parámetros de síntesis ────────────────────────────────────────────────────
# `total_steps` 5 (bajo) .. 12 (alto), default del modelo 8. `speed` 0.7 .. 2.0.
# NO son el `LENGTH_SCALE` de Piper (1.50): acá 1.05 ya sale pausado (el modelo lo trae).
PASOS = int(os.environ.get("JARVIS_SUPERTONIC_PASOS", "8"))
VELOCIDAD = float(os.environ.get("JARVIS_SUPERTONIC_VELOCIDAD", "1.05"))
IDIOMA = os.environ.get("JARVIS_SUPERTONIC_IDIOMA", "es")

# Ganancia propia (ver el encabezado). Medida: con +8,5 dB la media cae en ~-15,6 dB, la misma
# sonoridad que daniela en Piper, con el limitador en 0,95.
GANANCIA_DB = float(os.environ.get("JARVIS_VOLUMEN_DB_SUPERTONIC", "8.5"))

# ── Catálogo ─────────────────────────────────────────────────────────────────
# `estilo` es el nombre del estilo en el modelo (F1..F5). La F0 la midió `tools/probar-onnx*`
# sobre la frase de comparación estándar del proyecto.
VOCES = {
    "f1": {
        "estilo": "F1",
        "nombre": "F1",
        "acento": "Supertonic · 44 kHz",
        "pais": "xx",
        "emoji": "🎙️",
        "descripcion": "Supertonic 3 estilo F1 — F0 191 Hz. Local, 44 100 Hz, corre en CPU",
    },
    "f2": {
        "estilo": "F2",
        "nombre": "F2",
        "acento": "Supertonic · 44 kHz",
        "pais": "xx",
        "emoji": "🎙️",
        "descripcion": "Supertonic 3 estilo F2 — F0 225 Hz (la más aguda). Local, 44 100 Hz",
    },
    "f3": {
        "estilo": "F3",
        "nombre": "F3",
        "acento": "Supertonic · 44 kHz",
        "pais": "xx",
        "emoji": "🎙️",
        "descripcion": "Supertonic 3 estilo F3 — F0 174 Hz. Local, 44 100 Hz. La voz por defecto",
    },
    "f4": {
        "estilo": "F4",
        "nombre": "F4",
        "acento": "Supertonic · 44 kHz",
        "pais": "xx",
        "emoji": "🎙️",
        "descripcion": "Supertonic 3 estilo F4 — F0 194 Hz. Local, 44 100 Hz",
    },
    "f5": {
        "estilo": "F5",
        # La "original" para Tavo (10-oct-2026). Convención suya: las voces GRAVES llevan
        # nombre con G (F8 → "Greta"); ésta es Almudena.
        "nombre": "Almudena",
        "acento": "Supertonic · 44 kHz",
        "pais": "xx",
        "emoji": "🎙️",
        "descripcion": "Supertonic 3 estilo F5 — F0 167 Hz (la más grave). Local, 44 100 Hz",
    },
}

_METAS = {}                        # ruta -> (mtime, metadata) del estilo, para no releer 290 KB


def _meta_de(f: pathlib.Path) -> dict:
    """Metadata de un estilo suelto, cacheada por ruta+mtime. `{}` si no se puede leer.

    Es la fuente de la PROCEDENCIA de la voz (de qué modelo viene): además de `nombre`, acepta
    `acento`, `emoji`, `pais` y `descripcion`. Así una voz nueva se describe sola desde su JSON,
    sin tocar código — que es como se diferencian hoy los clones de los estilos de la comunidad.
    """
    try:
        mtime = f.stat().st_mtime
        cache = _METAS.get(str(f))
        if cache and cache[0] == mtime:
            return cache[1]
        md = json.loads(f.read_text()).get("metadata") or {}
        if not isinstance(md, dict):
            md = {}
        _METAS[str(f)] = (mtime, md)
        return md
    except Exception:
        return {}


def _nombre_de(f: pathlib.Path) -> str:
    """Nombre visible: `metadata.nombre` si está; si no, el nombre del archivo."""
    return str(_meta_de(f).get("nombre") or "").strip() or f.stem


def _custom() -> dict:
    """Estilos JSON encontrados en `CUSTOM_DIRS` → {clave: {nombre, ruta, ...}}.

    Se rescannea en cada llamada (un `glob`, barato y NO recursivo: un subdirectorio se ignora
    solo), así una voz nueva aparece en el catálogo sin reiniciar el servicio. La clave sale del
    nombre del archivo, en minúsculas; el nombre visible puede venir del metadata.
    """
    out = {}
    for d in CUSTOM_DIRS:
        try:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.json")):
                clave = re.sub(r"[^a-z0-9]+", "-", f.stem.lower()).strip("-")
                if not clave or clave in VOCES or clave in out:
                    continue
                md = _meta_de(f)
                out[clave] = {
                    "nombre": _nombre_de(f), "ruta": str(f),
                    # La procedencia sale del metadata del estilo (ver `_meta_de`): es lo que
                    # distingue un clon de un estilo de la comunidad en el botón de la tablet.
                    "acento": str(md.get("acento") or "estilo propio · 44 kHz").strip(),
                    "pais": str(md.get("pais") or "xx").strip(),
                    "emoji": str(md.get("emoji") or "🧬").strip(),
                    "descripcion": str(md.get("descripcion")
                                       or f"Estilo propio cargado de {f.parent.name}/{f.name}").strip(),
                }
        except Exception:
            continue
    return out


# Voces que se muestran en el botón. Vacío = todas. Tavo eligió 2 (Almudena y Greta) y el resto
# no ensucia el picker; para recuperar una, agregarla acá o vaciar la variable.
VISIBLES = {c.strip().lower() for c in os.environ.get("JARVIS_VOCES_SUPERTONIC", "").split(",")
            if c.strip()}


def _todas() -> dict:
    """Catálogo efectivo: built-in + estilos sueltos, filtrado por `VISIBLES`. Es lo que ve el botón."""
    todas = {**VOCES, **_custom()}
    if VISIBLES:
        vis = {k: v for k, v in todas.items() if k in VISIBLES}
        if vis:
            return vis
    return todas


# Voz activa por defecto: la elige Tavo en `JARVIS_VOZ_SUPERTONIC` (hoy "clara", la entrenada local).
# Se valida contra `_todas()` y NO contra `VOCES`: las built-in son sólo F1-F5, así que un estilo suelto
# (clara, greta...) quedaba fuera y caía a f5 en silencio — bug encontrado al fijar Clara como default.
_DEFAULT = os.environ.get("JARVIS_VOZ_SUPERTONIC", "clara")
VOZ_ACTIVA = _DEFAULT if _DEFAULT in _todas() else next(iter(_todas()), "f5")

_tts = None                         # motor Supertonic cacheado (carga perezosa, ~1-12 s)
_estilos = {}                       # clave -> objeto Style del modelo
_LOCK = threading.Lock()           # una síntesis a la vez (el server es multi-hilo)


# ── API del catálogo (misma forma que tts_piper / tts_eleven) ─────────────────
def clave_de(clave_o_id: str) -> str:
    """Clave del catálogo. Acepta la clave (`f3`), el estilo (`F3`) o una variante con prefijo."""
    k = str(clave_o_id or "").strip().lower()
    todas = _todas()
    if k in todas:
        return k
    k2 = re.sub(r"^(st|supertonic)[-_]?", "", k)
    if k2 in todas:
        return k2
    for c, v in todas.items():
        if v.get("estilo", "").lower() == k or v["nombre"].lower() == k:
            return c
    return k


def catalogo() -> dict:
    """Las voces elegibles + cuál está activa. Lo consume el botón de la tablet."""
    todas = _todas()
    return {
        "voces": [
            {"clave": k, "activa": k == VOZ_ACTIVA,
             "nombre": v["nombre"], "acento": v["acento"], "pais": v["pais"],
             "emoji": v["emoji"], "descripcion": v["descripcion"]}
            for k, v in todas.items()
        ],
        "activa": VOZ_ACTIVA if VOZ_ACTIVA in todas else next(iter(todas), None),
        "motor": "supertonic",
    }


def usar_voz(clave_o_id: str) -> dict:
    """Cambia la voz activa EN CALIENTE. No hay caché de modelo que invalidar: Supertonic elige
    el estilo por llamada, así que alcanza con mover `VOZ_ACTIVA` (y los fijos se regeneran solos
    desde `jarvis-tablet.py`)."""
    global VOZ_ACTIVA
    clave = clave_de(clave_o_id)
    if clave not in _todas():
        raise ValueError(f"voz Supertonic desconocida: {clave_o_id!r}")
    VOZ_ACTIVA = clave
    return catalogo()


def invalidar_cache():
    """Paridad de interfaz con Piper: acá no hay nada que invalidar (los estilos se resuelven por
    llamada). Existe para que quien cambia de voz no tenga que saber qué motor corre."""
    _estilos.clear()


def disponible() -> bool:
    """Local y sin red: disponible si los .onnx del modelo están en el cache."""
    try:
        return all((MODELO_DIR / f).exists() for f in ONNX_ESPERADOS)
    except Exception:
        return False


def faltantes() -> list:
    """Archivos del modelo que faltan (para diagnóstico)."""
    return [f for f in ONNX_ESPERADOS if not (MODELO_DIR / f).exists()]


# ── Síntesis ─────────────────────────────────────────────────────────────────
def _asegurar():
    """Carga (y cachea) el motor Supertonic. La primera vez descarga ~386 MB a
    `~/.cache/supertonic3` si no están los modelos (requiere red una sola vez)."""
    global _tts
    if _tts is None:
        import time
        t0 = time.time()
        from supertonic import TTS
        _tts = TTS(auto_download=True)
        print(f"[supertonic] motor cargado en {time.time()-t0:.1f}s "
              f"({_tts.model_name}, {_tts.sample_rate} Hz)", flush=True)
    return _tts


def _estilo(clave=None):
    """Objeto de estilo de la voz pedida (la activa si no se pasa clave).

    Built-in → `get_voice_style`; estilo suelto → `get_voice_style_from_path`. Se cachea por ruta
    y mtime: reemplazar un JSON no deja el estilo viejo pegado en memoria.
    """
    k = VOZ_ACTIVA if clave is None else clave_de(clave)
    v = _todas().get(k)
    if v is None:
        raise ValueError(f"voz Supertonic desconocida: {k!r}")
    if "ruta" not in v:                       # built-in (F1..F5)
        if k not in _estilos:
            _estilos[k] = _asegurar().get_voice_style(voice_name=v["estilo"])
        return _estilos[k]
    ruta = pathlib.Path(v["ruta"])
    marca = f"{ruta}:{ruta.stat().st_mtime_ns}"
    if marca not in _estilos:
        _estilos[marca] = _asegurar().get_voice_style_from_path(str(ruta))
    return _estilos[marca]


def preprocesar(texto: str) -> str:
    """Limpieza previa a la síntesis: la MISMA que el carril local de `jennas-voz.py` (saca
    `**negrita**`, backticks, viñetas y guiones largos). Supertonic, igual que Piper, lee lo que
    le des: sin esto pronunciaría los asteriscos."""
    try:
        jv = sys.modules.get("jennas_voz")
        if jv is None:
            import importlib.util
            ruta = pathlib.Path(__file__).resolve().parent / "jennas-voz.py"
            spec = importlib.util.spec_from_file_location("jennas_voz", ruta)
            if spec is None or spec.loader is None:
                return re.sub(r"\s+", " ", str(texto or "")).strip()
            jv = importlib.util.module_from_spec(spec)
            sys.modules["jennas_voz"] = jv
            spec.loader.exec_module(jv)
        return jv.para_voz(texto)
    except Exception:
        return re.sub(r"\s+", " ", str(texto or "")).strip()


def sintetizar(texto: str) -> bytes:
    """Texto → bytes de WAV (mono, 16-bit, 44 100 Hz).

    El modelo devuelve float32 en (1, N); acá se pasa a PCM 16-bit y se arma el WAV a mano para
    no escribir en disco (el server ya escribe el archivo final por su cuenta).
    """
    tts = _asegurar()
    with _LOCK:
        wav, _dur = tts.synthesize(preprocesar(texto), voice_style=_estilo(),
                                   total_steps=PASOS, speed=VELOCIDAD, lang=IDIOMA)
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def normalizar(audio: bytes) -> bytes:
    """Sube el volumen con la ganancia PROPIA de este motor y devuelve WAV.

    ANTE CUALQUIER PROBLEMA DEVUELVE LOS BYTES ORIGINALES: un fallo de ffmpeg no puede costar la
    voz — mejor volumen bajo que silencio (mismo contrato que tts_piper).
    """
    if not GANANCIA_DB:
        return audio
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-i", "pipe:0",
             "-af", f"volume={GANANCIA_DB}dB,alimiter=limit=0.95",
             "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
            input=audio, capture_output=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:
        pass
    return audio

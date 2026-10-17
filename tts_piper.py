"""Catálogo de voces LOCALES (Piper) para Jarvis — TTS oficial desde el 9-oct-2026.

Misma interfaz que `tts_eleven.py` (`catalogo` / `usar_voz` / `clave_de` / `disponible` /
`sintetizar` / `normalizar`) para que `jarvis-tablet.py` trate a los dos motores igual y los
endpoints `/voces` y `/voz` no cambien.

POR QUÉ EXISTE: Tavo dejó de pagar ElevenLabs (caro). El TTS oficial vuelve a Piper local.
ElevenLabs queda archivado (tag `elevenlabs-v1` + `~/.backups/2026-10-09-elevenlabs-archive/`)
y se reactiva con `JARVIS_TTS=eleven` sin tocar código.

⚠️ NO HAY VOZ CHILENA EN PIPER. El catálogo oficial (rhasspy/piper-voices, 177 voces) sólo
trae `es_ES` y `es_MX` para español (más `es_AR` que ya teníamos). Ver ARCHIVO.md § Historial
9-oct-2026 (2) para las alternativas evaluadas y descartadas.

Los modelos viven en `~/.hermes/tts-local/voices/<lang>/<region>/<nombre>/<calidad>/`, que es
la convención que ya usaba `jennas-voz.py § PIPER_MODEL`.
"""

import io
import os
import pathlib
import re
import subprocess
import sys
import wave

VOICES_DIR = pathlib.Path.home() / ".hermes" / "tts-local" / "voices"

# ── Velocidad del habla ──────────────────────────────────────────────────────
# 1.50 = la misma que usa `jennas-voz.py` para el carril local. No cambiar sin medir: es lo
# que hace que Jarvis suene pausada y no atropellada.
LENGTH_SCALE = float(os.environ.get("JARVIS_LENGTH_SCALE", "1.50"))

# ── Volumen ──────────────────────────────────────────────────────────────────
# MEDIDO 9-oct-2026 (no elegido a ojo): Piper YA entrega el audio normalizado a full scale
# (pico 0,0 dB, media -15,6 dB), o sea ~4,4 dB más fuerte que ElevenLabs crudo (~-20 dB).
# La referencia que Tavo tenía en el oído era ElevenLabs +8 dB ≈ **-13 dB de media**.
# Para Piper, +2 dB cae en -13,4 dB → misma sonoridad SIN que el limitador aplaste.
# Con +8 dB el limitador entra en acción (media -10,5) y comprime la dinámica de más.
# Se usa una variable PROPIA para no heredar el 8 que necesita ElevenLabs.
GANANCIA_DB = float(os.environ.get("JARVIS_VOLUMEN_DB_PIPER", "2"))

# ── Catálogo ─────────────────────────────────────────────────────────────────
# SOLO VOCES FEMENINAS (pedido de Tavo, 9-oct-2026). El género NO se asumió: se midió la F0
# mediana sintetizando la misma frase con cada voz y clasificando (>165 Hz femenina, <150
# masculina). Del catálogo español de Piper quedaron 5 femeninas; las masculinas (davefx 120 Hz,
# carlfm 123, sharvard speaker 0 a 117) y la dudosa (es_MX-ald, 158) se descartaron.
# `ruta` es relativa a VOICES_DIR. `speaker_id` sólo para voces multi-hablante (sharvard).
VOCES = {
    "daniela": {
        "ruta": "es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "nombre": "Daniela",
        "acento": "Argentina",
        "pais": "ar",
        "emoji": "🇦🇷",
        "descripcion": "Piper es_AR-daniela (high) — F0 184 Hz. La voz local histórica de Jarvis",
    },
    # Voz CLONADA localmente (piper1-gpl entrenado por Jennasys desde una muestra sintética de
    # ElevenLabs; ver LEEME.txt junto al modelo). No depende de ninguna nube: es la respuesta al
    # "no hay española/chilena de calidad en Piper". Medido: F0 169,6 Hz, pico 0 dB.
    "clara": {
        "ruta": "es/es_ES/clara/medium/es_ES-clara-medium.onnx",
        "nombre": "Clara (clon local)",
        "acento": "España",
        "pais": "es",
        "emoji": "🧬",
        "descripcion": "Clon entrenado local (epoch 189, val_mel 0.3958) — F0 170 Hz. No depende de la nube",
    },
    "peninsular": {
        "ruta": "es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "speaker_id": 1,
        "nombre": "Peninsular",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "Piper es_ES-sharvard (medium), speaker 1 — F0 199 Hz",
    },
    "esp_a": {
        "ruta": "es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx",
        "nombre": "Española A",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "Piper es_ES-mls_9972 (low), corpus MLS — F0 186 Hz",
    },
    "esp_b": {
        "ruta": "es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx",
        "nombre": "Española B",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "Piper es_ES-mls_10246 (low), corpus MLS — F0 200 Hz",
    },
    "mexicana": {
        "ruta": "es/es_MX/claude/high/es_MX-claude-high.onnx",
        "nombre": "Mexicana",
        "acento": "México",
        "pais": "mx",
        "emoji": "🇲🇽",
        "descripcion": "Piper es_MX-claude (high) — F0 178 Hz. Latinoamericano neutro",
    },
    # Candidata de Tavo, medida con `tools/probar-onnx-suelto.py` (10-oct-2026): tier `high`,
    # 22 050 Hz, F0 200,5 Hz FEMENINA y ESTABLE (6,64/6,79/6,69 s → dispersión 0,15 s, contra los
    # 15,65/18,99/9,70 s de `esp_b`), a nivel correcto (-16,4 dB, pico 0 → sin gain extra).
    # Repo de la comunidad (`HirCoir/Piper-TTS-Laura`), licencia **CC-BY-NC-4.0** → uso personal.
    "laura": {
        "ruta": "es/es_MX/laura/high/es_MX-laura-high.onnx",
        "nombre": "Laura",
        "acento": "México",
        "pais": "mx",
        "emoji": "🇲🇽",
        "descripcion": "Piper es_MX-laura (high) — F0 200 Hz, estable. Comunidad, licencia no comercial",
    },
    # Las otras dos voces del Space público `HirCoir/Piper-TTS-Spanish` (mismo autor que las gated
    # `piper-voice-es_MX-*`: esos repos piden aprobación MANUAL del autor, pero el Space tiene los
    # .onnx sueltos y son públicos). Medidas el 10-oct-2026 con `tools/probar-onnx-suelto.py`.
    "cortana": {
        "ruta": "es/es_MX/cortana/high/es_MX-cortana-high.onnx",
        "nombre": "Cortana",
        "acento": "México",
        "pais": "mx",
        "emoji": "🇲🇽",
        "descripcion": "Piper es_MX-cortana (high) — F0 199 Hz, 22 kHz y estable. Latinoamericano",
    },
    "gevy": {
        "ruta": "es/es_MX/gevy/high/es_MX-gevy-high.onnx",
        "nombre": "Gevy",
        "acento": "México",
        "pais": "mx",
        "emoji": "🇲🇽",
        "descripcion": "Piper es_MX-gevy (high) — F0 165 Hz, estable. La más fuerte de todas (-13 dB)",
    },
}

# Clave activa. `JARVIS_VOZ_PIPER` permite arrancar con otra sin tocar la tablet.
_DEFAULT = os.environ.get("JARVIS_VOZ_PIPER", "daniela")
VOZ_ACTIVA = _DEFAULT if _DEFAULT in VOCES else "daniela"

_voice = None                       # PiperVoice cacheada (se invalida al cambiar de voz)


# ── API del catálogo (igual forma que tts_eleven) ────────────────────────────
def clave_de(clave_o_id: str) -> str:
    """Clave del catálogo (acepta la clave o el nombre de archivo)."""
    k = str(clave_o_id or "").strip()
    if k in VOCES:
        return k
    for c, v in VOCES.items():
        if pathlib.Path(v["ruta"]).name == k:
            return c
    return k


def catalogo() -> dict:
    """Las voces elegibles + cuál está activa. Lo consume el botón de la tablet."""
    return {
        "voces": [
            {"clave": k, "activa": k == VOZ_ACTIVA,
             "nombre": v["nombre"], "acento": v["acento"], "pais": v["pais"],
             "emoji": v["emoji"], "descripcion": v["descripcion"]}
            for k, v in VOCES.items()
        ],
        "activa": VOZ_ACTIVA,
        "motor": "piper",
    }


def usar_voz(clave_o_id: str) -> dict:
    """Cambia la voz activa EN CALIENTE. Devuelve el catálogo nuevo.

    Invalida el caché del modelo: SIN esto, `_cargar_voz()` seguiría devolviendo la voz
    anterior y TODAS las claves sonarían igual (bug real detectado 9-oct-2026 midiendo el
    género de las voces: las 9 daban la misma F0 porque ninguna recargaba el modelo).
    """
    global VOZ_ACTIVA
    clave = str(clave_o_id or "").strip()
    if not clave:
        raise ValueError("voz vacía")
    clave = clave_de(clave)
    if clave not in VOCES:
        raise ValueError(f"voz Piper desconocida: {clave!r}")
    VOZ_ACTIVA = clave
    invalidar_cache()
    return catalogo()


def ruta_modelo(clave: str | None = None) -> str:
    """Ruta absoluta del .onnx de la voz (la activa, si no se pasa clave)."""
    v = VOCES[VOZ_ACTIVA if clave is None else clave_de(clave)]
    return str(VOICES_DIR / v["ruta"])


def speaker_id(clave: str | None = None):
    v = VOCES[VOZ_ACTIVA if clave is None else clave_de(clave)]
    return v.get("speaker_id")


def disponible() -> bool:
    """Piper es local: disponible si el modelo de la voz activa está en disco."""
    return pathlib.Path(ruta_modelo()).exists()


def faltantes() -> list:
    """Voces del catálogo cuyo .onnx no está descargado (para diagnóstico)."""
    return [k for k, v in VOCES.items() if not (VOICES_DIR / v["ruta"]).exists()]


# ── Síntesis ─────────────────────────────────────────────────────────────────
def _cargar_voz():
    """Carga (y cachea) la PiperVoice de la voz activa."""
    global _voice
    if _voice is None:
        from piper import PiperVoice
        ruta = ruta_modelo()
        if not pathlib.Path(ruta).exists():
            raise FileNotFoundError(f"falta el modelo Piper: {ruta}")
        _voice = PiperVoice.load(ruta)
        list(_voice.synthesize("listo", syn_config=_syn()))
    return _voice


def _syn():
    from piper.config import SynthesisConfig
    return SynthesisConfig(length_scale=LENGTH_SCALE, speaker_id=speaker_id())


def invalidar_cache():
    """Fuerza recargar el modelo en la próxima síntesis (tras cambiar de voz)."""
    global _voice
    _voice = None


def preprocesar(texto: str) -> str:
    """Limpieza de texto previa a Piper (misma que el carril local de jennas-voz).

    Se importa `jennas-voz.py` (nombre con guion) igual que hace la tablet, reusando la
    entrada de `sys.modules` si ya está cargado para no ejecutarlo dos veces.
    """
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
    """Texto → bytes de WAV (mono, 16-bit, sample rate del modelo)."""
    voz = _cargar_voz()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voz.config.sample_rate)
        for c in voz.synthesize(preprocesar(texto), syn_config=_syn()):
            wf.writeframes(c.audio_int16_bytes)
    return buf.getvalue()


def normalizar(audio: bytes) -> bytes:
    """Sube el volumen (misma ganancia que el carril ElevenLabs) y devuelve WAV.

    ANTE CUALQUIER PROBLEMA DEVUELVE LOS BYTES ORIGINALES: un fallo de ffmpeg no puede costar
    la voz — mejor volumen bajo que silencio.
    """
    if not GANANCIA_DB:
        return audio
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-i", "pipe:0",
             "-af", f"volume={GANANCIA_DB}dB,alimiter=limit=0.95",
             "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
            input=audio, capture_output=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:
        pass
    return audio

"""STT de ElevenLabs (Scribe) para Jarvis — el carril de dictado de la tablet.

POR QUÉ EXISTE (7-oct-2026). Tavo notó que el STT local **no capta la entonación**:
*"tenemos fotos de nola?, en expresión de voz lo lee plano como tenemos fotos de nola"*.
La puntuación interrogativa se deriva de la PROSODIA (la curva melódica), y Whisper la
pierde. Scribe la conserva (medido: `'¿Tenemos fotos de Nola?'` vs `'Tenemos fotos de Nola'`).

⚠️ HONESTIDAD SOBRE LA MEDICIÓN: lo verifiqué con audio **sintetizado** con la etiqueta
`[curious]` de ElevenLabs v3 — una entonación IDEALIZADA y exagerada, que hasta el Whisper
local marcó con "?". O sea: **el test no reproduce el caso real de Tavo** (pregunta
retórica/dubitativa con su voz). Este módulo es la apuesta de que Scribe lo hace mejor en
voz real; la confirmación tiene que venir del uso, no de mi banco de pruebas.

LATENCIA MEDIDA (mismo audio, p50 de 3 corridas):
  Scribe v1     0.68–0.76 s   (nube)
  Whisper local 0.55–0.57 s   (GPU en VRAM)
Diferencia ~0.15 s — aceptable para dictado. La clave sigue en ~/.secrets/elevenlabs.txt.

El Whisper local NO se retira: `JARVIS_STT=whisper` vuelve sin tocar código, y si Scribe
falla o no hay key, se cae a Whisper (degradación limpia, igual que el motor de voz).
"""

import json
import pathlib
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1"
CLAVE = pathlib.Path.home() / ".secrets" / "elevenlabs.txt"

# scribe_v1 es el que verificamos. OJO (corregido 7-oct-2026): scribe_v2 NO cuesta más —
# el rate card publica $0.22/h para v2 igual que v1. Lo que agrega costo es el *keyterm
# prompting* opcional (+$0.05/h ≈ +23%), no el modelo. O sea: pasar a v2 es gratis y trae
# más precisión; queda pendiente porque Tavo acaba de validar el tono con v1 y cambiarlo
# invalidaría esa confirmación sin medir. No cambiar dos cosas a la vez.
MODELO = "scribe_v1"

# El idioma va explícito: Tavo habla español chileno y con `language_code` el modelo no
# gasta presupuesto adivinando. Con null (auto) también funciona, pero medimos con es.
IDIOMA = "es"

# tag_audio_events se APAGA: para dictado queremos el texto, no "(risas)" metido adentro,
# que ensuciaría el prompt del agente.
ETIQUETAR_EVENTOS = False


def clave():
    """La API key. Devuelve None si el archivo no está (degrada a Whisper)."""
    try:
        v = CLAVE.read_text().strip()
        return v or None
    except OSError:
        return None


def disponible() -> bool:
    """¿Se puede usar Scribe? Sin key no — y conviene saberlo ANTES del turno."""
    return bool(clave())


def transcribir(ruta, timeout=120):
    """Audio → texto, con la MISMA firma que `jennas-voz.transcribir`.

    Multipart armado a mano (sin `requests`, que no está en el venv del proyecto).
    Lanza la excepción si falla: el caller decide si cae a Whisper.
    """
    k = clave()
    if not k:
        raise RuntimeError("no hay API key de ElevenLabs (~/.secrets/elevenlabs.txt)")

    ruta = pathlib.Path(ruta)
    datos = ruta.read_bytes()
    bnd = "----jarvisstt"

    def campo(nombre, valor):
        return (f'--{bnd}\r\nContent-Disposition: form-data; name="{nombre}"\r\n\r\n'
                f'{valor}\r\n').encode()

    tip = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".webm": "audio/webm",
           ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac"}
    ctype = tip.get(ruta.suffix.lower(), "application/octet-stream")

    cuerpo = (campo("model_id", MODELO)
              + campo("language_code", IDIOMA)
              + campo("tag_audio_events", "true" if ETIQUETAR_EVENTOS else "false")
              + (f'--{bnd}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="a{ruta.suffix}"\r\nContent-Type: {ctype}\r\n\r\n').encode()
              + datos + f'\r\n--{bnd}--\r\n'.encode())

    req = urllib.request.Request(f"{API}/speech-to-text", data=cuerpo, method="POST")
    req.add_header("xi-api-key", k)
    req.add_header("Content-Type", f"multipart/form-data; boundary={bnd}")

    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return (d.get("text") or "").strip()

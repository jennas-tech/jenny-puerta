"""Voz de ElevenLabs para Jarvis: catálogo de voces + tags expresivos.

Separado de `jennas-voz.py` a propósito: así se prueba sin tocar el carril de voz que
funciona. Si algo falla, `hablar()` de Piper sigue intacto.

Lo que resuelve: Piper tiene `length_scale` y nada más — no sabe de emoción. Este motor
sí: los tags (`[sighs]`, `[sarcastic]`…) los **interpreta** en vez de leerlos, y el LLM
los escribe en el texto. Los tags NO deben llegar a la pantalla → `quitar_tags()`.

Medido (4-oct-2026): primer byte 356 ms por `/stream` — más rápido que el Piper local
(693 ms). El endpoint REST acepta `eleven_v4_turbo` sin el WebSocket de Text to Dialogue.
"""

import json
import os
import pathlib
import re
import subprocess
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1"
CLAVE = pathlib.Path.home() / ".secrets" / "elevenlabs.txt"

# Voz por DEFECTO: **Clara** (7-oct-2026, pedido de Tavo: "las españolas tienen las voces más
# expresivas, pone a clara como default"). Es-ES (`es-ES`), peninsular, joven, `professional`
# — en nuestra cuenta (GET /v1/voices → 200). Antes fue Catalina (chilena) ese mismo día, y
# Valeria (argentina) antes.
# OJO: la voz es CONFIG, no secreto — es un identificador público
# del TTS, no una credencial. Va en JARVIS_ELEVEN_VOZ del env-tablet; no la muevas a ~/.secrets/.
# Este literal es solo el FALLBACK si falta la variable de entorno; la fuente de verdad es el env.
VOZ = os.environ.get("JARVIS_ELEVEN_VOZ", "LudcwvHIZaqQOcQfVZSY")
MODELO = os.environ.get("JARVIS_ELEVEN_MODELO", "eleven_v4_turbo")

# ── Catálogo de voces seleccionables (7-oct-2026, pedido de Tavo) ─────────────
# "quiero probar esta voz en jarvis… puedes hacer un botón para cambiar voces, con las
#  que estén disponibles. en este caso Valeria argentina y este nueva Olivia chilena".
#
# La voz es CONFIG, no secreto: son IDs públicos del TTS. El catálogo vive acá (código)
# porque el botón tiene que poder listarlas; `JARVIS_ELEVEN_VOZ` sigue mandando como
# SELECCIÓN inicial, así que cambiar de voz desde la tablet NO reescribe el env (el
# servicio sigue arrancando con la marcada como inicial).
#
# OJO: "Olivia Pro" NO está en nuestra cuenta de Voice Library propia — es una voz
# COMPARTIDA (public_owner_id d8f35aa4…). La API la sintetiza igual (verificado: HTTP 200,
# MP3 válido), pero no aparece en GET /v1/voices. Por eso el catálogo está hardcodeado.
VOCES = {
    "valeria": {
        "voz_id": "9oPKasc15pfAbMr7N6Gs",
        "nombre": "Valeria",
        "acento": "Argentina",
        "pais": "ar",
        "emoji": "🇦🇷",
        "descripcion": "Cheerful, Youthful, Catchy — professional, accent=argentine",
    },
    "catalina": {
        # Reemplaza a Olivia (7-oct-2026, pedido de Tavo: "cambia Olivia por esta voz").
        # A diferencia de Olivia/Cami, esta SÍ está en nuestra cuenta (GET /v1/voices → 200),
        # así que además aparece en shared-voices como high_quality.
        "voz_id": "6Gr4AVmTax1pMJO0lHRK",
        "nombre": "Catalina",
        "acento": "Chile",
        "pais": "cl",
        "emoji": "🇨🇱",
        "descripcion": "Catalina — Chilean Spanish (es-CL, young, conversational)",
    },
    "camila": {
        # OJO con el nombre: el voice_id dice "Cami" (`Cami - Warm & Energetic Spanish`)
        # pero Tavo la pidió como "Camila". Hay OTRA voz DISTINTA llamada literalmente
        # "Camila" (oJIuRMopN0sojGjwD6rQ, accent=chilean, use_case=conversational) que
        # también está en la cuenta — no confundirlas si algún día se agrega.
        "voz_id": "dZFAIFXrSaHlJY1R7xj2",
        "nombre": "Camila",
        "acento": "Chile",
        "pais": "cl",
        "emoji": "🇨🇱",
        "descripcion": "Cami — Warm & Energetic Spanish (es-CL, young, upbeat)",
    },
    "sara": {
        # Agregada el 7-oct-2026 (pedido de Tavo: "nueva voz para jarvis, españa").
        # Es "Sara Martin - Light, Intimate and Tender", de la Voice Library COMPARTIDA
        # (accent=peninsular, Spain) → la API la sintetiza pero no lista en GET /v1/voices.
        # OJO: NO es la "Sarah" original del proyecto (EXAVITQu4vr4xnSDxMaL, el stock sin
        # acento que Tavo eligió el 4-oct y después reemplazó por Valeria). Son voces
        # distintas con nombre parecido — por eso la clave es "sara" y no "sarah".
        "voz_id": "Ir1QNHvhaJXbAGhT50w3",
        "nombre": "Sara",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "Sara Martin — Light, Intimate and Tender (accent=peninsular)",
    },
    "clara": {
        # Agregada el 7-oct-2026 (pedido de Tavo: "otra mas LudcwvHIZaqQOcQfVZSY").
        # "Clara - Elegant, Sultry and Soft" — es-ES (peninsular), femenina, joven.
        # A diferencia de Sara, esta SÍ está en nuestra cuenta (GET /v1/voices → 200,
        # category=professional), así que además aparece en shared-voices.
        "voz_id": "LudcwvHIZaqQOcQfVZSY",
        "nombre": "Clara",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "Clara — Elegant, Sultry and Soft (es-ES, young, formal)",
    },
    "alba": {
        # Agregada el 7-oct-2026 (pedido de Tavo: "otra mas Y78FoQowFAZ83lz3YjCw").
        # "Alba Noches - Velvety, Intimate & Warm" — es-ES (peninsular), femenina,
        # middle_aged, calm, use_case=narrative_story. En la cuenta (professional).
        "voz_id": "Y78FoQowFAZ83lz3YjCw",
        "nombre": "Alba",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "Alba Noches — Velvety, Intimate & Warm (es-ES, middle_aged, calm)",
    },
    "maricarmen": {
        # Agregada el 7-oct-2026 (pedido de Tavo: "otra YDDaC9XKjODs7hY78qEW").
        # "MariCarmen - Hurried and Intimidating" — es-ES (peninsular), femenina,
        # middle_aged. Voice Library COMPARTIDA (GET /v1/voices → 400).
        # OJO: es una voz de PERSONAJE (`use_case=characters_animation`,
        # `descriptive=anxious`, "hurried and intimidating") — no está pensada para
        # conversación de asistente. Suma al catálogo como opción; no es candidata a default.
        "voz_id": "YDDaC9XKjODs7hY78qEW",
        "nombre": "MariCarmen",
        "acento": "España",
        "pais": "es",
        "emoji": "🇪🇸",
        "descripcion": "MariCarmen — Hurried and Intimidating (es-ES, anxious, characters_animation)",
    },
    "luisa": {
        # Agregada el 7-oct-2026 (pedido de Tavo: "otra ByzPpv6d6pT90CCpgb7k").
        # "Luisa - Smooth, Friendly and Engaging" — Voice Library COMPARTIDA (GET /v1/voices → 400).
        # OJO con el acento: el label dice `accent=latin american` pero el `locale` es **es-AR**
        # (Argentina). Se muestra 🇦🇷 por el locale; si suena más neutra que rioplatense, es eso.
        # middle_aged, `excited`, `narrative_story`.
        "voz_id": "ByzPpv6d6pT90CCpgb7k",
        "nombre": "Luisa",
        "acento": "Argentina",
        "pais": "ar",
        "emoji": "🇦🇷",
        "descripcion": "Luisa — Smooth, Friendly and Engaging (locale es-AR, latin american, middle_aged)",
    },
    "jhenny": {
        # Agregada el 7-oct-2026 (pedido de Tavo: "otra D9MdulIxfrCUUJcGNQon").
        # "Jhenny - Gentle, Soothing and Meditative" — Voice Library COMPARTIDA (→ 400).
        # `accent=latin american`, `locale=es-AR` (Argentina), middle_aged, `calm`, narrative_story.
        # DETALLE: el nombre real es "Jhenny", a una letra de "Jenny" (el alias de voz de Jennas).
        # La clave va con la h para no confundirla con la asistente.
        "voz_id": "D9MdulIxfrCUUJcGNQon",
        "nombre": "Jhenny",
        "acento": "Argentina",
        "pais": "ar",
        "emoji": "🇦🇷",
        "descripcion": "Jhenny — Gentle, Soothing and Meditative (locale es-AR, calm, middle_aged)",
    },
    "jenny": {
        # VOZ PROPIA DISEÑADA POR TAVO (7-oct-2026: "diseñé una voz propia para jarvis").
        # `category=generated` → NO es de la Voice Library, la creó él con Voice Design y vive
        # en nuestra cuenta (GET /v1/voices → 200, no aparece en shared-voices).
        # Descripción escrita por él, verbatim:
        #   "una voz de mujer joven, chilena, una voz grave y profunda, sensual y carismática,
        #    con mucha expresión y calidad emocional."
        # Settings guardados en la voz: stability 0.5, similarity_boost 0.75, style 0.0, speed 1.0
        # (los aplica ElevenLabs por defecto; `sintetizar()` no los pisa).
        # OJO CON EL NOMBRE: la voz se llama "jenny" y en el pool ya había `jhenny`
        # ("Jhenny - Gentle, Soothing and Meditative", de la LIBRERÍA). Se diferencian por la
        # clave (jenny/jhenny) y por el acento en el menú: esta es 🇨🇱 Chile, aquella 🇦🇷 Argentina.
        "voz_id": "mhHG4B1NXm5g4p7hohB1",
        "nombre": "Jenny",
        "acento": "Chile",
        "pais": "cl",
        "emoji": "🇨🇱",
        "descripcion": "Voz propia de Tavo (Voice Design): joven, grave y profunda, sensual y carismática",
    },
}

# Voz ACTIVA en este proceso. El server lo cambia en caliente (POST /voz); no requiere
# reiniciar el servicio. `sintetizar()` lee esta variable en cada llamado.
VOZ_ACTIVA = VOZ


def catalogo() -> dict:
    """Las voces elegibles + cuál está activa. Lo consume el botón de la tablet."""
    return {
        "voces": [
            {"clave": k, "activa": k == clave_de(VOZ_ACTIVA), **v}
            for k, v in VOCES.items()
        ],
        "activa": clave_de(VOZ_ACTIVA),
        "motor": "eleven",
    }


def clave_de(voz_id: str) -> str:
    """Clave del catálogo para un voice_id (o el propio id si no está en el catálogo)."""
    for k, v in VOCES.items():
        if v["voz_id"] == voz_id:
            return k
    return voz_id


def usar_voz(clave_o_id: str) -> dict:
    """Cambia la voz activa EN CALIENTE. Acepta la clave del catálogo o un voice_id crudo.

    Devuelve el catálogo (con la nueva activa). Lanza ValueError si no se reconoce.
    No valida contra la API: el ID puede ser una voz compartida que no liste en
    /v1/voices (caso Olivia) y la síntesis se encarga; si el ID no existe, el turno
    cae a Piper, que es la degradación que ya existe.
    """
    global VOZ_ACTIVA
    clave = str(clave_o_id or "").strip()
    if not clave:
        raise ValueError("voz vacía")
    if clave in VOCES:
        VOZ_ACTIVA = VOCES[clave]["voz_id"]
        return catalogo()
    # voice_id crudo: aceptarlo solo si matchea el formato de ElevenLabs (20 chars alfanum.)
    if re.fullmatch(r"[A-Za-z0-9]{20}", clave):
        VOZ_ACTIVA = clave
        return catalogo()
    raise ValueError(f"voz desconocida: {clave!r}")

# Los tags que el modelo puede escribir. Van en el prompt; el motor los interpreta.
# Si agregás uno, verificar que exista en la doc de v4 — un tag inválido se lee en voz alta.
TAGS = [
    "sighs", "laughs", "whispers", "excited", "sarcastic", "curious",
    "sad", "happy", "surprised", "tired", "amused", "serious",
    # Agregados 5-oct-2026 (Tavo). La doc de v4 los lista como válidos:
    # 'thoughtful' (calma/reflexiva) y 'playful' (alta energía) son de la rueda
    # de emociones; 'inhales deeply' y 'clears throat' son reacciones humanas.
    "thoughtful", "playful", "inhales deeply", "clears throat",
]

# OJO: este patrón se usa para LIMPIAR (lo que va a pantalla), no para validar.
# Acepta cualquier contenido entre corchetes y el filtro fino lo hace `quitar_tags`.
_PATRON_TAGS = re.compile(r"\[([^\[\]]{1,80})\]")

# Máximos de lo que se considera "indicación de dirección" y no texto real.
# Una cue de v4 es natural-language y puede ser larga ('[out of breath after
# running up the stairs]', ejemplo de la doc). Si pasa de esto es más probable
# que sea texto legítimo entre corchetes.
_MAX_CHARS_CUE = 60
_MAX_PALABRAS_CUE = 8
_PUNTUACION_ORACION = re.compile(r"[.!?;:]")
# Fin de oración: dónde van las cues (la doc dice "before the words it affects";
# los ejemplos las ponen al inicio o tras un cierre de frase).
_FIN_ORACION = re.compile(r"[.!?…\n]\s*$")


def clave():
    """La API key, del archivo (nunca del código ni del entorno)."""
    try:
        return CLAVE.read_text().strip()
    except FileNotFoundError:
        return ""


# ── Volumen (6-oct-2026, pedido de Tavo) ─────────────────────────────────────────
# "el volumen de Jarvis es un poco bajo y tengo que ponerlo siempre muy alto en mi tablet".
# MEDIDO sobre audios reales: ElevenLabs entrega ~-20 dB de media con picos cerca de 0 dB — o
# sea que suena bajo Y no tiene margen parejo (unos clips llegan al techo y otros no).
# Fix: **ganancia fija + limitador**. Se eligió sobre `loudnorm` a propósito: medido, loudnorm
# es INCONSISTENTE por clip (a un `relleno` de -19,4 lo dejaba en -19,9, sin tocarlo), mientras
# que la ganancia fija da el MISMO salto en todos (+8 dB → todos ~-13,3 dB). Y la voz tiene que
# sonar igual de fuerte siempre, no solo "en promedio".
# +8 dB con `alimiter`: medido `Flat factor: 0.000000` (sin distorsión); de +8 a +12 el
# limitador ya aplasta y el volumen casi no sube (rendimientos decrecientes), así que 8 es el
# punto dulce. Ajustable sin tocar código: `JARVIS_VOLUMEN_DB` en el env del servicio.
GANANCIA_DB = os.environ.get("JARVIS_VOLUMEN_DB", "8")


def normalizar(audio: bytes) -> bytes:
    """Devuelve el MP3 con el volumen subido. **Ante cualquier problema, los bytes ORIGINALES.**

    Un fallo acá no puede costar la voz: mejor volumen bajo que silencio. Por eso el try entero
    cae al `return audio`. Costo medido: 78 ms sobre un clip corto (la síntesis son 745 ms), y
    corre mientras el resto del turno sigue generándose — no bloquea el stream.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error",
             "-i", "pipe:0",
             "-af", f"volume={GANANCIA_DB}dB,alimiter=limit=0.95",
             "-c:a", "libmp3lame", "-b:a", "128k",
             "-f", "mp3", "pipe:1"],
            input=audio, capture_output=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:
        pass
    return audio



def disponible():
    """¿Se puede usar? Sin key no hay voz — y conviene saberlo ANTES del turno."""
    return bool(clave())


def quitar_tags(texto):
    """Saca los tags del texto que va a la PANTALLA (el motor auditivo ya los interpretó).

    Tres capas, porque el modelo NO está limitado a nuestra lista blanca: v4 acepta
    cualquier cue en lenguaje natural, así que un sinónimo ('[playful]', '[softly]') se
    filtraba a pantalla como texto de máquina (bug detectado 5-oct-2026).

    1. Si el contenido está en `TAGS` → se borra, sin importar el resto.
    2. Si no, se borra igual si PARECE dirección: sin puntuación de oración, dentro de
       los máximos, y **en posición de cue** (inicio del texto o justo tras cerrar
       una oración). La posición es lo que distingue un cue de una cita inline:
       `[sighs] Hola` es cue; `Ver [Gadamer, 1960] para más` es una cita y se deja.
    3. Cualquier otra cosa se deja tal cual — mejor que se lea a que desaparezca
       texto con sentido."""

    def _limpiar(m):
        interior = m.group(1).strip()
        if interior.lower() in TAGS:
            return ""
        if _PUNTUACION_ORACION.search(interior):
            return m.group(0)
        if len(interior) > _MAX_CHARS_CUE:
            return m.group(0)
        if len(interior.split()) > _MAX_PALABRAS_CUE:
            return m.group(0)
        antes = texto[:m.start()]
        if antes and not _FIN_ORACION.search(antes):
            return m.group(0)          # inline: probablemente cita, no cue
        return ""

    limpio = _PATRON_TAGS.sub(_limpiar, texto)
    # Un tag solo dejaba doble espacio ('[sighs]  Hola' → '  Hola').
    return re.sub(r"[ \t]{2,}", " ", limpio).strip()


def instruccion_tags():
    """El bloque que se le agrega al prompt para que escriba los tags."""
    return (
        "\n\nEXPRESIVIDAD. Tienes una voz que interpreta indicaciones entre corchetes; los "
        "escribes y ella los ACTÚA (no los lee en voz alta). Úsalos con moderación: uno o "
        "dos por respuesta, y solo si de verdad aportan. Los que entiende:\n"
        + ", ".join(f"[{t}]" for t in TAGS)
        + "\nEjemplo: [sighs] Son las tres y veinte. [sarcastic] Y otra vez te olvidaste del almuerzo."
        + "\nPara una pausa reflexiva escribe tres puntos: `…`."
    )


def sintetizar(texto, timeout=60):
    """Texto → bytes de MP3. Lanza la excepción de la API si algo falla (el caller decide).

    Usa `/stream`: el REST simple tarda 831 ms contra 356 ms del streaming — la diferencia
    se oye."""
    k = clave()
    if not k:
        raise RuntimeError("no hay API key de ElevenLabs (~/.secrets/elevenlabs.txt)")

    req = urllib.request.Request(f"{API}/text-to-speech/{VOZ_ACTIVA}/stream", method="POST")
    req.add_header("xi-api-key", k)
    req.add_header("Content-Type", "application/json")
    cuerpo = json.dumps({
        "text": texto,
        "model_id": MODELO,
        "output_format": "mp3_44100_128",
    }).encode()

    trozos = []
    with urllib.request.urlopen(req, cuerpo, timeout=timeout) as r:
        while True:
            t = r.read(4096)
            if not t:
                break
            trozos.append(t)
    return b"".join(trozos)

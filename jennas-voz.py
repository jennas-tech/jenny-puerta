#!/usr/bin/env python3
"""
jennas-voz — Bucle de voz local: micrófono → STT → decisión → respuesta → voz.

Diseño (3-oct-2026, [[concepts/arquitectura-jarvis]]):
  - STT: faster-whisper small (local, GPU)
  - Decisión: deepseek-v4.1-flash sin pensamiento (ollama.com) — el mismo modelo
    superior con reasoning_effort=none, que solo enruta; el pensamiento se enciende
    cuando la frase pasa a Jennas
  - Respuesta: Jennas superior (`hermes chat -q`) o handler local si es trivial
  - Voz: Piper local (primer sonido ~90 ms)
  - Pantalla: la TV (HDMI del PC) muestra la conversación

Un proceso, modelos residentes (el orden paga por el desorden: RAM compra tiempo).
"""
import os, sys, json, re, time, subprocess, threading, urllib.request, wave
import numpy as np
from pathlib import Path

HOME = Path.home()
# El proyecto es autocontenido: todo se resuelve desde donde vive este archivo,
# NO desde ~/.hermes/voz. Así el código puede versionarse sin arrastrar estado.
AQUI = Path(__file__).resolve().parent
VOZ = AQUI                      # runtime (state.json, entrada.wav, voz.wav) vive acá
PIPER_MODEL = os.environ.get(
    "JARVIS_VOZ",
    str(HOME / ".hermes/tts-local/voices/es/es_AR/daniela/high/es_AR-daniela-high.onnx"))
STATE = VOZ / "state.json"
WHISPER_MODEL = os.environ.get("JARVIS_WHISPER", "small")

# ── vocabulario para el STT ──
# El STT no conoce los términos del mundo de Tavo. Sesgarlo con ellos corrige errores de
# palabra suelta (Dennett→"Denet") sin tocar el modelo ni gastar cómputo.
# Fuente: el propio vault (los términos que Tavo realmente usa). Si el vault no está,
# cae a una lista mínima de nombres propios frecuentes.
def _cargar_vocabulario(max_palabras=120):
    """Términos distintivos del mundo de Tavo, del propio vault. Filtra palabras comunes
    (stopwords) para que el sesgo sea denso en nombres propios y jerga, no en 'Los/Por/Las'."""
    minima = ("Dennett, Gadamer, Russell, Hume, Minsky, Parfit, Ricoeur, Gazzaniga, "
              "Obsidian, Moon Reader, vault, OpenViking, Jennas, JennasSys, "
              "Jeskai, Pauper, Standard, Modern, Commander, Arena, MTG, "
              "Crunchyroll, YouTube, Netflix, Prime Video, Disney")
    STOP = {
        "Los", "Las", "Por", "Para", "Con", "Sin", "Del", "Una", "Uno", "Este", "Esta",
        "Esto", "Ese", "Esa", "Cuando", "Como", "Pero", "Más", "Menos", "Todo", "Toda",
        "Hay", "Está", "Están", "Son", "Fue", "Ser", "Hacer", "Tiene", "Puede", "Cada",
        "Regla", "Estado", "Lección", "Trigger", "Patrón", "Ejemplo", "Nota", "Caso",
        "Pitfall", "Síntoma", "Causa", "Fix", "Tabla", "Lista", "Texto", "Fecha", "Hora",
        "Sobre", "Desde", "Hasta", "Entre", "Según", "Mismo", "Otra", "Otro", "Algo",
        "Nada", "Nunca", "Siempre", "También", "Además", "Aunque", "Mientras", "Porque",
    }
    candidatos = [
        HOME / "Dropbox/obsidian/tavo/llm-wiki/concepts",
        HOME / "Dropbox/obsidian/tavo/llm-wiki/entities",
        HOME / "Dropbox/obsidian/tavo/llm-wiki",
    ]
    for base in candidatos:
        if not base.exists():
            continue
        try:
            from collections import Counter
            c = Counter()
            # nombres propios / jerga: palabra capitalizada, 4+ letras (evita siglas cortas)
            for f in base.rglob("*.md"):
                try:
                    txt = f.read_text(errors="ignore")
                except Exception:
                    continue
                for p in txt.replace(".", " . ").replace(",", " , ").split():
                    w = p.strip("()[]*`\"'—–:;!")
                    if len(w) >= 4 and w[0].isupper() and w[1:].islower() and w not in STOP:
                        c[w] += 1
            comunes = [w for w, n in c.most_common(max_palabras * 2) if n >= 3]
            if comunes:
                return minima + ", " + ", ".join(comunes[:max_palabras])
        except Exception:
            pass
    return minima

_VOCABULARIO = _cargar_vocabulario()

# ── estado para la pantalla ──
_lock = threading.Lock()

# ── cierre limpio: la captura de audio NO debe sobrevivir al proceso ──
# Defecto real (3-oct-2026, reportado por Tavo: *"cuando se cierra la app también debe
# desconectarse la captura de audio"*): si el bucle moría por SIGTERM, el `finally` de
# `grabar()` NO corría y `pw-record` quedaba huérfano reteniendo el micrófono.
# Fix: registro global de subprocesos + atexit + handler de SIGTERM/SIGINT que los mata.
_SUBPROCS = set()

def _registrar(p):
    _SUBPROCS.add(p)
    return p

def _matar_subprocs():
    for p in list(_SUBPROCS):
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except Exception:
                    p.kill()
        except Exception:
            pass
        _SUBPROCS.discard(p)

def _instalar_cierre_limpio():
    import atexit, signal
    atexit.register(_matar_subprocs)
    def _handler(sig, frame):
        print(f"\n[jennas-voz] señal {sig} — suelto la captura de audio y salgo", flush=True)
        _matar_subprocs()
        raise SystemExit(0)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass

def publicar(turno):
    """Escribe el estado que la pantalla (TV) lee por polling."""
    try:
        st = json.loads(STATE.read_text()) if STATE.exists() else {"turnos": []}
    except Exception:
        st = {"turnos": []}
    st["turnos"].append(turno)
    st["turnos"] = st["turnos"][-12:]
    st["actualizado"] = time.strftime("%H:%M:%S")
    with _lock:
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))

def pon_estado(nombre):
    """Cambia el sprite que muestra la pantalla (idle/listening/thinking/…).
    No pisa los turnos: solo el campo 'estado' del mismo JSON."""
    try:
        st = json.loads(STATE.read_text()) if STATE.exists() else {"turnos": []}
    except Exception:
        st = {"turnos": []}
    if st.get("estado") == nombre:
        return
    st["estado"] = nombre
    with _lock:
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))

# ── STT ──
_whisper = None
def _decodificar(wav_path, rate=16000):
    """Decodifica cualquier audio a float32 mono 16k con ffmpeg -> numpy.
    Evita el decodificador de faster-whisper (incompatibilidad de `av`)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", str(wav_path),
                        "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)

def _quitar_retumbo(a, rate=16000, fc=120):
    """Pasa-altos IIR de 1er orden (sin scipy), aplicado adelante y atrás (fase cero).
    El "ruido blanco" del living es en realidad ~150 Hz (64% de la energía); la voz
    vive en 300-3000 Hz, así que el filtro saca el retumbo sin tocar la voz."""
    rc = 1.0/(2*np.pi*fc)
    al = rc/(rc + 1.0/rate)
    def una_pasada(x):
        y = np.empty_like(x); prev_x = x[0]; prev_y = 0.0
        for i in range(len(x)):
            prev_y = al*(prev_y + x[i] - prev_x)
            prev_x = x[i]; y[i] = prev_y
        return y
    ida = una_pasada(a)
    vuelta = una_pasada(a[::-1])[::-1]
    return ((ida + vuelta)/2.0).astype(np.float32)

def _normalizar(a, objetivo_db=-20.0, techo_db=-3.0):
    """Lleva el audio a un nivel parejo para Whisper, sin pasar del techo (evita
    re-saturar). Medido 3-oct: Whisper transcribe igual de −6 a −24 dB → el nivel no
    importa mientras NO haya clipping. Esto desacopla la perilla del micrófono de la
    calidad de transcripción."""
    rms = float(np.sqrt(np.mean(a**2))) + 1e-9
    pico = float(np.max(np.abs(a))) + 1e-9
    g = 10**(objetivo_db/20) / rms
    if pico * g > 10**(techo_db/20):          # limitar por el pico
        g = 10**(techo_db/20) / pico
    return (a * g).astype(np.float32)

def transcribir(wav_path, modelo=None):
    """Transcribe con Whisper local. `modelo` permite pasar un modelo ya cargado distinto
    del global (la tablet usa `large-v3-turbo` para mejor precisión; la TV sigue en `small`).
    Si es None, usa el modelo del bucle (WHISPER_MODEL)."""
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    m = _whisper if modelo is None else modelo
    audio = _decodificar(wav_path)
    if audio.size == 0:
        return ""
    audio = _quitar_retumbo(audio)     # saca el retumbo de baja frecuencia
    audio = _normalizar(audio)         # nivel parejo, sin re-saturar
    # VOCABULARIO: sesga la decodificación hacia los términos de Tavo (Dennett, Gadamer…).
    # MEDIDO 3-oct-2026 con audio real: el efecto es REAL pero MARGINAL (~1 palabra distinta).
    # "Dennett"→"Net" NO se arregla solo con el prompt: la señal acústica del nombre no está
    # (se pierde en la articulación + ruido). El lever de fondo es un modelo mayor
    # (JARVIS_WHISPER=medium, +0.3 GB VRAM) — medido: "Net"→"Enet", "Sanía"→"asaniga".
    # Se deja el vocabulario porque no cuesta nada y ayuda en el margen, pero NO se vende
    # como solución: los nombres propios siguen siendo el techo del STT en modo voz.
    segs, _ = m.transcribe(audio, language="es", beam_size=1, vad_filter=True,
                           initial_prompt=_VOCABULARIO or None)
    return "".join(s.text for s in segs).strip()

# ── grabación con VAD (Silero) ──
# RMS es la herramienta EQUIVOCADA acá: con ruido blanco de fondo (−33 dB, constante,
# todo el día en el living) el RMS del ruido se confunde con voz. Tavo lo dijo directo:
# *"en el living tengo ruido blanco todo el día, pero no debería ensuciar la voz"* —
# y tiene razón: Whisper/Silero filtran ruido estacionario; lo que fallaba era MI detector.
# Ahora: VAD Silero (incluido en faster-whisper, corre en onnxruntime local) decide si
# hay habla real, no un umbral de energía.
#
# CORTE POR PAUSA SOSTENIDA (3-oct-2026, corrección de Tavo: *"tengo que hablar corto,
# se corta rápido, no puedo explayarme"*). El diseño viejo cortaba al PRIMER bloque sin
# voz → una pausa de un segundo para pensar mataba la frase. Y max_s=15 te techa la
# frase entera. Ahora: se tolera una pausa de hasta `pausa_max` s SIN voz antes de
# cerrar, y el techo sube a 35 s. Es el trade-off clásico de VAD: tolerar pausas cuesta
# ~pausa_max segundos de latencia al final de cada turno.
def grabar(out, max_s=35, rate=16000,
           umbral_vad=0.5, min_habla_ms=250, seg_voz=0.8, pausa_max=2.5):
    """Graba hasta detectar una PAUSA SOSTENIDA (pausa_max s sin voz) DESPUÉS de haber
    hablado. Instrumentado: imprime el nivel (RMS dB) y si el VAD ve voz.
    - seg_voz: ventana de análisis del VAD (s). Debe ser ≈ el intervalo de chequeo
               (0,8 s) para que el silencio medido sea real y no heredado de la ventana.
    - pausa_max: silencio continuo que cierra el turno (s). Súbelo si Tavo necesita
                 pausar más para pensar; es exactamente la latencia extra al cerrar."""
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    proc = _registrar(subprocess.Popen(["pw-record", "--rate", str(rate), "--channels", "1",
                             "--format", "s16", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    frames, t0, hablado = [], time.time(), False
    ultimo_voz = t0                    # instante del último bloque CON voz
    n_ventana = int(seg_voz * 10)      # frames en la ventana del VAD (100 ms c/u)
    try:
        while time.time() - t0 <= max_s:
            buf = proc.stdout.read(rate // 10 * 2)   # 100 ms
            if not buf:
                break
            frames.append(buf)
            if len(frames) % 8 == 0:                 # cada ~0,8 s, correr el VAD
                a = np.frombuffer(b"".join(frames[-n_ventana:]),
                                  dtype=np.int16).astype(np.float32)/32768.0
                rms = float(np.sqrt(np.mean(a**2))) if a.size else 0.0
                db = 20*np.log10(rms+1e-9)
                try:
                    segs = get_speech_timestamps(a, VadOptions(
                        threshold=umbral_vad, min_speech_duration_ms=min_habla_ms,
                        min_silence_duration_ms=300), sampling_rate=rate)
                except Exception:
                    segs = []
                tiene = len(segs) > 0
                ahora = time.time()
                print(f"    [mic] {db:6.1f} dB · voz={'SÍ' if tiene else 'no'}", flush=True)
                if tiene:
                    hablado = True
                    ultimo_voz = ahora
                elif hablado:
                    # solo cierra si el silencio ya es SOSTENIDO (no una pausa para pensar)
                    silencio = ahora - ultimo_voz
                    if silencio >= pausa_max:
                        print(f"    [mic] pausa de {silencio:.1f}s → cierro el turno", flush=True)
                        break
    finally:
        proc.terminate()
        try: proc.wait(timeout=2)
        except Exception: proc.kill()
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    return hablado

# ── TTS local ──
# Velocidad: length_scale > 1 = más pausado.
# Pedido de Tavo (3-oct: "habla más pausado") → 1.18. Segunda pasada (4-oct: "habla un poco
# rápido, más pausado sería mejor"): medido, 1.18 daba **211 palabras/min** — por encima del
# español conversacional natural (150-190). 1.40 lo baja a **186 pal/min**, dentro de rango.
# Costo de latencia: despreciable — la síntesis es 0,16x el audio (RTF), y en streaming por
# oración el primer sonido no espera al resto. Con 1.40: +0,14 s de síntesis por frase.
# Escala medida sobre la misma frase (26 palabras): 1.18→7,41s (211/min) · 1.30→7,74s (201) ·
# 1.40→8,38s (186) · 1.50→8,84s (177) · 1.65→9,71s (161). Tavo probó 1.65 y eligió **1.50**
# (4-oct): el punto medio entre el rango natural y el pausado. Muestras: ~/.hermes/cache/scratch/ritmo/.
LENGTH_SCALE = 1.50

def para_voz(texto):
    """Limpia el texto antes de sintetizarlo: Piper LEE lo que le des.

    Marcado que el modelo mete por costumbre y suena mal en voz alta: `**negrita**` (Piper
    pronuncia el asterisco o hace un chasquido), `#` de título, viñetas, backticks, y los
    guiones largos de aislante. El prompt de conversación ya pide texto plano — esto es el
    respaldo para cuando igual se cuela (medido: se colaba).
    """
    import re
    t = texto
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)      # **negrita**
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", t)   # *cursiva*
    t = re.sub(r"`([^`]*)`", r"\1", t)          # `code`
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t) # ## títulos
    t = re.sub(r"(?m)^\s*[-•*]\s+", "", t)      # viñetas
    t = re.sub(r"\s*—\s*", ", ", t)             # — → coma (pausa natural)
    return re.sub(r"\s+", " ", t).strip()

_voice = None
def hablar(texto, bloquear=True):
    global _voice
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    if _voice is None:
        _voice = PiperVoice.load(PIPER_MODEL)
        list(_voice.synthesize("listo"))
    syn = SynthesisConfig(length_scale=LENGTH_SCALE)
    # frase corta primero: streaming por oración → primer sonido ~90 ms
    out = VOZ / "voz.wav"
    t0 = time.time(); first = None
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(_voice.config.sample_rate)
        for c in _voice.synthesize(para_voz(texto), syn_config=syn):
            if first is None: first = time.time() - t0
            wf.writeframes(c.audio_int16_bytes)
    p = _registrar(subprocess.Popen(["mpv", "--no-terminal", "--really-quiet",
                          str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    if bloquear:
        p.wait()
    return first * 1000 if first else None

# ── decisión (portero) ──
# PORTERO = YO MISMO SIN PENSAMIENTO (idea de Tavo, 3-oct-2026, medida):
#   deepseek-v4.1-flash (ollama.com) con reasoning_effort=none →
#   15/15 · mediana 791 ms · separación limpia (trivial 0.05-0.10 / real 0.90-1.00).
#   Gana a gpt-oss:20b (14/15 · 1430 ms) y a nimble local (10/15 · no cabe en VRAM).
#   Cuando la frase pasa a Jennas, el pensamiento se ENCIENDE (turno normal).
#   Ver: ~/.hermes/scripts/jarvis-latencia/portero_mismo_modelo.py
UMBRAL_PORTERO = 0.5
MODELO_PORTERO = os.environ.get("JARVIS_MODELO_PORTERO", "deepseek-v4.1-flash")

def _ollama_key():
    # 1º: variable de entorno (config.example). 2º: archivo local del perfil.
    k = os.environ.get("JARVIS_OLLAMA_KEY")
    if k:
        return k
    for line in (HOME / ".hermes/.env").read_text().splitlines():
        if line.startswith("OLLAMA_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

_INSTR_PORTERO = ("Eres el portero de un asistente de voz. Decide si la frase del usuario "
    "requiere al asistente real (memoria, el vault, datos, criterio) o si es trivial "
    "(saludo, cortesía, saber la hora). Responde SOLO JSON: {\"necesita\": <0..1>} "
    "con tu probabilidad de que REQUIERA al asistente. Nada más.")
_KEY = _ollama_key()

def decidir(texto, timeout=30):
    """¿Necesita a Jennas? Devuelve (prob, tipo, conf). tipo/conf son de apoyo
    (para el léxico de saludo/hora), no la señal principal."""
    body = {"model": MODELO_PORTERO,
            "messages": [{"role": "system", "content": _INSTR_PORTERO},
                         {"role": "user", "content": texto}],
            "stream": False, "temperature": 0,
            "reasoning_effort": "none",          # el portero no piensa: solo enruta
            "response_format": {"type": "json_object"}}
    try:
        req = urllib.request.Request("https://ollama.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {_KEY}"})
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        txt = r["choices"][0]["message"]["content"]
        m = re.search(r'"necesita"\s*:\s*([0-9.]+)', txt)
        prob = float(m.group(1)) if m else 1.0
        return prob, "?", prob
    except Exception:
        return 1.0, "?", 0.0  # ante la duda o sin red: que responda Jennas

# ── respuesta de Jennas (yo) ──
# MODO VOZ: respuestas cortas. Un párrafo de enciclopedia se siente como que le
# estoy leyendo un libro; 1-3 frases habladas se sienten como conversación.
PROMPT_VOZ = ("[Modo voz: Tavo te está hablando por micrófono y tu respuesta se lee "
              "en voz alta. Dale SOLO la conclusión, en 1 o 2 frases cortas y habladas. "
              "NO narres el trabajo intermedio: nada de 'déjame ver', 'procedo', 'ahora "
              "reviso', 'ya lo tengo', ni anunciar qué archivo mueves o qué paso sigue. "
              "NO le cuentes lo que hiciste, revisaste ni encontraste por el camino: nada de "
              "'revisé', 'chequeé', 'está corriendo en el puerto X', 'respondía 200', ni listas "
              "de servicios, puertos, métricas, rutas o pasos. Si la tarea se demora, el "
              "silencio lo llena la app con su propia voz — tú entregas SOLO el resultado "
              "final, una vez, cuando ya está hecho. "
              "Solo lo que necesita saber, dicho como se lo diría alguien de confianza al "
              "pasar. Nada de listas, títulos ni markdown. Si falta contexto, pregunta "
              "en una frase.] ")

# MODO CONVERSACIÓN (4-oct-2026, pedido de Tavo): "respuestas más largas y fundamentadas,
# típica de discusión filosófica o narrativa". Se activa con un botón en la tablet y es
# POR TURNO — no cambia el default, que sigue siendo corto.
#
# Tres cosas cambian juntas (ninguna sola alcanza):
#   1. El largo: de 1-2 frases a un desarrollo fundamentado.
#   2. El razonamiento: `none` → `low`. Acá SÍ conviene pensar — pedir fundamento con el
#      pensamiento apagado da opinión sin respaldo. Es el trade-off de latencia, a propósito.
#   3. La voz: el texto hablado se lee IGUAL (no hay superficie de lectura), pero el audio
#      largo se reproduce por oración (streaming) mientras se sigue generando.
#
# El modo "solo el desarrollo, sin la muletilla de relleno" está en PROMPT_CONVERSACION:
# sin la regla de "no repitas el pedido" el modelo arranca con "Claro, es interesante…".
# ── expresividad (ElevenLabs/Valeria, 4-oct-2026) ──
# El motor INTERPRETA las indicaciones entre corchetes en vez de leerlas. Aplica a TODO
# Jarvis (corto, conversación y thinking) desde que Valeria entró en los tres carriles.
#
# DOSIS — lo medido el 4-oct (A/B sobre 6 entradas con carga emocional, sesión fresca):
#   · prompt de "moderación" genérica ....... 0,83 tags/respuesta
#   · prompt de DISPARADORES concretos ...... 1,00 tags/respuesta  (+17%)
# El modelo no adivina CUÁNDO corresponde: si se le dice el disparador, los usa. La regla
# vieja ("una o dos por respuesta, y solo si aportan") producía ~0,22 en producción real,
# porque las consultas mundanas (la hora, un dato) no disparan emoción sola.
# Muestras de los dos prompts: ~/.hermes/cache/scratch/ab-expresiones.py
PROMPT_TAGS = (
    "Tu voz actúa las indicaciones entre corchetes: las escribes y las INTERPRETA "
    "(no las lee en voz alta). Abre casi todas tus respuestas con el que corresponda:\n"
    "- algo te da risa o es absurdo → [laughs]\n"
    "- algo es irónico, o ya te lo pidió antes → [sarcastic]\n"
    "- algo te sorprende → [surprised]\n"
    "- algo te da curiosidad o es ambiguo → [curious]\n"
    "- algo te cansa o es monótono → [sighs] o [tired]\n"
    "- algo te gustó → [amused] o [happy]\n"
    "- vas a decir algo sin vueltas → [serious]\n"
    "- bajas la voz para algo íntimo o confidencial → [whispers]\n"
    "- algo te entusiasma → [excited]\n"
    "- estás jugando, o el tono es liviano y divertido → [playful]\n"
    "- te pones a pensar algo con calma, sin apuro → [thoughtful]\n"
    "En respuestas de una sola frase, igual pon UNO si el tono lo pide. No son adorno: "
    "son el tono. Para una pausa, tres puntos: …\n\n")

PROMPT_CONVERSACION = (PROMPT_TAGS +
                       "[Modo conversación: Tavo quiere pensar CON vos — discusión filosófica, "
                       "narrativa o de ideas. Desarrolla tu posición con fundamento: por qué "
                       "sostienes lo que sostienes, con qué se conecta. "
                       "Largo: 2 o 3 párrafos, no más — esto se ESCUCHA, no se lee, y un "
                       "monólogo largo se vuelve pared de sonido. Si el tema pide más, "
                       "cierra con la arista que falta y deja que Tavo siga la hebra. "
                       "Sin relleno ni repetición: cada frase aporta. "
                       "NO narres el trabajo intermedio: nada de 'déjame ver', 'procedo', "
                       "'ahora reviso', ni anunciar qué archivo mueves o qué paso sigue. Si la "
                       "tarea se demora, el silencio lo llena la app con su propia voz — tú "
                       "entregas el resultado, no el proceso. "
                       "Hablas como una PERSONA, no como un log: OMITE el lenguaje técnico — "
                       "ni puertos, ni métricas (%, GB, temperaturas), ni nombres de servicios "
                       "o archivos, ni códigos de estado, ni rutas, ni el relato de lo que "
                       "revisaste. Si un dato técnico importa, di qué significa para Tavo, "
                       "no el número crudo. Ejemplo de lo que NO quieres: 'Todo verde: disco "
                       "47%, carga 0.59, 0 fallos, RTX 3050 a 35°C'. "
                       "Entra directo al contenido, sin muletillas de apertura. "
                       "Texto plano para ser leído en voz alta: nada de markdown.")

# Modo voz = SIN pensamiento (palanca de latencia, medida 4-oct-2026): el modelo
# razonaba 1343 chars para un "di ok" (2,6s de LLM). Con `none` baja a 1,6s y la
# salida 4,5×. Verificado que NO pierde precisión (contó bien los días a una fecha).
# Override por entorno para volver atrás sin tocar código.
REASONING_VOZ = os.environ.get("JARVIS_REASONING_VOZ", "none")

# Modo conversación = SÍ piensa. Pedir fundamento con el pensamiento apagado da opinión sin
# respaldo — acá el razonamiento es parte de la respuesta, no overhead. Es ~1 s más de LLM,
# y en modo conversación eso se acepta a cambio de profundidad. Override por entorno.
REASONING_CONV = os.environ.get("JARVIS_REASONING_CONV", "low")

# Herramientas del modo voz (palanca de latencia, medida 4-oct-2026): montar los ~20
# toolsets completos (~80 tools) costaba ~1,2s POR TURNO. El carril de voz solo necesita
# leer/escribir archivos (vault, pendientes, memoria), buscar en la web y llamar la API
# del calendario (terminal). Verificado que con estas tres sigue respondiendo pendientes,
# vault y web igual. Override por entorno.
TOOLSETS_VOZ = os.environ.get("JARVIS_TOOLSETS_VOZ", "file,web,terminal")

# UNA sesión por conversación de voz (pedido de Tavo, 3-oct: "sería mejor consolidar
# en una sola con nombre sesión audio y fecha hora"). El nombre se fija al arrancar
# el bucle: todos los turnos de este diálogo van a la misma sesión, y Jennas recuerda
# lo dicho antes. Al reiniciar el bucle nace una sesión nueva (hora nueva).
def _nombre_sesion():
    return "audio-" + time.strftime("%Y-%m-%d-%H%M")

def preguntar_jennas(texto, sesion, timeout=300, conversacion=False, tags=False):
    prompt = PROMPT_CONVERSACION if conversacion else PROMPT_VOZ
    # Los tags solo los entiende el motor de ElevenLabs. Jarvis ahora lo usa en TODO, pero
    # la TV sigue con Piper y Piper los LEERÍA en voz alta ("corchete sighs corchete") →
    # por eso van como parámetro y no como parte del PROMPT_VOZ compartido.
    if tags and not conversacion:
        prompt = PROMPT_TAGS + prompt
    razon = REASONING_CONV if conversacion else REASONING_VOZ
    try:
        r = subprocess.run(["hermes", "chat", "-q", prompt + texto,
                            "--continue", sesion, "--create-if-missing", "--quiet",
                            "--reasoning", razon, "-t", TOOLSETS_VOZ],
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "").strip()
        # quitar metadatos de sesión
        lineas = [l for l in out.split("\n")
                  if not l.startswith("session_id:") and not l.startswith("↻ ")]
        return " ".join(" ".join(lineas).split())
    except subprocess.TimeoutExpired:
        return "Me tomó demasiado tiempo responder. Probá de nuevo."
    except Exception as e:
        return f"No pude responder: {str(e)[:80]}"


# Verbos de ACCIÓN que, precedidos por "dejame/permíteme" o por "ahora", marcan una promesa
# en 1a persona. Deliberadamente amplia pero anclada a la construcción: se busca
# "<marcador> + verbo", no la palabra suelta, para no disparar con el uso conversacional
# ("ahora entiendo", "ya sé"). Validado contra 275 respuestas reales de la tablet (7-oct-2026).
_VERBOS_ACCION = (
    r"revis\w+|cheque\w+|verific\w+|mir\w+|busc\w+|abr\w+|le[eo]\w+|"
    r"consult\w+|actualiz\w+|cre[oé]\w*|gener\w+|ejecut\w+|corr\w+|"
    r"compar\w+|investig\w+|recolect\w+|prepar\w+|arm\w+|escrib\w+|"
    r"anot\w+|guard\w+|sac\w+|list\w+|calcul\w+|corrij\w+|cier[rw]\w*|"
    r"carg\w+|limpi\w+|compil\w+|instal\w+"
)


def _parece_promesa(texto):
    """¿El modelo cerró anunciando una acción que NO ejecutó?

    Bug reportado por Tavo (5-oct-2026, dos veces): contesta *"Sí, procede"* y el modelo cierra
    con el plan + *"Voy."* / *"Procedo."* / *"Ya lo hago."* **sin llamar ninguna herramienta**
    (`finish_reason: stop`) — hay que repetir "procede". Pasa en sesiones largas, con el
    contexto cargado.

    Señal: el texto TERMINA con una promesa de acción en primera persona. Se deja conservador
    a propósito (solo cierres inequívocos) para no reintentar un turno conversacional normal.

    GUARDA CRÍTICA: si cierra con `?` es una PREGUNTA pidiendo permiso ("¿Lo hago?"), no una
    promesa — reintentar ahí haría que Jarvis actúe SIN autorización de Tavo."""
    limpio = texto.strip()
    if not limpio:
        return False
    # Pregunta = pide permiso. Nunca es promesa: no se reintenta.
    if limpio.rstrip("\"')» ").endswith(("?", "¿")):
        return False
    cola = limpio[-90:].lower()
    patrones = (
        r"\b(voy|procedo|dale|ahora|ya)\s*[.!]?$",           # "... Voy." / "... Procedo."
        r"\b(lo|la|los|las)\s+(hago|muevo|creo|borro|actualizo)\b[^.]*[.!]?\s*$",
        r"\b(ya lo|ya la|ahora lo|ahora la)\s+\w+\s*[.!]?$",
        r"\bprocedo a\b[^.]*[.!]?$",
        r"\bvoy a\s+\w+[^.]*[.!]?$",
    )
    if any(re.search(p, cola) for p in patrones):
        return True

    # ── Promesas en 1a persona que el patrón de cierre NO veía (7-oct-2026) ──
    # Bug real: Tavo quedó colgado con *"Dejame revisar el registro de ayer, porque no quiero
    # recitarte de memoria."* — la acción se anunciaba al PRINCIPIO, no en la cola, y el verbo
    # ("revisar") no estaba en la lista. Medido sobre 275 respuestas reales de la tablet:
    # 7 cerraban así y el detector viejo agarraba 0.
    # La ventana sube a 160 chars (la promesa no siempre cierra la frase) y se busca
    # "<marcador> + verbo de acción" y "ahora + verbo", no coincidencias de palabras sueltas.
    cola_ancha = limpio[-160:].lower()
    if re.search(rf"\b(d[eé]jame|perm[ií]time)\s+({_VERBOS_ACCION})", cola_ancha):
        return True
    if re.search(rf"\bahora\s+({_VERBOS_ACCION})", cola_ancha):
        return True
    if re.search(rf"\b({_VERBOS_ACCION})\s+(ahora|ya|enseguida)\b", cola_ancha):
        return True
    return False


def preguntar_jennas_stream(texto, sesion, timeout=300, conversacion=False, tags=False):
    """Igual que `preguntar_jennas`, pero emite eventos a medida que llegan (stream-json).

    Devuelve dicts:
      {'tipo': 'tool',  'fase': 'inicio'|'fin', 'nombre': <tool>}
      {'tipo': 'texto', 'texto': <oración completa>}

    Por qué los eventos de herramienta (4-oct-2026, reportado por Tavo): mientras Jennas
    trabajaba (tool calls) el sprite quedaba en `talking` aunque NO sonaba nada — se veía
    "hablando en silencio". Con los eventos `tool_use`/`tool_result` del protocolo, el
    consumidor puede pasar a `thinking` durante el trabajo y volver a `talking` al hablar.

    El protocolo es ``system/init`` → ``text`` deltas / ``tool_use`` / ``tool_result`` →
    ``result`` terminal (ver hermes_cli/stream_json.py).

    REINTENTO POR PROMESA (5-oct-2026): si el turno termina sin usar UNA sola herramienta y el
    texto cierra con una promesa de acción ("... Voy."), se le pide UNA vez que lo ejecute ahora.
    El prólogo de la primera pasada lo descarta el consumidor (responder_stream retiene el texto
    anterior al primer tool)."""
    import json as _json
    import threading
    prompt = PROMPT_CONVERSACION if conversacion else PROMPT_VOZ
    if tags and not conversacion:
        prompt = PROMPT_TAGS + prompt
    razon = REASONING_CONV if conversacion else REASONING_VOZ

    # Hasta 2 pasadas: la normal, y una de "ejecuta ahora lo que dijiste" si quedó en promesa.
    for pasada in range(2):
        mensaje = texto if pasada == 0 else (
            "Ejecuta AHORA lo que dijiste que ibas a hacer. No lo anuncies ni lo describas: "
            "hazlo con las herramientas y después cuéntame el resultado.")
        proc = subprocess.Popen(
            ["hermes", "chat", "-q", prompt + mensaje,
             "--continue", sesion,
             "--create-if-missing", "--format", "stream-json",
             "--reasoning", razon,
             "-t", TOOLSETS_VOZ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        buf = ""
        texto_total = ""
        hubo_tool = False
        # Watchdog de INACTIVIDAD: si pasan `timeout` segundos sin una línea nueva del
        # subproceso, quedó colgado → se lo mata y el turno cierra limpio. Antes un cuelgue
        # dejaba el turno abierto para siempre (tablet esperando, sprite en 'thinking').
        _wd = [threading.Timer(timeout, proc.kill)]
        _wd[0].daemon = True
        _wd[0].start()
        try:
            for line in (proc.stdout or []):
                _wd[0].cancel()
                _wd[0] = threading.Timer(timeout, proc.kill)
                _wd[0].daemon = True
                _wd[0].start()
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue                       # ignora la línea "Session ... found"
                try:
                    ev = _json.loads(line)
                except Exception:
                    continue
                tipo = ev.get("type")
                if tipo == "tool_use":
                    hubo_tool = True
                    yield {"tipo": "tool", "fase": "inicio", "nombre": ev.get("name", "?")}
                    continue
                if tipo == "tool_result":
                    yield {"tipo": "tool", "fase": "fin", "nombre": ev.get("name", "?")}
                    continue
                if tipo != "text":
                    continue
                texto_total += ev.get("text", "")
                buf += ev.get("text", "")
                # corta en oraciones completas (., !, ?, …) respetando abreviaturas simples
                while True:
                    m = re.search(r'([.!?…]+)(\s|$)', buf)
                    if not m:
                        break
                    fin = m.end()
                    oracion = buf[:fin].strip()
                    buf = buf[fin:]
                    if oracion:
                        yield {"tipo": "texto", "texto": oracion}
        finally:
            try:
                _wd[0].cancel()          # apaga el watchdog: el turno terminó (bien o por kill)
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass
        if buf.strip():
            yield {"tipo": "texto", "texto": buf.strip()}

        # ¿Cerró con una promesa sin haber hecho nada? → una segunda pasada, y corta.
        if hubo_tool or pasada == 1 or not _parece_promesa(texto_total):
            break
        print("[jennas-voz] el turno cerró en promesa sin ejecutar → reintento que lo haga",
              flush=True)

# ── bucle principal ──
def main():
    _instalar_cierre_limpio()          # al salir (señal o normal), suelta la captura de audio
    print("[jennas-voz] calentando modelos…", flush=True)
    from piper import PiperVoice
    global _voice, _whisper
    _voice = PiperVoice.load(PIPER_MODEL); list(_voice.synthesize("listo"))
    from faster_whisper import WhisperModel
    _whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    _whisper.transcribe(_decodificar(VOZ / "voz.wav"), language="es")  # warm-up
    print("[jennas-voz] listo. Habla.", flush=True)
    pon_estado("talking")
    hablar("Listo. Te escucho.")
    pon_estado("listening")
    sesion = _nombre_sesion()          # una sesión por diálogo (nombre al arrancar)
    print(f"[jennas-voz] sesión de voz: {sesion}", flush=True)

    while True:
        wav = VOZ / "entrada.wav"
        print("[jennas-voz] escuchando…", flush=True)
        pon_estado("listening")            # ← la TV muestra "escuchando"
        hablado = grabar(wav)
        if not hablado:
            continue
        pon_estado("thinking")             # ← transcribiendo / decidiendo
        t0 = time.time()
        texto = transcribir(wav)
        if not texto:
            pon_estado("listening")
            continue
        publicar({"rol": "tavo", "texto": texto, "t": time.strftime("%H:%M:%S")})
        print(f"[Tavo] {texto}", flush=True)

        if any(p in texto.lower() for p in ["apagate", "chao", "adiós", "adios", "hasta luego", "corta"]):
            pon_estado("talking")
            publicar({"rol": "jennas", "texto": "Nos vemos.", "t": time.strftime("%H:%M:%S")})
            hablar("Nos vemos, Tavo.")
            break

        necesita, tipo, conf = decidir(texto)
        # Handler local SOLO si el portero dice que NO me necesita Y el texto es
        # léxicamente un saludo/hora. El portero trae separación limpia (0.00-0.10
        # trivial / 1.00 real), así que el umbral es franco; el léxico evita
        # improvisar en casos ambiguos.
        t = texto.lower().strip().strip("¿?¡! ")
        saludo_lex = (len(t.split()) <= 4 and any(t.startswith(p) for p in
                      ("hola", "buenos días", "buenas", "aló", "alo", "hey", "qué tal",
                       "que tal", "me escuchas", "me oyes", "estás ahí", "estas ahi",
                       "sigues ahí", "sigues ahi", "estás ahí?", "gracias")))
        hora_lex = t.strip("¿?¡! .") in ("qué hora es", "que hora es", "dime la hora",
                                          "la hora", "qué hora tienes", "que hora tienes")
        if necesita < UMBRAL_PORTERO and (saludo_lex or hora_lex):
            resp = "Son las " + time.strftime("%H:%M") + "." if hora_lex else "Hola Tavo."
            pon_estado("talking")
            publicar({"rol": "jennas", "texto": resp, "t": time.strftime("%H:%M:%S")})
            hablar(resp)
            pon_estado("listening")
            continue

        # todo lo demás → Jennas
        print(f"[routing] Jennas (necesita={necesita:.2f} tipo={tipo} conf={conf:.2f}) → pensando…", flush=True)
        publicar({"rol": "jennas", "texto": "…", "t": time.strftime("%H:%M:%S")})
        # EL CASO QUE PIDIÓ TAVO: mientras Jennas piensa, la TV muestra la foto pensativa
        pon_estado("thinking")
        threading.Thread(target=hablar, args=("Déjame pensarlo.",), daemon=True).start()
        resp = preguntar_jennas(texto, sesion)
        publicar({"rol": "jennas", "texto": resp, "t": time.strftime("%H:%M:%S")})
        print(f"[Jennas] {resp}  ({(time.time()-t0):.1f}s)", flush=True)
        pon_estado("talking")              # ← ahora habla
        hablar(resp)
        pon_estado("listening")            # ← vuelve a escuchar

if __name__ == "__main__":
    main()

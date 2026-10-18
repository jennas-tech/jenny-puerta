#!/usr/bin/env python3
"""
Jenny-Puerta — la estación de voz de la casa, EN UNA APP (17-oct-2026).

Sucesora del jarvis-jennas deprecado (Tavo, 17-oct 20:25: «jenny depreca el
proyecto jarvis y crea uno nuevo, proyecto jenny puerta y consolida todo en
una sola app. esto de tener una app con 2 cabezas es muy desordenado»).

UNA CABEZA = un proceso con todo:
  1. el wake word (openwakeword, jenny_custom_v0.1, 100% local)
  2. el STT directo (faster-whisper en este proceso — sin HTTP interno)
  3. el pipeline (jennas-voz.py como módulo: portero → Jennas → TTS)
  4. el HTTP único (:8096 TLS): la estación y la app tablet SAME-ORIGIN —
     se mueren de paso el vigía, el landzer, el SOP y los dos services.

Los puertos que esto consolida: :8096 (estación) + :8094/:8095 (JJ) → 8096 solo.

Puerto :8096 (HTTPS):
  GET  /            → la estación (client/estacion.html)
  GET  /tablet      → la app tablet (client/tablet.html)
  GET  /sprites/    → los sprites (compartidos entre las dos)
  GET  /beep.wav    → el doble-bip de «te tomé, puede hablar»
  GET  /voz/<f>     → el audio sintetizado (fijos + respuestas)
  GET  /state.json  → el estado del pipeline (lo escriben los turnos)
  GET  /estado      → el estado consolidado del wake para la página
  POST /audio       → chunks del mic (PCM16 16 kHz)
  POST /stop        → pausa/reanuda la escucha
  POST /preguntar   → turno completo ({texto, hablar, conversacion, stream})

Variables de entorno (env-wake): MODELO, UMBRAL (0.9), GRABAR_S (12.0),
COOLDOWN_S (4.0), FIN_SIL_MS (2500), TOPE_DURO_S (150.0), MODO_EXPLAYA (off).

REGLAS del corte (la cadena del 17-oct, cada una nacida de un corte con voz
en mano): el tope JAMÁS decide solo — solo cierra lo que el fin-de-habla
cerraría; y ring(BUF_MAX) >= tope-duro + pre, SIEMPRE. El único que cierra
un turno es el silencio del usuario.
"""
import importlib.util
import io
import json
import os
import re
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
PUERTO = int(os.environ.get("JENNY_PUERTA_PORT", "8096"))
CERT = AQUI / "certs" / "jarvis.pem"
KEY = AQUI / "certs" / "jarvis-key.pem"
SR = 16000
CHUNK = 1280
BUF_MAX = 2000        # el ring ≈ 160 s — regla: ring >= tope-duro + pre SIEMPRE
UMBRAL = float(os.environ.get("JARVIS_WAKE_UMBRAL", "0.9"))
GRABAR_S = float(os.environ.get("JARVIS_WAKE_GRABAR", "12.0"))
COOLDOWN_S = float(os.environ.get("JARVIS_WAKE_COOLDOWN", "4.0"))
MODELO = os.environ.get("JARVIS_WAKE_MODELO", "wake-models/jenny_custom_v0.1.onnx")
PRE_CHUNKS = 30       # ~2.4 s pre-despertar
FIN_SIL_MS = int(os.environ.get("JARVIS_WAKE_FIN_SIL_MS", "2500"))
TOPE_DURO_S = float(os.environ.get("JARVIS_WAKE_TOPE_DURO_S", "150.0"))
MODO_EXPLAYA_ON = os.environ.get("JARVIS_WAKE_MODO_EXPLAYA", "on") != "off"
WHISPER_TABLET = os.environ.get("JARVIS_TABLET_WHISPER", "large-v3-turbo")

# ── el bucle jv: el CEREBRO (portero → Jennas → TTS) — reuso, no copia ──
def _cargar_bucle():
    ruta = AQUI / "jennas-voz.py"
    if not ruta.exists():
        raise SystemExit(f"[jenny] no encuentro el bucle en {ruta}")
    spec = importlib.util.spec_from_file_location("jennas_voz", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jennas_voz"] = mod
    spec.loader.exec_module(mod)
    return mod


jv = _cargar_bucle()
jv._instalar_cierre_limpio()
SESION = "tablet-" + time.strftime("%Y-%m-%d-%H%M")
_SESION_CANDADO = threading.Lock()

_TTS = threading.Lock()
_EST = threading.Lock()
VOZ_DIR = AQUI / "runtime"        # audio sintetizado (los fijos + respuestas)
VOZ_DIR.mkdir(exist_ok=True)

# ── el STT: NO re-implementado — ES EL DEL CEREBRO (jv.transcribir) ──
# El bucle (jennas-voz.py:185) ya trae: decodificación, anti-retumbo, normalización,
# vocabulario de Tavo, Whisper local float16 — y el modelo de la tablet lo carga el
# JJ (_asegurar_stt: large-v3-turbo, cuda/float16, lazy con el primer turno).
# Aquí solo la CARGA del modelo de la tablet, y el turno llama jv.transcribir.
_stt_modelo = None
_STT_LOCK = threading.Lock()


def _asegurar_stt():
    """El mismo modelo del JJ (large-v3-turbo, cuda, float16) — lazy."""
    global _stt_modelo
    with _STT_LOCK:
        if _stt_modelo is None:
            from faster_whisper import WhisperModel
            t0 = time.time()
            _stt_modelo = WhisperModel(WHISPER_TABLET, device="cuda", compute_type="float16")
            log(f"STT {WHISPER_TABLET} cargado en {time.time()-t0:.1f}s")
    return _stt_modelo


def transcribir(wav_bytes: bytes) -> str:
    """STT con el CEREBRO: disco temporal + jv.transcribir (el mismo camino que
    la tablet ya probó). El error 'metadata_errors' del transcribir directo con
    BytesIO lo mató aquí; el camino del cerebro con archivo es el que funciona."""
    import tempfile
    ruta = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=VOZ_DIR) as f:
            f.write(wav_bytes)
            ruta = Path(f.name)
        texto = jv.transcribir(str(ruta), modelo=_asegurar_stt())
        ruta.unlink(missing_ok=True)
        return texto
    except Exception as e:
        if ruta:
            try:
                ruta.unlink(missing_ok=True)
            except Exception:
                pass
        log("transcribe falló:", str(e)[:200])
    return ""


# ── estado compartido del wake ──
lock = threading.Lock()
estado = {"v": "dormido", "score": 0.0, "ultimo": None, "ultimo_t": 0.0, "gracia_t": 0.0}
pausado = {"v": False}
buf: deque[tuple[int, np.ndarray]] = deque(maxlen=BUF_MAX)
sec_global = {"v": 0}
last_procesado = {"v": 0}
NOMBRE_RE = re.compile(
    r"^\s*(oye\s+|hey\s+|hola\s+|eh\s+|perd[oó]n[aá]*\s+)*(jenny|jeni|jena|keny|keni|kenny|jarvis|"
    r"jervis|gervis|hervis|arvis|ravis)\s*[,;:!.]?\s*", re.IGNORECASE)
# el comando POR VOZ de rotación (la frase completa — sin falsos positivos en
# medio de una charla; el STT con/sin tildes se normaliza antes del match.
# «noda»/«nuea» = las formas reales del STT cuando la voz sale raspada)
SESION_NUEVA_RE = re.compile(
    r"^(?:(?:haz(?:me)?|dame|abre|empieza|comienza|vamos\s+a)\s+)?"
    r"(?:una\s+|otra\s+|la\s+|(?:[a-zñ]{1,4}\s+){1,2})?"
    r"(?:nueva\s+)?(?:sesion|conversacion)"
    r"(?:(?:\s+(?:nueva|noda|nuea|\w{1,3})){0,2})?$"
    r"|^empe[a-z]{0,4}mos\s+de\s+cero$|^desde\s+cero$")


def log(*a):
    print(f"[jenny] {time.strftime('%H:%M:%S')}", *a, flush=True)


# ── las SESIONES (la herencia real de la tablet app, 6-oct: pedido de Tavo) ──
def _sesion():
    """El título de sesión VIGENTE (mutable: `/sesion-nueva` lo rota — otro thread)."""
    with _SESION_CANDADO:
        return SESION


def _nombre_sesion_libre(base):
    """Devuelve `base` si no está tomado en state.db; si no `base-2`, `base-3`…
    El título de sesión es UNIQUE en state.db (idx_sessions_title_unique): reasignar
    un nombre usado hace fallar el INSERT de Hermes. Si la DB no se puede leer, el
    candidato sin chequear: el peor caso es el comportamiento de antes."""
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{Path.home()}/.hermes/state.db?mode=ro", uri=True)
        try:
            for i in range(1, 21):
                cand = base if i == 1 else f"{base}-{i}"
                fila = con.execute("SELECT 1 FROM sessions WHERE title = ? LIMIT 1", (cand,)).fetchone()
                if not fila:
                    return cand
        finally:
            con.close()
    except Exception:
        pass
    return f"{base}-{int(time.time())}"   # último recurso: siempre único


def _sesion_nueva():
    """Rotación de sesión (el botón 🗒️ de la tablet, ahora en la casa entera):
    - Rota el título VIGENTE → el próximo turno parte el hilo limpio (el historial
      de Hermes queda en el título viejo, recuperable — NO se borra nada).
    - LIMPIA el historial en PANTALLA (turnos=[] en state.json) — la pantalla no
      puede seguir mostrando la frase de la charla vieja al lado de un sprite en
      reposo. Solo turnos: el estado lo maneja el turn-cycle."""
    global SESION
    vieja = _sesion()
    with _SESION_CANDADO:
        # la base AL SEGUNDO: una rotación en el mismo minuto que el arranque
        # no puede volver a parir el mismo título (22:19 lo delató: 2219 → 2219)
        nueva = _nombre_sesion_libre("tablet-" + time.strftime("%Y-%m-%d-%H%M%S"))
        SESION = nueva
    with _EST:
        try:
            st = json.loads(jv.STATE.read_text()) if jv.STATE.exists() else {}
        except Exception:
            st = {}
        st["turnos"] = []
        st["actualizado"] = time.strftime("%H:%M:%S")
        jv.STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    with lock:                       # la respuesta de la charla vieja no puede
        estado["ultimo"] = None      # re-sonar con el sprite ya limpito —
        estado["score"] = 0.0        # ni el replay 🔊 tiene de qué volver
    log(f"SESIÓN NUEVA: {nueva} (la vieja sigue en el historial: {vieja})")
    return {"ok": True, "sesion": nueva, "anterior": vieja}


def limpiar_prefijo(texto: str) -> str:
    m = NOMBRE_RE.match(texto)
    if m:
        return texto[m.end():].strip()
    m2 = re.search(r"\b(jenny|jeni|jena|keny|keni|kenny|jarvis|harvis)\b", texto, re.IGNORECASE)
    if m2:
        cola = texto[m2.end():].strip(" ,;:.")
        if len(cola) > 8:
            return cola
    return texto.strip()


# ── el detector (openwakeword) — CORRE EN CPU (sin VRAM) ──
from openwakeword.model import Model  # noqa: E402

modelo_oww = Model(wakeword_model_paths=[str(AQUI / MODELO)])
print(f"[jenny] modelo: {MODELO} · umbral {UMBRAL} · grabación {GRABAR_S}s · "
      f"tope-duro {TOPE_DURO_S}s · ring {BUF_MAX} chunks", flush=True)


def hilo_detector():
    while True:
        time.sleep(0.04)
        if pausado["v"]:
            continue
        with lock:
            if not buf:
                continue
            items = [(s, d) for (s, d) in list(buf) if s > last_procesado["v"]]
        if not items:
            continue
        despertar_sec = None
        for (sec, chunk) in items:
            try:
                if len(chunk) < 400:                    # chunk degenerado (<25 ms @16k):
                    last_procesado["v"] = sec           # oww lo rechaza gritando (22:13)
                    continue
                scores = modelo_oww.predict(chunk)
            except Exception as e:
                log("predict falló:", str(e)[:140])
                last_procesado["v"] = sec
                continue
            v = max(float(s) for s in scores.values()) if isinstance(scores, dict) else 0.0
            with lock:
                estado["score"] = v
            last_procesado["v"] = sec
            if v >= UMBRAL and estado["v"] == "dormido":
                log(f"¡WAKE! sec={sec} score={v:.3f}")
                despertar_sec = sec
                break
        if despertar_sec is not None:
            hilo_flujo(despertar_sec)


def hilo_flujo(sec_despierto: int):
    """Post-despertar: grabar con el fin-de-habla dinámico, transcribir,
    preguntar a Jennas, sonar. Las REGLAS del 17-oct van TODAS acá."""
    cooldown_hasta = time.time() + COOLDOWN_S + GRABAR_S

    def run():
        with lock:
            estado["v"] = "listening"
        sil_ms = 0
        hubiere_voz = False
        tope = time.time() + min(GRABAR_S, 30.0)
        tope_duro = time.time() + TOPE_DURO_S
        tope_estirado = False
        modo_corte = FIN_SIL_MS
        modo = "normal"

        def _voz_ms_ring():
            """La voz REAL dentro de la ventana del clip (sec >= despertar-pre).
            El ring entero es un metro AMPLIO (contamine con los turnos anteriores
            y el ambiente — el test del 19:11 lo delató)."""
            frontera = sec_despierto - PRE_CHUNKS
            with lock:
                items = sorted(list(buf))
            return sum(80 for s, c in items if s >= frontera
                       and (float(np.sqrt(np.mean(np.frombuffer(c, dtype=np.int16)
                                                  .astype(np.float32) ** 2))) / 32768.0) > 0.02)

        while True:
            if time.time() >= tope:
                # EL TOPE JAMÁS DECIDE SOLO: solo cierra lo que el
                # fin-de-habla también cerraría — si sil_ms < modo_corte
                # es ESPERA, no calma → se estira (hasta el tope-duro).
                if sil_ms < modo_corte and time.time() < tope_duro:
                    tope = min(time.time() + 4.0, tope_duro)
                    if not tope_estirado:
                        tope_estirado = True
                        log(f"la voz sigue o es pausa de pensar — el tope se estira (sil_ms={sil_ms})")
                    continue
                break
            time.sleep(0.08)
            with lock:
                items = sorted(list(buf))
            if not items:
                continue
            c = np.frombuffer(items[-1][1], dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(c * c))) / 32768.0
            if rms > 0.02:
                hubiere_voz = True
                sil_ms = 0
            elif hubiere_voz:
                sil_ms += 80
                if sil_ms >= modo_corte:
                    if MODO_EXPLAYA_ON and modo == "normal" and _voz_ms_ring() >= int(os.environ.get("JARVIS_WAKE_VOZ_EXPLAYA_MS", "3000")):
                        # modo «explayándose» (inhabilitable por env — 19:52)
                        modo = "explayando"
                        modo_corte = int(os.environ.get("JARVIS_WAKE_GRACIA_MS", "9000"))
                        tope = max(tope, time.time() + modo_corte / 1000.0)
                        sil_ms = 0
                        with lock:
                            estado["gracia_t"] = time.time()
                        log(f"explayándose ({_voz_ms_ring()/1000:.1f}s de voz) — pausas hasta {modo_corte/1000:.0f}s")
                        continue
                    break
                    # sin voz todavía: NO acumula silencio — espera el tope completo

        with lock:
            items = sorted(list(buf))
        pre = [d for (s, d) in items if sec_despierto - PRE_CHUNKS <= s <= sec_despierto]
        post = [d for (s, d) in items if s > sec_despierto]
        parts = pre + post
        if not parts:
            log("sin audio en la ventana — duermo")
            with lock:
                estado["v"] = "dormido"
            return
        pcm = np.concatenate([np.frombuffer(c, dtype=np.int16) for c in parts])
        if len(pcm) / SR < 1.0:
            log(f"ventana corta ({len(pcm)/SR:.1f}s) — duermo")
            with lock:
                estado["v"] = "dormido"
            return
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.astype(np.int16).tobytes())
        texto = transcribir(bio.getvalue())
        if not texto:
            with lock:
                estado["v"] = "dormido"
            return
        texto_limpio = limpiar_prefijo(texto)
        if len(texto_limpio.split()) < 2:
            log(f"STT={texto!r} → solo el nombre — duermo sin turno")
            with lock:
                estado["v"] = "dormido"
            return
        # «Jenny, sesión nueva» POR VOZ (la casa es manos libres): rota y
        # confirma con la voz de la casa — SIN gastar el LLM (mismo principio
        # del llamado sin frase). Comando exacto: el match contra la FRASE
        # COMPLETA evita rotaciones accidentales en medio de una charla.
        tlow = re.sub(r"[^a-zñ0-9 ]", " ",
                      texto_limpio.lower().replace("á", "a").replace("é", "e")
                      .replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ü", "u"))
        tlow = re.sub(r"\s+", " ", tlow).strip()
        if SESION_NUEVA_RE.fullmatch(tlow):
            nueva = _sesion_nueva()
            frase = "Listo, tavito. Sesión nueva: empezamos de cero."
            audio = _tts(frase)
            with lock:
                estado["v"] = "talking"
                estado["ultimo"] = {"respuesta": frase, "audio": audio, "ms": 0}
                estado["ultimo_t"] = time.time()
            time.sleep(1.0)
            with lock:
                estado["v"] = "dormido"
            return
        log(f"pregunta limpia → {texto_limpio[:80]!r}")
        with lock:                                  # LA CASA PIENSA EN VOZ:
            estado["v"] = "thinking"                # el sprite + el «Déjame pensar» del cliente
        resp = _turno(texto_limpio)
        with lock:
            estado["v"] = "talking"
            estado["ultimo"] = resp
            estado["ultimo_t"] = time.time()
        time.sleep(1.0)
        with lock:
            estado["v"] = "dormido"

    threading.Thread(target=run, daemon=True).start()


# ── el turno: el pipeline del cerebro (el mismo jv, ahora directo) ──
def _turno(texto: str, conversacion: bool = True) -> dict:
    t0 = time.time()
    _publicar({"rol": "tavo", "texto": texto, "t": time.strftime("%H:%M:%S")})
    _pon_estado("thinking")
    audio = None
    try:
        resp = jv.preguntar_jennas(texto, _sesion(), conversacion=conversacion)
        resp_limpia = _solo_texto(resp)
        _publicar({"rol": "jennas", "texto": resp_limpia, "t": time.strftime("%H:%M:%S")})
        _pon_estado("talking")
        base = VOZ_DIR / f"tabla-{int(time.time() * 1000)}.wav"
        try:
            audio = _tts(_solo_texto(resp))
        except Exception as e:
            audio = None
            log("TTS falló:", str(e)[:100])
    except Exception as e:
        resp = "(se me atoró la garganta, intenta de nuevo)"
        log("turno explotó:", str(e)[:140])
    finally:
        _pon_estado("listening")
    return {"respuesta": resp if isinstance(resp, str) else str(resp), "audio": audio,
            "ms": int((time.time() - t0) * 1000)}


# el TTS: el misMOr Piper del cerebro (jv), sin el reproductor — el audio
# debe SERVIRSE por el <audio> de la página (el canal que ya sonaba en la tablet),
# no sonarlo localmente (así lo hacía jv.hablar y la pared o la tablet lo tapaba).
_tts_voice = None
_TTS_LOCK = threading.Lock()


# ── LOS FIJOS DE LA CASA (la fábrica del bicéfalo, port 1:1 — 17-oct) ──
# Tavo pidió «estoy pensando y frases similares mientras espero respuesta» —
# la frase de espera + los rellenos al turn largo. Sintetizados UNA vez al
# ARRANQUE con la voz de la casa (el turno no paga TTS por lo que se sabe de
# antemano). ATÓMICA: primero TODO en memoria, después a disco — un fallo a
# mitad no deja un lote mezclado (la lección del 8-9-oct del bicéfalo).
#
# EL MOTOR: SUPERTONIC 3 (el OFICIAL desde 10-oct: «supertonic son buenísimas,
# las piper son basura» — verbatim de Tavo en el env-tablet del bicéfalo).
# LA VOZ ACTIVA: «clara» (la entrenada local) por JARVIS_VOZ_SUPERTONIC —
# cambio en caliente con _cambiar_voz_local(). Piper queda como degradación
# limpia (si faltan modelos de Supertonic, el log lo dice y la voz sigue).
RELLENO = ["Sigo trabajando.", "Esto se está alargando un poco.",
           "Ya falta poco.", "Sigo en eso.", "Casi listo."]
FIJOS_MARKER = VOZ_DIR / ".fijos-voz"


def _cargar_supertonic():
    """El motor OFICIAL de la casa: Supertonic 3 (ONNX local 44 100 Hz, CPU —
    la VRAM queda para el Whisper del dictado). Degradación limpia: si el
    módulo o el modelo no están, devuelve None y `_motor_local()` cae a Piper
    avisando UNA vez (el bicéfalo ya había vivido esto)."""
    try:
        spec = importlib.util.spec_from_file_location("tts_supertonic", AQUI / "tts_supertonic.py")
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tts_supertonic"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[jenny] (motor Supertonic no disponible: {str(e)[:80]})", flush=True)
        return None


def _cargar_piper():
    """La degradación del bicéfalo: la voz que a Tavo le parece basura, pero
    es la ÚNICA garantía de que la casa no quede muda."""
    try:
        spec = importlib.util.spec_from_file_location("tts_piper", AQUI / "tts_piper.py")
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tts_piper"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[jenny] (Piper no disponible: {str(e)[:80]})", flush=True)
        return None


st = _cargar_supertonic()
pyp = _cargar_piper()
USAR_SUPERTONIC = os.environ.get("JARVIS_TTS", "supertonic") == "supertonic"
_ST_AVISADO = {"v": False}


def _motor_local():
    """El motor local ACTIVO con su interfaz común (sintetizar+normalizar+usar_voz):
    Supertonic si está seleccionado Y disponible; si no, Piper avisando una sola vez."""
    global _ST_AVISADO
    if USAR_SUPERTONIC and st is not None and st.disponible():
        return st
    if USAR_SUPERTONIC and not _ST_AVISADO["v"]:
        motivo = "el módulo no cargó" if st is None else f"faltan modelos {st.faltantes()}"
        print(f"[jenny] ⚠ Supertonic seleccionado pero no disponible ({motivo}) → Piper", flush=True)
        _ST_AVISADO["v"] = True
    return pyp


def _fijos_completos() -> bool:
    try:
        if not any(VOZ_DIR.glob("pensar.*")) and not any(VOZ_DIR.glob("pensar.mp3")):
            return False
        return all(any(VOZ_DIR.glob(f"relleno-{i}.*")) for i in range(len(RELLENO)))
    except Exception:
        return False


def _reempaqueta_wav(data: bytes) -> bytes:
    """El WAV SANO para el navegador. El `normalizar` de los motores pasa por ffmpeg
    pipe y el RIFF sale en modo streaming (riffSize=0xFFFFFFFF): el Python `wave` y
    el Chrome Android leen la duración FANTASMA (~13,5 h) de un archivo real de ~1,6 s.
    Extraigo el PCM real de los chunks y re-escribo un header chico y honesto."""
    import struct
    if data[:4] != b'RIFF':
        return data
    pcm = None
    rate, ch = 44100, 1
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos+4]
        try:
            sz = struct.unpack('<I', data[pos+4:pos+8])[0]
        except Exception:
            break
        body = data[pos+8:pos+8+sz]
        if cid == b'fmt ' and len(body) >= 16:
            ch = struct.unpack('<H', body[2:4])[0]
            rate = struct.unpack('<I', body[4:8])[0]
        elif cid == b'data':
            pcm = body
            break
        pos += 8 + sz + (sz & 1)
    if not pcm:
        return data                      # sin data chunk: dejo el original (raro pero vivo)
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(ch if 1 <= ch <= 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return bio.getvalue()


def _tts_bytes(texto: str) -> bytes:
    """WAV del motor local ACTIVO (Supertonic / Piper de respaldo) con la voz
    activa, normalizado al MISMO volumen que la respuesta — y saneado para el
    navegador (RIFF con duración REAL, no el «unknown» del ffmpeg pipe)."""
    local = _motor_local()
    if local is None:
        raise RuntimeError("no hay motor de voz local cargado")
    with _TTS_LOCK:
        return _reempaqueta_wav(local.normalizar(local.sintetizar(texto)))


def _tts(texto: str) -> str:
    """Sintetiza y SIRVE: el .wav nombrado por turno en runtime/ → /voz/<f>."""
    out = VOZ_DIR / f"tabla-{int(time.time() * 1000)}.wav"
    out.write_bytes(_tts_bytes(texto))
    return "/voz/" + out.name


def _clave_voz_activa() -> str:
    """La clave REAL de la voz activa del motor corriente (para el marker de los fijos)."""
    local = _motor_local()
    try:
        if local is st and st is not None:
            return f"supertonic:{st.VOZ_ACTIVA}"
        if local is pyp and pyp is not None:
            return f"piper:{getattr(pyp, 'VOZ_ACTIVA', 'modelo')}"
    except Exception:
        pass
    return "desconocida"


def _fijos_al_dia() -> bool:
    """¿Los fijos son de la voz ACTIVA y son todos? (el marker de la fábrica —
    si cambian el motor o la voz con /voz, el arranque los regenera solo)."""
    try:
        return FIJOS_MARKER.read_text().strip() == _clave_voz_activa() and _fijos_completos()
    except Exception:
        return False


def _preparar_fijos():
    """La voz de la casa para la espera: «Déjame pensar.» + los rellenos.
    Al arranque SI los fijos no son de la voz activa (marker), determinista,
    una vez."""
    if _fijos_al_dia():
        log(f"fijos al día ({_clave_voz_activa()}) — no regenero")
        return
    try:
        lote = {}
        for nombre, frase in ([("pensar.wav", "Déjame pensar.")]
                              + [(f"relleno-{i}.wav", f) for i, f in enumerate(RELLENO)]):
            lote[nombre] = _tts_bytes(frase)
        for nombre, data in lote.items():
            destino = VOZ_DIR / nombre
            tmp = destino.with_suffix(".wav.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, destino)
        FIJOS_MARKER.write_text(_clave_voz_activa())
        log(f"fijos listos: {len(lote)} audios con {_clave_voz_activa()}")
    except Exception as e:
        log(f"⚠ no pude preparar los fijos: {str(e)[:120]}")


def _solo_texto(resp):
    """Los tags son para la voz: la pantalla y el texto guardado van limpios."""
    try:
        return jv.para_voz(resp)
    except Exception:
        return re.sub(r"\[/?[a-z_]+\]", "", str(resp)).strip()


def _publicar(turno):
    with _EST:
        jv.publicar(turno)


def _pon_estado(nombre):
    with _EST:
        jv.pon_estado(nombre)


# ── HTTP único (el puerto que consolida 3) ──
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002
        camino = self.path.split("?", 1)[0]
        if (self.command, camino) != ("GET", "/estado") \
           and not (self.command == "POST" and camino == "/audio"):
            log("req", self.command, camino[:50])

    def _json(self, d, code=200):
        body = json.dumps(d, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _archivo(self, rel, ctype):
        p = AQUI / rel
        if not p.exists():
            self._json({"error": "no está"}, 404)
            return
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        camino = self.path.split("?", 1)[0]
        if camino == "/estado":
            with lock:
                d = {"estado": estado["v"], "score": round(estado["score"], 3),
                     "pausado": pausado["v"], "wakeword": Path(MODELO).stem,
                     "grabcando": estado["v"] == "listening",
                     "gracia_t": estado.get("gracia_t", 0.0),
                     "ultimo": estado["ultimo"], "ultimo_t": estado["ultimo_t"],
                     "sesion": _sesion()}
            self._json(d)
            return
        if camino in ("/", "/estacion", "/estacion/"):
            self._archivo("client/estacion.html", "text/html; charset=utf-8")
            return
        if camino in ("/tablet", "/tablet/"):
            self._archivo("client/tablet.html", "text/html; charset=utf-8")
            return
        if camino == "/beep.wav":
            self._archivo("client/beep.wav", "audio/wav")
            return
        if camino.startswith("/sprites/"):
            self._archivo(f"client/sprites/{os.path.basename(urllib.parse.urlsplit(camino).path)}", "image/png")
            return
        if camino.startswith("/voz/"):
            base = os.path.basename(urllib.parse.urlsplit(camino).path)
            p = VOZ_DIR / base
            ctype = "audio/mpeg" if p.suffix == ".mp3" else "audio/wav"
            if not p.exists():
                self._json({"error": "no está"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(p.stat().st_size))
            self.end_headers()
            with open(p, "rb") as f:
                self.wfile.write(f.read())
            return
        self._json({"error": "ruta desconocida"}, 404)

    def do_POST(self):
        camino = self.path.split("?", 1)[0]
        if camino == "/sesion-nueva":          # el 🗒️ de la tablet y la estación
            self._json(_sesion_nueva())
            return
        if camino == "/preguntar":             # el turno EXTERNO (la tablet, la API de la casa)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                d = json.loads(self.rfile.read(n).decode()) if n else {}
            except Exception:
                d = {}
            texto = str(d.get("texto") or "").strip()
            if not texto:
                self._json({"error": "sin texto"}, 400)
                return
            self._json(_turno(texto, conversacion=bool(d.get("conversacion", True))))
            return
        if camino == "/audio":
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except Exception:
                n = 0
            if n <= 0 or n > 65536:
                self._json({"error": "chunk inválido"}, 400)
                return
            datos = self.rfile.read(n)
            sec = sec_global["v"] + 1
            sec_global["v"] = sec
            with lock:
                buf.append((sec, np.frombuffer(datos, dtype=np.int16)))
            self._json({"ok": True})
            return
        if camino == "/stop":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                d = json.loads(self.rfile.read(n).decode()) if n else {}
            except Exception:
                d = {}
            on = bool(d.get("on"))
            pausado["v"] = on
            log("escucha", "PAUSADA" if on else "REANUDADA")
            self._json({"pausado": on})
            return
        self._json({"error": "ruta desconocida"}, 404)


def main():
    threading.Thread(target=hilo_detector, daemon=True).start()
    threading.Thread(target=_preparar_fijos, daemon=True).start()   # los fijos al arranque, sin atar el boot
    srv = ThreadingHTTPServer(("0.0.0.0", PUERTO), H)
    srv.daemon_threads = True
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(CERT), str(KEY))
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)

    def cerrar(signum, frame):
        log("salida limpia")
        try:
            srv.shutdown()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, cerrar)
    signal.signal(signal.SIGINT, cerrar)
    log(f"Jenny-Puerta arriba en https://0.0.0.0:{PUERTO}/ — UNA app, un puerto, un cerebro")
    srv.serve_forever()


if __name__ == "__main__":
    main()
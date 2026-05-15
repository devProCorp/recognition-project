"""
Analizador de voz de Fiore — v3
Paradigma: vocabulario conocido → búsqueda directa → IA solo si hay ambigüedad
Clave: Whisper recibe el vocabulario de Fiore como prompt antes de transcribir
"""
from __future__ import annotations
import os, json, subprocess, difflib, tempfile
import numpy as np
import noisereduce as nr
import anthropic
from openai import OpenAI
import imageio_ffmpeg
from pathlib import Path
from dotenv import load_dotenv
from scipy.signal import correlate, stft
from scipy.fftpack import dct
from scipy.io.wavfile import write as wav_write

load_dotenv()

# ─── Biblioteca de referencia acústica ────────────────────────────────────────
_BIBLIOTECA: list[dict] = []

def cargar_biblioteca(ruta: str = "biblioteca_fiore.json") -> None:
    global _BIBLIOTECA
    p = Path(ruta)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        _BIBLIOTECA = data.get("entries", [])
        print(f"Biblioteca cargada: {len(_BIBLIOTECA)} referencias acústicas.", flush=True)
    else:
        print("⚠ biblioteca_fiore.json no encontrada — solo modo texto.", flush=True)

cargar_biblioteca()

FFMPEG      = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000

# ─── Vocabulario de Fiore ──────────────────────────────────────────────────────
# Fuente de verdad: cada entrada tiene significado, categoría, urgencia y
# respuesta directa sugerida al cuidador.
VOCABULARIO: dict[str, dict] = {
    # Necesidades físicas — ALTA urgencia, acción inmediata
    "chichi":             {"significado": "Quiere hacer pipí",           "categoria": "necesidad_fisica",     "urgencia": "ALTA",  "respuesta": "Llevarla al baño ya."},
    "caca":               {"significado": "Quiere ir al baño (caca)",    "categoria": "necesidad_fisica",     "urgencia": "ALTA",  "respuesta": "Llevarla al baño ya."},
    "agua":               {"significado": "Quiere agua",                 "categoria": "necesidad_fisica",     "urgencia": "MEDIA", "respuesta": "Darle agua."},
    # Deseos sociales
    "calle":              {"significado": "Quiere salir a la calle",     "categoria": "deseo_social",         "urgencia": "MEDIA", "respuesta": "Preguntar cuándo quiere salir o planificarlo."},
    "acá":                {"significado": "AMBIGUO (señalar qué está mirando — puede ser Centro Andino, mensaje o baño)", "categoria": "ambigua", "urgencia": "MEDIA", "respuesta": "Observar qué está señalando para entender si quiere ir a algún lugar, enviar un mensaje o ir al baño."},
    # Necesidades alimentarias
    "chocolate caliente": {"significado": "Quiere chocolate caliente",   "categoria": "necesidad_alimentaria","urgencia": "BAJA",  "respuesta": "Prepararle chocolate caliente."},
    "milo":               {"significado": "Quiere Milo",                 "categoria": "necesidad_alimentaria","urgencia": "BAJA",  "respuesta": "Prepararle Milo."},
    "necesito":           {"significado": "Quiere algo (no especificó)", "categoria": "necesidad_vaga",       "urgencia": "MEDIA", "respuesta": "Preguntar: ¿qué necesitas? ¿agua, chocolate, milo?"},
    # Estado / tiempo
    "ya":                 {"significado": "Ya terminé / estoy lista",    "categoria": "estado",               "urgencia": "BAJA",  "respuesta": "Reconocer que terminó."},
    "ahora no más tarde": {"significado": "Ahorita no, después",        "categoria": "tiempo",               "urgencia": "BAJA",  "respuesta": "Respetar su ritmo."},
    # Personas / nombres
    "pipo":               {"significado": "Felipe",                      "categoria": "persona",              "urgencia": "BAJA",  "respuesta": "Verificar si quiere hablar con Felipe o está mencionándolo."},
    # Pronunciaciones especiales → traducción directa
    "duvo":               {"significado": "¿Dónde?",                     "categoria": "pregunta",             "urgencia": "BAJA",  "respuesta": "Responder a su pregunta: ¿dónde está qué?"},
    "dúvo":               {"significado": "¿Dónde?",                     "categoria": "pregunta",             "urgencia": "BAJA",  "respuesta": "Responder a su pregunta: ¿dónde está qué?"},
    "eneo no":            {"significado": "Eso no",                      "categoria": "negacion",             "urgencia": "BAJA",  "respuesta": "Entendido. Preguntar qué sí quiere."},
    "eneo sí":            {"significado": "Eso sí",                      "categoria": "afirmacion",           "urgencia": "BAJA",  "respuesta": "Perfecto, confirmado."},
    # Emociones positivas
    "rico rico rico":     {"significado": "¡Está rico! ¡Le encanta!",   "categoria": "emocion_positiva",     "urgencia": "BAJA",  "respuesta": "Compartir su alegría."},
    "ahora sí":           {"significado": "¡Ahora sí! (alivio/acuerdo)","categoria": "emocion_positiva",     "urgencia": "BAJA",  "respuesta": "Confirmar el acuerdo o alivio."},
    "wepa":               {"significado": "¡Alegría / emoción!",        "categoria": "emocion_positiva",     "urgencia": "BAJA",  "respuesta": "Celebrar con ella."},
    "súper fiore":        {"significado": "¡Alegría / emoción!",        "categoria": "emocion_positiva",     "urgencia": "BAJA",  "respuesta": "Celebrar con ella."},
    "uh sí ya":           {"significado": "Confirmación entusiasta",    "categoria": "afirmacion",           "urgencia": "BAJA",  "respuesta": "Confirmar entendido."},
    "ahí":                {"significado": "Conformidad ('bueno, está bien')", "categoria": "afirmacion",     "urgencia": "BAJA",  "respuesta": "Continuar con lo que se estaba haciendo."},
    "más o menos":        {"significado": "Más o menos",                "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Preguntar qué parte no está bien del todo."},
    # Emociones negativas
    "maluco":             {"significado": "No me gusta / está malo",    "categoria": "emocion_negativa",     "urgencia": "BAJA",  "respuesta": "Retirar o cambiar lo que no le gusta."},
    "uy":                 {"significado": "Disgusto",                   "categoria": "emocion_negativa",     "urgencia": "BAJA",  "respuesta": "Preguntar qué le molestó."},
    "bruja":              {"significado": "Bruja (calificativo negativo)", "categoria": "emocion_negativa",  "urgencia": "BAJA",  "respuesta": "Entender por qué está molesta."},
    "loca":               {"significado": "Loca (calificativo)",        "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Preguntar a qué se refiere."},
    # Correcciones sociales
    "dede":               {"significado": "Cállate",                    "categoria": "correccion_social",    "urgencia": "MEDIA", "respuesta": "Guardar silencio y darle espacio."},
    "eco":                {"significado": "Estás repitiendo lo mismo",  "categoria": "correccion_social",    "urgencia": "MEDIA", "respuesta": "Cambiar de tema o dejar de repetir."},
    "fuera":              {"significado": "Vete / fuera",               "categoria": "correccion_social",    "urgencia": "MEDIA", "respuesta": "Darle espacio y alejarse."},
    # Preguntas que hace
    "qué es eso":         {"significado": "¿Qué es eso?",              "categoria": "pregunta",             "urgencia": "BAJA",  "respuesta": "Explicarle qué es."},
    "cómo estás":         {"significado": "¿Cómo estás?",              "categoria": "pregunta",             "urgencia": "BAJA",  "respuesta": "Responder cómo estás."},
    "qué haces":          {"significado": "¿Qué haces?",               "categoria": "pregunta",             "urgencia": "BAJA",  "respuesta": "Explicarle qué estás haciendo."},
    "por qué":            {"significado": "¿Por qué?",                 "categoria": "pregunta",             "urgencia": "BAJA",  "respuesta": "Dar una explicación sencilla."},
    # Descripciones
    "chico":              {"significado": "Pequeño",                    "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Confirmar o preguntar a qué se refiere."},
    "grande":             {"significado": "Grande",                     "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Confirmar o preguntar a qué se refiere."},
    "caliente":           {"significado": "Caliente",                   "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Verificar si algo está demasiado caliente."},
    "frío":               {"significado": "Frío",                      "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Verificar si tiene frío o algo está frío."},
    "hay poquito":        {"significado": "Hay poco",                   "categoria": "descripcion",          "urgencia": "BAJA",  "respuesta": "Revisar qué le falta."},
    "no hay":             {"significado": "No hay",                     "categoria": "negacion",             "urgencia": "BAJA",  "respuesta": "Confirmar qué falta y conseguirlo."},
    # Frases dirigidas al interlocutor
    "tú sí sabes":        {"significado": "¡Tú sí sabes! (validación / celebración hacia el cuidador)", "categoria": "emocion_positiva",  "urgencia": "BAJA",  "respuesta": "Responderle con afirmación y celebración."},
    "tú qué":             {"significado": "¿Y tú qué? / ¿Tú qué haces?",                              "categoria": "pregunta",           "urgencia": "BAJA",  "respuesta": "Responder a su pregunta."},
    # Saludos (los dice correctamente)
    "hola":               {"significado": "Hola",                       "categoria": "saludo",               "urgencia": "BAJA",  "respuesta": "Saludarla de vuelta."},
    "chao":               {"significado": "Adiós",                      "categoria": "saludo",               "urgencia": "BAJA",  "respuesta": "Despedirse."},
    "sí":                 {"significado": "Sí",                         "categoria": "afirmacion",           "urgencia": "BAJA",  "respuesta": "Confirmar y proceder."},
    "no":                 {"significado": "No",                         "categoria": "negacion",             "urgencia": "BAJA",  "respuesta": "Respetar su negativa y preguntar qué quiere."},
    "eso no":             {"significado": "Eso no",                     "categoria": "negacion",             "urgencia": "BAJA",  "respuesta": "Cambiar lo que se le ofreció."},
    "eso sí":             {"significado": "Eso sí",                     "categoria": "afirmacion",           "urgencia": "BAJA",  "respuesta": "Confirmar y proceder."},
    "allá arriba":        {"significado": "Allá arriba",                "categoria": "lugar",                "urgencia": "BAJA",  "respuesta": "Verificar a qué se refiere arriba."},
    "allá abajo":         {"significado": "Allá abajo",                 "categoria": "lugar",                "urgencia": "BAJA",  "respuesta": "Verificar a qué se refiere abajo."},
}

# Errores conocidos de Whisper con el habla de Fiore
CORRECCIONES_WHISPER: dict[str, str] = {
    "chico chico chico": "rico rico rico",
    "chico chico":       "rico rico",
    "poblado":           "agua",
    "chibichi":          "chichi",
    "chibi":             "chichi",
    "dubo":              "duvo",
    "neo no":            "eneo no",
    "neo sí":            "eneo sí",
}

# Prompt de vocabulario para Whisper — le enseña las palabras antes de transcribir
WHISPER_VOCAB_PROMPT = (
    "Fiorela habla con vocabulario propio. Palabras exactas que usa: "
    "Pipo, dúvo, eneo no, eneo sí, chichi, caca, agua, calle, acá, "
    "necesito, ya, ahora no más tarde, chocolate caliente, milo, "
    "rico rico rico, maluco, dede, eco, wepa, súper Fiore, ahora sí, "
    "uh sí ya, ahí, más o menos, uy, bruja, loca, fuera, tú sí sabes, tú qué, "
    "allá arriba, allá abajo, hay poquito, no hay, eso no, eso sí."
)


# ─── Huella MFCC (misma implementación que construir_biblioteca.py) ───────────

def _calcular_mfcc_vector(audio: np.ndarray, sr: int = SAMPLE_RATE,
                           n_mfcc: int = 13) -> np.ndarray:
    if len(audio) < 200:
        return np.zeros(n_mfcc * 2)
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    _, _, Zxx  = stft(audio_pre, fs=sr, nperseg=512, noverlap=384, window="hann")
    power      = np.abs(Zxx) ** 2
    n_bins     = Zxx.shape[0]
    n_mels     = 40

    def hz_mel(hz):  return 2595 * np.log10(1 + hz / 700)
    def mel_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    mel_pts = np.linspace(hz_mel(0), hz_mel(sr / 2), n_mels + 2)
    bins    = np.clip(np.floor(mel_hz(mel_pts) * 512 / sr).astype(int), 0, n_bins - 1)
    fb      = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)

    mel_e   = np.dot(fb, power)
    mel_e   = np.where(mel_e == 0, 1e-10, mel_e)
    mfcc    = dct(np.log(mel_e), type=2, axis=0, norm="ortho")[:n_mfcc]
    mfcc   -= np.mean(mfcc, axis=1, keepdims=True)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def _similitud_coseno(a: np.ndarray, b: list) -> float:
    vb    = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(vb)
    return float(np.dot(a, vb) / denom) if denom > 1e-9 else 0.0


def buscar_en_biblioteca(audio: np.ndarray) -> tuple[dict | None, float]:
    """
    Compara el audio contra todas las referencias acústicas de la biblioteca.
    Agrupa por frase y devuelve la frase con mayor similitud promedio.
    """
    if not _BIBLIOTECA:
        return None, 0.0

    vector = _calcular_mfcc_vector(audio)

    # Calcular similitud contra cada entrada y agrupar por frase
    scores_por_frase: dict[str, list[float]] = {}
    entry_por_frase:  dict[str, dict]        = {}

    for entry in _BIBLIOTECA:
        sim = _similitud_coseno(vector, entry["mfcc_vector"])
        frase = entry["frase"]
        scores_por_frase.setdefault(frase, []).append(sim)
        if frase not in entry_por_frase or sim > max(scores_por_frase[frase][:-1], default=0):
            entry_por_frase[frase] = entry

    # Ordenar por similitud máxima por frase
    ranking = sorted(
        [(frase, max(scores)) for frase, scores in scores_por_frase.items()],
        key=lambda x: x[1], reverse=True
    )

    if not ranking:
        return None, 0.0

    mejor_frase, mejor_score = ranking[0]
    return entry_por_frase[mejor_frase], mejor_score


# ─── Audio: conversión y limpieza ─────────────────────────────────────────────

def get_model():
    """No-op: usa Whisper API, sin descarga de modelo local."""
    pass


def audio_a_numpy(ruta: str) -> np.ndarray:
    cmd = [FFMPEG, "-y", "-i", ruta, "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def limpiar_audio(audio: np.ndarray) -> np.ndarray:
    muestra = audio[: SAMPLE_RATE // 2]
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=muestra, prop_decrease=0.8)


# ─── Capa 1: Señales acústicas ────────────────────────────────────────────────

def extraer_features_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    duracion = len(audio) / sr
    frame = sr // 10
    hop   = frame // 2
    rms_frames = [
        np.sqrt(np.mean(audio[i: i + frame] ** 2))
        for i in range(0, len(audio) - frame, hop)
    ]
    energia_media = float(np.mean(rms_frames)) if rms_frames else 0.0
    energia_max   = float(np.max(rms_frames))  if rms_frames else 0.0
    variab        = float(np.std(rms_frames))  if rms_frames else 0.0
    zcr           = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
    pitch_hz      = _estimar_pitch(audio, sr)
    return {
        "energia_media":        round(energia_media, 5),
        "energia_max":          round(energia_max, 5),
        "variabilidad_energia": round(variab, 5),
        "zcr":                  round(zcr, 4),
        "pitch_hz":             round(pitch_hz, 1),
        "duracion_seg":         round(duracion, 1),
    }


def _estimar_pitch(audio: np.ndarray, sr: int,
                   min_freq: int = 80, max_freq: int = 500) -> float:
    start   = len(audio) // 4
    seg_len = min(sr // 2, len(audio) // 2)
    seg     = audio[start: start + seg_len]
    if len(seg) < 200:
        return 0.0
    seg  = seg / (np.max(np.abs(seg)) + 1e-9)
    corr = correlate(seg, seg, mode="full")
    corr = corr[len(corr) // 2:]
    min_lag = max(1, int(sr / max_freq))
    max_lag = min(len(corr) - 1, int(sr / min_freq))
    if min_lag >= max_lag:
        return 0.0
    peak = int(np.argmax(corr[min_lag:max_lag])) + min_lag
    return float(sr / peak) if peak > 0 else 0.0


def interpretar_señales(f: dict) -> dict:
    e, v, zcr, p = f["energia_media"], f["variabilidad_energia"], f["zcr"], f["pitch_hz"]
    if e > 0.06:
        urgencia, intensidad = "ALTA",  "voz intensa y fuerte"
    elif e > 0.02:
        urgencia, intensidad = "MEDIA", "voz de intensidad normal"
    else:
        urgencia, intensidad = "BAJA",  "voz suave o tranquila"

    if p > 240 and v > 0.03:
        tono, emo = "pitch alto y variable", "alegre"
    elif zcr > 0.09 and e > 0.03:
        tono, emo = "voz tensa",             "frustrada"
    elif e < 0.008:
        tono, emo = "voz muy baja",          "cansada"
    else:
        tono, emo = "tono normal",           "tranquila"

    return {"urgencia_audio": urgencia, "intensidad": intensidad,
            "tono": tono, "estado_emocional_audio": emo}


# ─── Capa 2: Transcripción con vocabulario de Fiore ───────────────────────────

def transcribir(ruta_audio: str) -> tuple[str, np.ndarray]:
    """
    1. Convierte a numpy y limpia ruido (para análisis acústico).
    2. Envía a Whisper API con el vocabulario de Fiore como prompt.
       Esto hace que Whisper reconozca sus palabras directamente.
    """
    print(f"  Cargando: {Path(ruta_audio).name}", flush=True)
    audio_raw    = audio_a_numpy(ruta_audio)
    audio_limpio = limpiar_audio(audio_raw)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    wav_write(tmp_path, SAMPLE_RATE,
              (np.clip(audio_limpio, -1.0, 1.0) * 32767).astype(np.int16))

    print("  Transcribiendo (Whisper API + vocabulario de Fiore)...", flush=True)
    cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    try:
        with open(tmp_path, "rb") as f:
            resp = cliente.audio.transcriptions.create(
                model           = "whisper-1",
                file            = f,
                language        = "es",
                prompt          = WHISPER_VOCAB_PROMPT,   # ← el cambio clave
                response_format = "verbose_json",
            )
        return resp.text.strip(), audio_limpio
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─── Capa 3: Corrección fonética post-Whisper ─────────────────────────────────

def corregir_transcripcion(texto: str) -> tuple[str, list[str]]:
    correcciones: list[str] = []
    t = texto.lower().strip()

    for error, fix in CORRECCIONES_WHISPER.items():
        if error in t:
            t = t.replace(error, fix)
            correcciones.append(f'"{error}" → "{fix}"')

    vocab_simple = [k for k in VOCABULARIO if " " not in k]
    palabras     = t.split()
    salida       = []
    for p in palabras:
        if p in VOCABULARIO:
            salida.append(p)
            continue
        m = difflib.get_close_matches(p, vocab_simple, n=1, cutoff=0.84)
        if m and m[0] != p:
            correcciones.append(f'"{p}" → "{m[0]}" (fonético)')
            salida.append(m[0])
        else:
            salida.append(p)

    return " ".join(salida), correcciones


# ─── Capa 4: Búsqueda directa en vocabulario ──────────────────────────────────
# Si el texto corresponde a una frase conocida → respuesta inmediata sin IA.

def buscar_en_vocabulario(texto: str) -> tuple[dict | None, str, float]:
    """
    Devuelve (entrada_vocab, metodo, score).
    score 1.0 = exacto, 0.9 = frase contenida, 0.75 = fuzzy.
    """
    t = texto.lower().strip()

    # 1. Coincidencia exacta
    if t in VOCABULARIO:
        return VOCABULARIO[t], "exacto", 1.0

    # 2. Frases del vocabulario contenidas en el texto (más largas primero)
    for frase in sorted(VOCABULARIO.keys(), key=len, reverse=True):
        if frase in t:
            return VOCABULARIO[frase], "frase_contenida", 0.9

    # 3. Fuzzy match global
    m = difflib.get_close_matches(t, VOCABULARIO.keys(), n=1, cutoff=0.75)
    if m:
        return VOCABULARIO[m[0]], f"fuzzy({m[0]})", 0.75

    return None, "no_encontrado", 0.0


def _resultado_directo(entrada: dict, metodo: str, score: float,
                       texto: str, señales: dict) -> dict:
    """Construye el resultado completo desde una entrada del vocabulario (sin IA)."""
    confianza = "ALTO" if score >= 0.9 else "MEDIO"
    nota = None
    if entrada["categoria"] == "ambigua":
        confianza = "BAJO"
        nota = "Palabra ambigua: observar qué está señalando Fiore físicamente."
    elif entrada["categoria"] == "necesidad_vaga":
        confianza = "MEDIO"
        nota = "No especificó qué necesita — preguntar directamente."

    return {
        "interpretacion":    entrada["significado"],
        "intencion":         entrada["categoria"],
        "estado_emocional":  señales["estado_emocional_audio"],
        "urgencia":          entrada["urgencia"],
        "confianza":         confianza,
        "palabras_clave":    [texto],
        "respuesta_sugerida":entrada["respuesta"],
        "nota":              nota,
        "metodo_deteccion":  metodo,
    }


# ─── Capa 5: IA para casos ambiguos ───────────────────────────────────────────

def _vocab_para_prompt() -> str:
    por_cat: dict[str, list[str]] = {}
    for p, info in VOCABULARIO.items():
        por_cat.setdefault(info["categoria"], []).append(
            f'"{p}" → {info["significado"]}'
        )
    return "\n".join(
        f"{cat.upper()}:\n" + "\n".join(f"  {e}" for e in ents)
        for cat, ents in por_cat.items()
    )


def _decidir_resultado(
    match_audio: dict | None, score_audio: float,
    entrada_texto: dict | None, metodo_texto: str, score_texto: float,
    texto_original: str, texto_corregido: str, correcciones: list[str],
    features: dict, señales: dict, sesion: list[dict],
) -> dict:
    """
    TEXTO ES EL MOTOR PRINCIPAL.
    La biblioteca de audio solo CONFIRMA cuando texto ya encontró un match,
    o cuando el score acústico es extraordinariamente alto (>0.93).
    Esto evita falsos positivos entre hablantes distintos.
    """
    frase_texto = _frase_en_vocabulario(entrada_texto) if entrada_texto else None
    frase_audio = match_audio["frase"] if match_audio else None
    ambas_concuerdan = (frase_texto is not None and frase_texto == frase_audio)

    # 1. Texto exacto/contenido → es la señal más confiable
    if entrada_texto and score_texto >= 0.9:
        confianza = "ALTO"
        if ambas_concuerdan and score_audio > 0.82:
            confianza  = "ALTO"
            metodo = f"texto({metodo_texto}) + audio({score_audio:.2f}) ✓"
        else:
            metodo = f"texto_directo({metodo_texto})"
        return {**_resultado_directo(entrada_texto, metodo, score_texto, texto_corregido, señales),
                "confianza": confianza, "metodo_deteccion": metodo}

    # 2. Texto fuzzy con buen score
    if entrada_texto and score_texto >= 0.75:
        metodo = f"texto_fuzzy({metodo_texto})"
        return {**_resultado_directo(entrada_texto, metodo, score_texto, texto_corregido, señales),
                "metodo_deteccion": metodo}

    # 3. Audio con score muy alto Y la frase existe en el vocabulario principal
    #    (umbral alto para compensar que las voces de referencia son del narrador)
    if match_audio and score_audio > 0.93:
        entrada = VOCABULARIO.get(frase_audio)
        if entrada:
            metodo = f"biblioteca_audio_alto({score_audio:.2f})"
            return {**_resultado_directo(entrada, metodo, score_audio, frase_audio, señales),
                    "confianza": "MEDIO", "metodo_deteccion": metodo}

    # 4. Ninguna señal confiable → Claude con todo el contexto
    print("  Sin match confiable → Claude interpreta...", flush=True)
    r = interpretar_con_ia(
        texto_original, texto_corregido, correcciones, features, señales, sesion
    )
    r["metodo_deteccion"] = "IA_claude"
    return r


def _frase_en_vocabulario(entrada: dict) -> str | None:
    """Devuelve la clave del VOCABULARIO cuyo valor es esta entrada."""
    for k, v in VOCABULARIO.items():
        if v is entrada:
            return k
    return None


def _buscar_frase_por_significado(entrada: dict):
    for k, v in VOCABULARIO.items():
        if v is entrada:
            yield k


def interpretar_con_ia(
    transcripcion: str,
    transcripcion_corregida: str,
    correcciones: list[str],
    features: dict,
    señales: dict,
    sesion: list[dict],
) -> dict:
    contexto = ""
    if sesion:
        contexto = "CONTEXTO SESIÓN (últimas frases):\n" + "\n".join(
            f"  {i+1}. \"{s.get('transcripcion_corregida','')}\" → {s.get('interpretacion','')}"
            for i, s in enumerate(sesion[-3:])
        )

    system = f"""Eres intérprete de Fiorela, persona con dificultades del habla.
La transcripción NO fue reconocida directamente en su vocabulario — necesitas interpretar.

{contexto}

VOCABULARIO COMPLETO:
{_vocab_para_prompt()}

REGLAS:
- "acá" sin contexto → confianza BAJO, observar señas físicas.
- "necesito" sin especificar → confianza MEDIO, preguntar qué.
- Urgencia ALTA: chichi, caca (baño inmediato).
- Usa señales de audio para calibrar estado emocional.

Responde SOLO JSON:
{{
  "interpretacion": "...",
  "intencion": "necesidad_fisica|necesidad_alimentaria|deseo_social|pregunta|emocion_positiva|emocion_negativa|correccion_social|descripcion|confirmacion|estado|saludo|ambigua",
  "estado_emocional": "tranquila|alegre|frustrada|urgente|cansada|incierta",
  "urgencia": "ALTA|MEDIA|BAJA",
  "confianza": "ALTO|MEDIO|BAJO",
  "palabras_clave": ["..."],
  "respuesta_sugerida": "...",
  "nota": "..."
}}"""

    user = (
        f'Whisper original: "{transcripcion}"\n'
        + (f'Corregida: "{transcripcion_corregida}"\n' if correcciones else "")
        + (f'Correcciones: {"; ".join(correcciones)}\n' if correcciones else "")
        + f'Audio — intensidad: {señales["intensidad"]}, tono: {señales["tono"]}, '
        f'estado: {señales["estado_emocional_audio"]}\n\nAnaliza y devuelve JSON.'
    )

    ant_key = os.getenv("ANTHROPIC_API_KEY", "")
    oai_key = os.getenv("OPENAI_API_KEY", "")

    if ant_key and not ant_key.startswith("sk-ant-..."):
        try:
            print("  Interpretando con Claude...", flush=True)
            c = anthropic.Anthropic(api_key=ant_key)
            r = c.messages.create(
                model="claude-opus-4-7", max_tokens=600,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return _extraer_json(r.content[0].text.strip())
        except Exception as e:
            print(f"  Claude no disponible ({e.__class__.__name__}), usando GPT-4o...", flush=True)

    if oai_key and not oai_key.startswith("sk-..."):
        print("  Interpretando con GPT-4o...", flush=True)
        c = OpenAI(api_key=oai_key)
        r = c.chat.completions.create(
            model="gpt-4o", max_tokens=600,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        return _extraer_json(r.choices[0].message.content.strip())

    raise RuntimeError("No hay API key válida configurada.")


def _extraer_json(texto: str) -> dict:
    i, f = texto.find("{"), texto.rfind("}") + 1
    if i != -1 and f > i:
        try:
            return json.loads(texto[i:f])
        except json.JSONDecodeError:
            pass
    return {
        "interpretacion": texto, "intencion": "ambigua",
        "estado_emocional": "incierta", "urgencia": "MEDIA",
        "confianza": "BAJO", "palabras_clave": [],
        "respuesta_sugerida": "Revisar manualmente.",
        "nota": "La IA no devolvió JSON válido.",
    }


# ─── Pipeline principal ────────────────────────────────────────────────────────

def analizar(ruta_audio: str, sesion: list[dict] | None = None) -> dict:
    if sesion is None:
        sesion = []

    print(f"\n{'='*55}", flush=True)
    print(f"Analizando: {Path(ruta_audio).name}", flush=True)

    # Capa 1 — transcripción + señales acústicas
    texto_original, audio_limpio = transcribir(ruta_audio)
    print(f"  WHISPER: \"{texto_original}\"", flush=True)

    features = extraer_features_audio(audio_limpio)
    señales  = interpretar_señales(features)

    # Capa 2 — corrección fonética
    texto_corregido, correcciones = corregir_transcripcion(texto_original)
    if correcciones:
        print(f"  CORRECCIONES: {'; '.join(correcciones)}", flush=True)
    print(f"  TEXTO FINAL: \"{texto_corregido}\"", flush=True)

    # Capa 3a — comparación acústica contra biblioteca de referencia
    match_audio, score_audio = buscar_en_biblioteca(audio_limpio)
    if match_audio:
        print(f"  AUDIO MATCH: '{match_audio['frase']}' score={score_audio:.3f}", flush=True)

    # Capa 3b — búsqueda directa por texto en vocabulario
    entrada_texto, metodo_texto, score_texto = buscar_en_vocabulario(texto_corregido)
    if entrada_texto:
        print(f"  TEXTO MATCH: '{texto_corregido}' [{metodo_texto}] score={score_texto:.2f}", flush=True)

    # Decisión: texto es la señal principal; audio solo confirma si score muy alto
    resultado = _decidir_resultado(
        match_audio, score_audio,
        entrada_texto, metodo_texto, score_texto,
        texto_original, texto_corregido, correcciones,
        features, señales, sesion,
    )
    resultado["score_audio"] = round(score_audio, 3)

    print(f"  → {resultado['interpretacion']} [{resultado['urgencia']}]", flush=True)

    return {
        "transcripcion":           texto_original,
        "transcripcion_corregida": texto_corregido,
        "correcciones":            correcciones,
        "features_audio":          features,
        "señales_audio":           señales,
        **resultado,
    }

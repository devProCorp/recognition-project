"""
Construye la biblioteca de referencia acústica de Fiore.
Procesa los audios-diccionario del proyecto, localiza cada frase del vocabulario
con timestamps de Whisper, extrae el segmento de audio y calcula su huella MFCC.
Guarda todo en biblioteca_fiore.json
"""
from __future__ import annotations
import os, json, subprocess, tempfile
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from scipy.fftpack import dct
from scipy.signal import stft
from scipy.io.wavfile import write as wav_write
import imageio_ffmpeg

load_dotenv()

FFMPEG      = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000
CLIENTE     = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

AUDIOS_DICCIONARIO = [
    "WhatsApp Audio 2026-05-14 at 2.48.59 PM.mp4",
    "WhatsApp Audio 2026-05-14 at 2.48.59 PM (1).mp4",
]

# Vocabulario de Fiore con sus frases de referencia para buscar en los audios
FRASES_VOCABULARIO = {
    "hola":               {"significado": "Hola",                        "categoria": "saludo",              "urgencia": "BAJA"},
    "chao":               {"significado": "Adiós",                       "categoria": "saludo",              "urgencia": "BAJA"},
    "cómo estás":         {"significado": "¿Cómo estás?",               "categoria": "pregunta",            "urgencia": "BAJA"},
    "pipo":               {"significado": "Felipe",                      "categoria": "persona",             "urgencia": "BAJA"},
    "sí":                 {"significado": "Sí",                          "categoria": "afirmacion",          "urgencia": "BAJA"},
    "no":                 {"significado": "No",                          "categoria": "negacion",            "urgencia": "BAJA"},
    "no hay":             {"significado": "No hay",                      "categoria": "negacion",            "urgencia": "BAJA"},
    "agua":               {"significado": "Quiere agua",                 "categoria": "necesidad_fisica",    "urgencia": "MEDIA"},
    "calle":              {"significado": "Quiere salir",                "categoria": "deseo_social",        "urgencia": "MEDIA"},
    "acá":                {"significado": "AMBIGUO (lugar/mensaje/baño)","categoria": "ambigua",             "urgencia": "MEDIA"},
    "chocolate caliente": {"significado": "Quiere chocolate caliente",   "categoria": "necesidad_alimentaria","urgencia": "BAJA"},
    "milo":               {"significado": "Quiere Milo",                 "categoria": "necesidad_alimentaria","urgencia": "BAJA"},
    "necesito":           {"significado": "Quiere algo (no especificó)", "categoria": "necesidad_vaga",      "urgencia": "MEDIA"},
    "chichi":             {"significado": "Quiere hacer pipí",           "categoria": "necesidad_fisica",    "urgencia": "ALTA"},
    "caca":               {"significado": "Quiere ir al baño (caca)",    "categoria": "necesidad_fisica",    "urgencia": "ALTA"},
    "listo":              {"significado": "Ya terminé / estoy lista",    "categoria": "estado",              "urgencia": "BAJA"},
    "ya":                 {"significado": "Ya terminé / estoy lista",    "categoria": "estado",              "urgencia": "BAJA"},
    "más tarde":          {"significado": "Ahorita no, después",         "categoria": "tiempo",              "urgencia": "BAJA"},
    "qué":                {"significado": "¿Qué?",                      "categoria": "pregunta",            "urgencia": "BAJA"},
    "qué es eso":         {"significado": "¿Qué es eso?",               "categoria": "pregunta",            "urgencia": "BAJA"},
    "qué haces":          {"significado": "¿Qué haces?",                "categoria": "pregunta",            "urgencia": "BAJA"},
    "por qué":            {"significado": "¿Por qué?",                  "categoria": "pregunta",            "urgencia": "BAJA"},
    "rico rico rico":     {"significado": "¡Está rico! ¡Le encanta!",   "categoria": "emocion_positiva",    "urgencia": "BAJA"},
    "maluco":             {"significado": "No me gusta / está malo",     "categoria": "emocion_negativa",    "urgencia": "BAJA"},
    "wepa":               {"significado": "Alegría / emoción",           "categoria": "emocion_positiva",    "urgencia": "BAJA"},
    "dede":               {"significado": "Cállate",                     "categoria": "correccion_social",   "urgencia": "MEDIA"},
    "eco":                {"significado": "Estás repitiendo lo mismo",   "categoria": "correccion_social",   "urgencia": "MEDIA"},
    "duvo":               {"significado": "¿Dónde?",                    "categoria": "pregunta",            "urgencia": "BAJA"},
    "eneo no":            {"significado": "Eso no",                     "categoria": "negacion",            "urgencia": "BAJA"},
    "eneo sí":            {"significado": "Eso sí",                     "categoria": "afirmacion",          "urgencia": "BAJA"},
    "eso no":             {"significado": "Eso no",                     "categoria": "negacion",            "urgencia": "BAJA"},
    "eso sí":             {"significado": "Eso sí",                     "categoria": "afirmacion",          "urgencia": "BAJA"},
    "más o menos":        {"significado": "Más o menos",                "categoria": "descripcion",         "urgencia": "BAJA"},
    "ahora sí":           {"significado": "¡Ahora sí! (alivio/acuerdo)","categoria": "emocion_positiva",   "urgencia": "BAJA"},
    "fuera":              {"significado": "Vete / fuera",               "categoria": "correccion_social",   "urgencia": "MEDIA"},
    "bruja":              {"significado": "Bruja (calificativo negativo)","categoria": "emocion_negativa",  "urgencia": "BAJA"},
    "loca":               {"significado": "Loca",                       "categoria": "descripcion",         "urgencia": "BAJA"},
    "uy":                 {"significado": "Disgusto",                   "categoria": "emocion_negativa",    "urgencia": "BAJA"},
    "súper fiore":        {"significado": "Alegría / emoción",          "categoria": "emocion_positiva",    "urgencia": "BAJA"},
    "súper":              {"significado": "Alegría / emoción",          "categoria": "emocion_positiva",    "urgencia": "BAJA"},
}


# ─── Conversión de audio ──────────────────────────────────────────────────────

def audio_a_numpy(ruta: str) -> np.ndarray:
    cmd = [FFMPEG, "-y", "-i", ruta, "-f", "f32le",
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def extraer_segmento(audio: np.ndarray, inicio: float, fin: float,
                     padding: float = 0.15) -> np.ndarray:
    """Extrae un segmento del audio con padding opcional."""
    i = max(0, int((inicio - padding) * SAMPLE_RATE))
    f = min(len(audio), int((fin + padding) * SAMPLE_RATE))
    return audio[i:f]


# ─── Huella MFCC ─────────────────────────────────────────────────────────────

def calcular_mfcc_vector(audio: np.ndarray, sr: int = SAMPLE_RATE,
                          n_mfcc: int = 13) -> list[float]:
    """
    Calcula el vector de huella MFCC: media + desviación estándar por coeficiente.
    Resultado: vector de 26 dimensiones (sin librosa, solo scipy).
    """
    if len(audio) < 200:
        return [0.0] * (n_mfcc * 2)

    # Pre-énfasis
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # STFT
    _, _, Zxx = stft(audio_pre, fs=sr, nperseg=512, noverlap=384, window="hann")
    power = np.abs(Zxx) ** 2

    n_bins  = Zxx.shape[0]
    n_mels  = 40

    # Mel filterbank
    def hz_mel(hz):  return 2595 * np.log10(1 + hz / 700)
    def mel_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    mel_pts = np.linspace(hz_mel(0), hz_mel(sr / 2), n_mels + 2)
    hz_pts  = mel_hz(mel_pts)
    bins    = np.clip(np.floor(hz_pts * 512 / sr).astype(int), 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            for k in range(l, c):
                fb[m - 1, k] = (k - l) / (c - l)
        if r > c:
            for k in range(c, r):
                fb[m - 1, k] = (r - k) / (r - c)

    mel_e = np.dot(fb, power)
    mel_e = np.where(mel_e == 0, 1e-10, mel_e)
    log_mel = np.log(mel_e)

    # DCT → MFCC
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]

    # Normalización cepstral media
    mfcc -= np.mean(mfcc, axis=1, keepdims=True)

    media = np.mean(mfcc, axis=1)
    desv  = np.std(mfcc, axis=1)
    return (np.concatenate([media, desv])).tolist()


# ─── Transcripción con timestamps de palabras ─────────────────────────────────

def transcribir_con_palabras(ruta: str) -> tuple[str, list]:
    """Transcribe un audio y devuelve (texto, lista_de_palabras_con_timestamps)."""
    print(f"  Transcribiendo: {Path(ruta).name}")
    with open(ruta, "rb") as f:
        resp = CLIENTE.audio.transcriptions.create(
            model                    = "whisper-1",
            file                     = f,
            language                 = "es",
            response_format          = "verbose_json",
            timestamp_granularities  = ["word"],
        )
    palabras = []
    if hasattr(resp, "words") and resp.words:
        palabras = [
            {"word": w.word.strip().lower(), "start": w.start, "end": w.end}
            for w in resp.words
        ]
    return resp.text, palabras


# ─── Construcción de la biblioteca ────────────────────────────────────────────

def encontrar_frase_en_palabras(frase: str, palabras: list, ventana: int = 5) -> list[dict]:
    """
    Busca una frase (puede ser varias palabras) en la lista de palabras timestamped.
    Devuelve lista de matches: {inicio, fin, palabras_encontradas}
    """
    tokens_frase = frase.lower().split()
    n = len(tokens_frase)
    matches = []

    for i in range(len(palabras) - n + 1):
        segmento = [p["word"].strip(".,¿?¡!") for p in palabras[i:i + n]]
        if segmento == tokens_frase:
            matches.append({
                "inicio": palabras[i]["start"],
                "fin":    palabras[i + n - 1]["end"],
                "texto":  " ".join(segmento),
            })

    return matches


def construir_biblioteca():
    biblioteca = {"entries": [], "version": "1.0", "audios_procesados": []}
    frases_encontradas: set[str] = set()

    for ruta_audio in AUDIOS_DICCIONARIO:
        if not Path(ruta_audio).exists():
            print(f"  ⚠ No encontrado: {ruta_audio}")
            continue

        print(f"\n{'='*55}")
        print(f"Procesando: {ruta_audio}")

        # Cargar audio completo
        audio_np = audio_a_numpy(ruta_audio)
        print(f"  Duración: {len(audio_np)/SAMPLE_RATE:.1f}s")

        # Transcribir con timestamps de palabras
        texto, palabras = transcribir_con_palabras(ruta_audio)
        print(f"  Palabras detectadas: {len(palabras)}")
        print(f"  Texto: {texto[:200]}...")

        biblioteca["audios_procesados"].append(ruta_audio)

        # Buscar cada frase del vocabulario en el audio
        for frase, info in FRASES_VOCABULARIO.items():
            matches = encontrar_frase_en_palabras(frase, palabras)
            for match in matches:
                segmento = extraer_segmento(audio_np, match["inicio"], match["fin"])
                if len(segmento) < 100:
                    continue

                vector = calcular_mfcc_vector(segmento)
                entrada = {
                    "frase":            frase,
                    "significado":      info["significado"],
                    "categoria":        info["categoria"],
                    "urgencia":         info["urgencia"],
                    "audio_fuente":     Path(ruta_audio).name,
                    "timestamp_inicio": round(match["inicio"], 3),
                    "timestamp_fin":    round(match["fin"], 3),
                    "duracion_seg":     round(len(segmento) / SAMPLE_RATE, 3),
                    "mfcc_vector":      vector,
                }
                biblioteca["entries"].append(entrada)
                frases_encontradas.add(frase)
                print(f"  + [{match['inicio']:.1f}s-{match['fin']:.1f}s] '{frase}'")

    # Guardar
    out_path = Path("biblioteca_fiore.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(biblioteca, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"BIBLIOTECA CONSTRUIDA: {out_path}")
    print(f"  Entradas: {len(biblioteca['entries'])}")
    print(f"  Frases únicas: {len(frases_encontradas)}")
    print(f"  Frases encontradas: {', '.join(sorted(frases_encontradas))}")

    frases_faltantes = set(FRASES_VOCABULARIO.keys()) - frases_encontradas
    if frases_faltantes:
        print(f"\n  ⚠ Frases NO encontradas en audios (necesitan grabación):")
        for f in sorted(frases_faltantes):
            print(f"    - {f}")

    return biblioteca


if __name__ == "__main__":
    construir_biblioteca()

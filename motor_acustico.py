"""
Motor acústico de Fiore — DTW enriquecido
Features: MFCC(13) + delta(13) + delta-delta(13) + GFCC(13) = 52 dims/frame
DTW multidimensional con banda Sakoe-Chiba via dtaidistance.
Sin IA ni transcripción — compara patrones acústicos directamente.
"""
from __future__ import annotations
import json, subprocess
import numpy as np
from pathlib import Path
import imageio_ffmpeg
from spafe.features.mfcc import mfcc as _spafe_mfcc
from spafe.features.gfcc import gfcc as _spafe_gfcc
from dtaidistance import dtw_ndim

FFMPEG          = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE     = 16000
BIBLIOTECA_PATH = Path("biblioteca_dtw.json")

# Banda Sakoe-Chiba: limita el warping al 20% de la secuencia más larga
SAKOE_CHIBA_FRAC = 0.20

# Umbral de rechazo: si la mejor distancia supera esto → "no reconocido"
UMBRAL_RECHAZO = 0.80

UMBRAL_CONFIANZA = {
    "ALTO":  0.30,
    "MEDIO": 0.55,
}

# Vocabulario de Fiore — significados y respuestas para el cuidador
VOCABULARIO: dict[str, dict] = {
    "chichi":             {"significado": "Quiere hacer pipí",            "urgencia": "ALTA",  "respuesta": "Llevarla al baño ya."},
    "caca":               {"significado": "Quiere ir al baño (caca)",     "urgencia": "ALTA",  "respuesta": "Llevarla al baño ya."},
    "agua":               {"significado": "Quiere agua",                  "urgencia": "MEDIA", "respuesta": "Darle agua."},
    "calle":              {"significado": "Quiere salir a la calle",      "urgencia": "MEDIA", "respuesta": "Planificar salida."},
    "acá":                {"significado": "AMBIGUO — observar qué señala","urgencia": "MEDIA", "respuesta": "Mirar qué está señalando."},
    "necesito":           {"significado": "Quiere algo (no especificó)",  "urgencia": "MEDIA", "respuesta": "Preguntar: ¿qué necesitas?"},
    "chocolate caliente": {"significado": "Quiere chocolate caliente",    "urgencia": "BAJA",  "respuesta": "Prepararle chocolate."},
    "milo":               {"significado": "Quiere Milo",                  "urgencia": "BAJA",  "respuesta": "Prepararle Milo."},
    "ya":                 {"significado": "Ya terminé / estoy lista",     "urgencia": "BAJA",  "respuesta": "Confirmar que terminó."},
    "pipo":               {"significado": "Felipe",                       "urgencia": "BAJA",  "respuesta": "Verificar si quiere hablar con Felipe."},
    "rico rico rico":     {"significado": "¡Está rico! ¡Le encanta!",    "urgencia": "BAJA",  "respuesta": "Compartir su alegría."},
    "maluco":             {"significado": "No me gusta / está malo",      "urgencia": "BAJA",  "respuesta": "Retirar o cambiar lo que no le gusta."},
    "wepa":               {"significado": "¡Alegría! ¡Emoción!",         "urgencia": "BAJA",  "respuesta": "Celebrar con ella."},
    "dede":               {"significado": "Cállate",                      "urgencia": "MEDIA", "respuesta": "Guardar silencio."},
    "eco":                {"significado": "Estás repitiendo lo mismo",    "urgencia": "MEDIA", "respuesta": "Cambiar de tema."},
    "fuera":              {"significado": "Vete / fuera",                 "urgencia": "MEDIA", "respuesta": "Darle espacio."},
    "duvo":               {"significado": "¿Dónde?",                     "urgencia": "BAJA",  "respuesta": "Responder dónde está qué."},
    "eneo no":            {"significado": "Eso no",                      "urgencia": "BAJA",  "respuesta": "Cambiar lo que se le ofreció."},
    "eneo sí":            {"significado": "Eso sí",                      "urgencia": "BAJA",  "respuesta": "Confirmar y proceder."},
    "ahora sí":           {"significado": "¡Ahora sí! (alivio/acuerdo)", "urgencia": "BAJA",  "respuesta": "Confirmar el acuerdo."},
    "más o menos":        {"significado": "Más o menos",                 "urgencia": "BAJA",  "respuesta": "Preguntar qué parte no está bien."},
    "tú sí sabes":        {"significado": "¡Tú sí sabes! (celebración)", "urgencia": "BAJA",  "respuesta": "Responder con afirmación."},
    "uy":                 {"significado": "Disgusto",                    "urgencia": "BAJA",  "respuesta": "Preguntar qué le molestó."},
    "bruja":              {"significado": "Bruja (negativo)",             "urgencia": "BAJA",  "respuesta": "Entender por qué está molesta."},
    "loca":               {"significado": "Loca (calificativo)",         "urgencia": "BAJA",  "respuesta": "Preguntar a qué se refiere."},
    "súper fiore":        {"significado": "¡Alegría / emoción!",         "urgencia": "BAJA",  "respuesta": "Celebrar con ella."},
    "hola":               {"significado": "Hola",                        "urgencia": "BAJA",  "respuesta": "Saludarla."},
    "chao":               {"significado": "Adiós",                       "urgencia": "BAJA",  "respuesta": "Despedirse."},
    "sí":                 {"significado": "Sí",                          "urgencia": "BAJA",  "respuesta": "Confirmar y proceder."},
    "no":                 {"significado": "No",                          "urgencia": "BAJA",  "respuesta": "Respetar su negativa."},
    "no hay":             {"significado": "No hay",                      "urgencia": "BAJA",  "respuesta": "Confirmar qué falta."},
}


# ─── Conversión de audio ──────────────────────────────────────────────────────

def audio_a_numpy(ruta: str) -> np.ndarray:
    cmd = [FFMPEG, "-y", "-i", str(ruta),
           "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


# ─── Features acústicas ───────────────────────────────────────────────────────

def _delta(feats: np.ndarray, N: int = 2) -> np.ndarray:
    """Primera derivada de una matriz de features (regresión de ventana N)."""
    n_frames = feats.shape[0]
    delta = np.zeros_like(feats)
    pad = np.pad(feats, ((N, N), (0, 0)), mode='edge')
    denom = 2.0 * sum(i * i for i in range(1, N + 1))
    for t in range(n_frames):
        delta[t] = sum(
            i * (pad[t + N + i] - pad[t + N - i]) for i in range(1, N + 1)
        ) / denom
    return delta


def extraer_huella(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extrae huella acústica 52-dimensional por frame:
      MFCC(13) + delta-MFCC(13) + delta2-MFCC(13) + GFCC(13)

    DTW trabaja con la secuencia completa sin resamplear, manejando
    variaciones de velocidad de habla de forma nativa.
    Resultado: (n_frames, 52) — normalizado por z-score por dimensión.
    """
    sig = audio.astype(np.float64)

    try:
        mfcc_feats = _spafe_mfcc(sig, fs=sr, num_ceps=13, pre_emph=True)
    except Exception:
        mfcc_feats = np.zeros((10, 13))

    if len(mfcc_feats) < 4:
        return np.zeros((10, 52), dtype=np.float32)

    d1 = _delta(mfcc_feats, N=2)
    d2 = _delta(d1, N=2)

    try:
        gfcc_feats = _spafe_gfcc(sig, fs=sr, num_ceps=13, pre_emph=True)
    except Exception:
        gfcc_feats = np.zeros_like(mfcc_feats)

    # Alinear longitudes (spafe puede devolver arrays de distinto largo)
    min_len = min(len(mfcc_feats), len(gfcc_feats))
    if min_len < 4:
        return np.zeros((10, 52), dtype=np.float32)

    huella = np.concatenate([
        mfcc_feats[:min_len],
        d1[:min_len],
        d2[:min_len],
        gfcc_feats[:min_len],
    ], axis=1).astype(np.float32)

    # Reemplazar NaN/inf por 0 (puede aparecer en audio muy silencioso)
    huella = np.nan_to_num(huella, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score por dimensión
    mean = np.mean(huella, axis=0, keepdims=True)
    std  = np.std(huella,  axis=0, keepdims=True) + 1e-9
    return ((huella - mean) / std).astype(np.float32)


# ─── DTW con banda Sakoe-Chiba ────────────────────────────────────────────────

def dtw_distancia(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Distancia DTW multidimensional entre dos huellas (n_frames, 52).
    Banda Sakoe-Chiba al 20% evita alineaciones absurdas.
    Normalizada por (len1 + len2) para comparar frases de distinta duración.
    """
    window = max(int(max(len(h1), len(h2)) * SAKOE_CHIBA_FRAC), 5)
    h1_d = np.ascontiguousarray(h1, dtype=np.double)
    h2_d = np.ascontiguousarray(h2, dtype=np.double)
    dist  = dtw_ndim.distance(h1_d, h2_d, window=window, inner_dist='euclidean')
    return float(dist) / (len(h1) + len(h2))


# ─── Biblioteca DTW ───────────────────────────────────────────────────────────

def cargar_biblioteca() -> list[dict]:
    if not BIBLIOTECA_PATH.exists():
        return []
    with open(BIBLIOTECA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for e in data.get("entries", []):
        arr = np.array(e["huella"], dtype=np.float32)
        # Descartar entradas del formato antiguo (n_frames, 4)
        if arr.ndim == 2 and arr.shape[1] == 52:
            e["huella"] = arr
            entries.append(e)
    return entries


def guardar_biblioteca(entries: list[dict]) -> None:
    serializables = []
    for e in entries:
        copia = {k: v for k, v in e.items() if k != "huella"}
        copia["huella"] = e["huella"].tolist()
        serializables.append(copia)
    with open(BIBLIOTECA_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": serializables}, f, ensure_ascii=False, indent=2)


def agregar_muestra(frase: str, audio: np.ndarray) -> int:
    """Agrega una muestra de calibración. Devuelve total de muestras para esa frase."""
    entries = cargar_biblioteca()
    huella  = extraer_huella(audio)
    entries.append({"frase": frase, "huella": huella})
    guardar_biblioteca(entries)
    return sum(1 for e in entries if e["frase"] == frase)


def estado_calibracion() -> dict[str, int]:
    """Cuántas muestras tiene cada frase del vocabulario."""
    entries = cargar_biblioteca()
    conteo  = {frase: 0 for frase in VOCABULARIO}
    for e in entries:
        if e["frase"] in conteo:
            conteo[e["frase"]] += 1
    return conteo


# ─── Reconocimiento ───────────────────────────────────────────────────────────

def reconocer(audio: np.ndarray) -> dict:
    """
    Compara el audio contra todas las referencias de la biblioteca.
    Devuelve la frase más cercana, su confianza y distancia DTW.
    Si la mejor distancia supera UMBRAL_RECHAZO → frase=None (no reconocido).
    """
    entries = cargar_biblioteca()
    if not entries:
        return {
            "frase": None, "confianza": "BAJO",
            "distancia_dtw": None,
            "mensaje": "Biblioteca vacía — calibrar primero.",
        }

    huella_nueva = extraer_huella(audio)

    # Calcular distancia DTW contra cada referencia, agrupar por frase
    distancias: dict[str, list[float]] = {}
    for e in entries:
        dist  = dtw_distancia(huella_nueva, e["huella"])
        frase = e["frase"]
        distancias.setdefault(frase, []).append(dist)

    # Para cada frase: distancia mínima (mejor muestra)
    min_por_frase = {f: min(ds) for f, ds in distancias.items()}

    # Ranking de menor a mayor distancia
    ranking = sorted(min_por_frase.items(), key=lambda x: x[1])
    mejor_frase, mejor_dist = ranking[0]

    # Rechazo duro: demasiado lejos de cualquier referencia
    if mejor_dist > UMBRAL_RECHAZO:
        return {
            "frase": None, "confianza": "BAJO",
            "distancia_dtw": round(mejor_dist, 4),
            "mensaje": f"No reconocido (distancia {mejor_dist:.3f} > umbral {UMBRAL_RECHAZO})",
        }

    # Nivel de confianza
    if mejor_dist < UMBRAL_CONFIANZA["ALTO"]:
        confianza = "ALTO"
    elif mejor_dist < UMBRAL_CONFIANZA["MEDIO"]:
        confianza = "MEDIO"
    else:
        confianza = "BAJO"

    segunda = ranking[1] if len(ranking) > 1 else None

    return {
        "frase":          mejor_frase,
        "confianza":      confianza,
        "distancia_dtw":  round(mejor_dist, 4),
        "segunda_opcion": segunda[0] if segunda else None,
        "dist_segunda":   round(segunda[1], 4) if segunda else None,
    }

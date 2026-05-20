"""
Migra biblioteca_fiore.json → biblioteca_dtw.json

Lee los timestamps de biblioteca_fiore.json, re-extrae los segmentos
de audio y calcula las huellas (n_frames, 52) que necesita motor_acustico.
"""
import json
import numpy as np
import subprocess
from pathlib import Path
import imageio_ffmpeg
from motor_acustico import extraer_huella, guardar_biblioteca, VOCABULARIO

FFMPEG      = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000
PADDING     = 0.15  # segundos de margen en cada lado del segmento


def audio_a_numpy(ruta: str) -> np.ndarray:
    cmd = [FFMPEG, "-y", "-i", ruta, "-f", "f32le",
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def extraer_segmento(audio: np.ndarray, inicio: float, fin: float) -> np.ndarray:
    i = max(0, int((inicio - PADDING) * SAMPLE_RATE))
    f = min(len(audio), int((fin + PADDING) * SAMPLE_RATE))
    return audio[i:f]


def migrar():
    with open("biblioteca_fiore.json", encoding="utf-8") as f:
        data = json.load(f)

    entradas_src = data.get("entries", [])
    print(f"Entradas en biblioteca_fiore.json: {len(entradas_src)}")

    # Cache de audios cargados (evita recargar el mismo archivo)
    cache_audio: dict[str, np.ndarray] = {}

    nuevas_entradas = []
    saltadas = 0

    for entrada in entradas_src:
        frase      = entrada["frase"]
        fuente     = entrada["audio_fuente"]
        t_inicio   = entrada["timestamp_inicio"]
        t_fin      = entrada["timestamp_fin"]

        # Solo migrar frases que estén en el vocabulario del motor
        if frase not in VOCABULARIO:
            saltadas += 1
            continue

        if not Path(fuente).exists():
            print(f"  ⚠ Audio no encontrado: {fuente} — saltando '{frase}'")
            saltadas += 1
            continue

        if fuente not in cache_audio:
            print(f"  Cargando {fuente}...")
            cache_audio[fuente] = audio_a_numpy(fuente)

        segmento = extraer_segmento(cache_audio[fuente], t_inicio, t_fin)

        if len(segmento) < 400:  # menos de 25ms — demasiado corto
            saltadas += 1
            continue

        huella = extraer_huella(segmento)

        if huella.shape[0] < 4:  # muy pocos frames
            saltadas += 1
            continue

        nuevas_entradas.append({
            "frase":  frase,
            "huella": huella,
        })
        print(f"  ✓ '{frase}'  [{t_inicio:.2f}s-{t_fin:.2f}s]  huella={huella.shape}")

    guardar_biblioteca(nuevas_entradas)

    # Reporte
    frases_migradas: dict[str, int] = {}
    for e in nuevas_entradas:
        frases_migradas[e["frase"]] = frases_migradas.get(e["frase"], 0) + 1

    print(f"\n{'='*50}")
    print(f"Migradas:  {len(nuevas_entradas)} entradas")
    print(f"Saltadas:  {saltadas}")
    print(f"Frases en biblioteca_dtw.json:")
    for frase, n in sorted(frases_migradas.items()):
        estado = "✓" if n >= 3 else f"⚠ solo {n}"
        print(f"  {estado}  {n}x  {frase}")

    faltantes = [f for f in VOCABULARIO if f not in frases_migradas]
    if faltantes:
        print(f"\nFrases sin muestras (necesitan calibración manual):")
        for f in sorted(faltantes):
            print(f"  ✗  {f}")

    print(f"\n✅ biblioteca_dtw.json lista con {len(nuevas_entradas)} entradas.")


if __name__ == "__main__":
    migrar()

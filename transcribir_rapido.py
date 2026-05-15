"""
Transcripción mejorada: Whisper large-v3 + denoising de audio.
Compara resultado limpio vs sin limpiar para cada audio.
"""
import whisper
import numpy as np
import noisereduce as nr
import subprocess
import os
import json
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000

AUDIOS = [
    "WhatsApp Audio 2026-05-14 at 2.48.59 PM.mp4",
    "WhatsApp Audio 2026-05-14 at 2.48.59 PM (1).mp4",
]


def audio_a_numpy(ruta: str) -> np.ndarray:
    cmd = [
        FFMPEG, "-y", "-i", ruta,
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def limpiar_audio(audio: np.ndarray) -> np.ndarray:
    """Elimina ruido de fondo usando los primeros 0.5s como muestra de ruido."""
    muestra_ruido = audio[:SAMPLE_RATE // 2]
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=muestra_ruido, prop_decrease=0.8)


print("Cargando Whisper large-v3 (primera vez descarga ~3GB, luego es instantáneo)...", flush=True)
model = whisper.load_model("large-v3")
print("Modelo listo.\n", flush=True)

resultados = {}

for i, audio_path in enumerate(AUDIOS, 1):
    print(f"{'='*60}", flush=True)
    print(f"Audio {i}: {audio_path}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  Cargando audio...", flush=True)
    audio_raw = audio_a_numpy(audio_path)

    print("  Limpiando ruido de fondo...", flush=True)
    audio_limpio = limpiar_audio(audio_raw)

    print("  Transcribiendo audio ORIGINAL...", flush=True)
    res_original = model.transcribe(audio_raw, language="es", verbose=False)
    texto_original = res_original["text"].strip()

    print("  Transcribiendo audio LIMPIO...", flush=True)
    res_limpio = model.transcribe(audio_limpio, language="es", verbose=False)
    texto_limpio = res_limpio["text"].strip()

    print(f"\n  [ORIGINAL] {texto_original}", flush=True)
    print(f"\n  [LIMPIO]   {texto_limpio}\n", flush=True)

    resultados[f"audio_{i}"] = {
        "archivo": audio_path,
        "original": texto_original,
        "limpio": texto_limpio,
    }

with open("transcripciones_mejoradas.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("✅ Guardado en transcripciones_mejoradas.json", flush=True)

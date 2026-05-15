"""
Pipeline completo: Audio de Fiore → Whisper → Claude → ElevenLabs
"""
import os
import json
import subprocess
import numpy as np
import noisereduce as nr
import whisper
import anthropic
from openai import OpenAI
import imageio_ffmpeg
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000

DICCIONARIO = """
NOMBRES / PERSONAS:
- Pipo → Felipe
- Fiore / Fiorela → ella misma

PRONUNCIACIONES ESPECIALES:
- "dúvo" → ¿Dónde?
- "Eneo no" → Eso no
- "Eneo sí" → Eso sí

NECESIDADES Y DESEOS:
- Agua → Quiere agua (lo dice correctamente)
- Calle → Quiere salir a la calle
- Acá (contexto: lugar) → Quiere ir al Centro Andino
- Acá (contexto: mensaje) → Quiere enviar un mensaje
- Acá (contexto: baño) → Quiere ir al baño
- Chichi → Quiere hacer pipí
- Caca → Quiere ir al baño (caca)
- Chocolate caliente → Quiere chocolate caliente
- Milo → Quiere Milo
- Necesito → Quiere algo (pedir contexto si no está claro)
- Ya → Ya terminé / estoy lista
- Ahora no más tarde → Ahorita no, después

PREGUNTAS:
- ¿Qué es eso? → ¿Qué es eso?
- ¿Cómo estás? → ¿Cómo estás?
- ¿Qué haces? → ¿Qué haces?
- ¿Por qué? → ¿Por qué?

DESCRIPCIONES:
- Rico, rico, rico → ¡Está rico! / ¡Me encanta!
- Maluco → Está malo / no me gusta
- Chico → Pequeño
- Grande → Grande
- Caliente → Caliente / está caliente
- Frío → Frío
- Hay poquito → Hay poco
- No hay → No hay

EXPRESIONES:
- Ahora sí → ¡Ahora sí! (alivio o acuerdo)
- Uh sí ya → Expresión de confirmación
- Ahí → Conformidad ("bueno, está bien")
- Uy → Disgusto / no me gusta
- Eso no → Eso no
- Eso sí → Eso sí
- Más o menos → Más o menos
- Dede → Cállate
- Eco → Estás repitiendo lo mismo
- Wepa / Súper Fiore → Alegría / emoción

PERSONAS / ADJETIVOS:
- Loca → Loca
- Bruja → Bruja
"""

SYSTEM_PROMPT = f"""Eres un intérprete especializado en comunicación de Fiore, una persona con dificultades del habla que ha desarrollado su propio vocabulario.

Tu tarea es recibir la transcripción de lo que dijo Fiore y devolver:
1. Lo que Fiore realmente quiso decir, en español claro y natural.
2. Tu nivel de confianza: ALTO, MEDIO o BAJO.
3. Si la confianza es BAJA, explica brevemente por qué.

Usa este diccionario de su comunicación:
{DICCIONARIO}

REGLAS IMPORTANTES:
- Si Fiore dice algo que no está en el diccionario pero puedes inferirlo por contexto, hazlo con confianza MEDIA.
- Si la transcripción es ambigua o confusa, di confianza BAJA y explica qué parte no entendiste.
- "Acá" sin contexto claro → confianza BAJA, pide más información.
- "Necesito" sin especificar qué → confianza MEDIA, sugiere preguntar qué quiere.
- No inventes ni supongas cosas que no estén en el audio o el diccionario.
- Responde SIEMPRE en este formato JSON:

{{
  "interpretacion": "Lo que Fiore realmente quiso decir",
  "confianza": "ALTO | MEDIO | BAJO",
  "nota": "Solo si confianza es MEDIO o BAJO — explica brevemente"
}}"""


# ─── Whisper (se carga una sola vez) ──────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print("Cargando Whisper large-v3...", flush=True)
        _model = whisper.load_model("large-v3")
        print("Modelo listo.", flush=True)
    return _model


# ─── Audio ────────────────────────────────────────────────────────────────────
def audio_a_numpy(ruta: str) -> np.ndarray:
    cmd = [FFMPEG, "-y", "-i", ruta, "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def limpiar_audio(audio: np.ndarray) -> np.ndarray:
    muestra_ruido = audio[:SAMPLE_RATE // 2]
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=muestra_ruido, prop_decrease=0.8)


# ─── Transcripción ────────────────────────────────────────────────────────────
def transcribir(ruta_audio: str) -> str:
    print(f"  Cargando audio: {Path(ruta_audio).name}", flush=True)
    audio_raw = audio_a_numpy(ruta_audio)
    print("  Limpiando ruido...", flush=True)
    audio_limpio = limpiar_audio(audio_raw)
    print("  Transcribiendo con Whisper large-v3...", flush=True)
    result = get_model().transcribe(audio_limpio, language="es", verbose=False)
    return result["text"].strip()


# ─── Interpretación con IA ────────────────────────────────────────────────────
def _extraer_json(texto: str) -> dict:
    inicio = texto.find("{")
    fin = texto.rfind("}") + 1
    if inicio != -1 and fin > inicio:
        return json.loads(texto[inicio:fin])
    return {"interpretacion": texto, "confianza": "BAJO", "nota": "Respuesta no estructurada"}


def interpretar_con_claude(transcripcion: str) -> dict:
    cliente = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    respuesta = cliente.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Fiore dijo esto (transcripción):\n\n\"{transcripcion}\""}]
    )
    return _extraer_json(respuesta.content[0].text.strip())


def interpretar_con_openai(transcripcion: str) -> dict:
    cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    respuesta = cliente.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Fiore dijo esto (transcripción):\n\n\"{transcripcion}\""}
        ]
    )
    return _extraer_json(respuesta.choices[0].message.content.strip())


def interpretar(transcripcion: str) -> dict:
    """Intenta Claude primero, si falla usa OpenAI como fallback."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key and not anthropic_key.startswith("sk-ant-..."):
        try:
            print("  Interpretando con Claude...", flush=True)
            return interpretar_con_claude(transcripcion)
        except Exception as e:
            print(f"  Claude no disponible ({e.__class__.__name__}), usando OpenAI...", flush=True)

    if openai_key and not openai_key.startswith("sk-..."):
        print("  Interpretando con GPT-4o...", flush=True)
        return interpretar_con_openai(transcripcion)

    raise RuntimeError("No hay API key válida de Claude ni de OpenAI en el .env")


# ─── Pipeline completo ────────────────────────────────────────────────────────
def analizar(ruta_audio: str) -> dict:
    print(f"\n{'='*50}", flush=True)
    print(f"Analizando: {Path(ruta_audio).name}", flush=True)
    print(f"{'='*50}", flush=True)

    transcripcion = transcribir(ruta_audio)
    print(f"\n  TRANSCRIPCIÓN: {transcripcion}", flush=True)

    resultado = interpretar(transcripcion)
    print(f"\n  INTERPRETACIÓN: {resultado['interpretacion']}", flush=True)
    print(f"  CONFIANZA: {resultado['confianza']}", flush=True)
    if resultado.get("nota"):
        print(f"  NOTA: {resultado['nota']}", flush=True)

    return {
        "audio": ruta_audio,
        "transcripcion": transcripcion,
        **resultado
    }


# ─── Prueba rápida ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Sin argumento: prueba con texto directo (sin audio)
        print("Modo prueba: interpretando frases del diccionario...\n")
        cliente = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        frases_prueba = [
            "Rico, rico, rico",
            "Acá",
            "Necesito",
            "Eneo no, calle",
            "Pipo, dúvo",
            "Chichi",
            "Ya",
        ]

        for frase in frases_prueba:
            resultado = interpretar(frase)
            print(f"Fiore: \"{frase}\"")
            print(f"  → {resultado['interpretacion']} [{resultado['confianza']}]")
            if resultado.get("nota"):
                print(f"  ⚠ {resultado['nota']}")
            print()
    else:
        # Con argumento: analizar audio real
        resultado = analizar(sys.argv[1])
        print(f"\nResultado final: {json.dumps(resultado, ensure_ascii=False, indent=2)}")

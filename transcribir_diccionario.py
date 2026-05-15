"""
Paso 1: Transcribir los audios-diccionario con Whisper local Y con OpenAI API.
Compara los dos resultados para elegir el más preciso.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AUDIO_1 = "WhatsApp Audio 2026-05-14 at 2.48.59 PM.mp4"
AUDIO_2 = "WhatsApp Audio 2026-05-14 at 2.48.59 PM (1).mp4"
AUDIOS = [AUDIO_1, AUDIO_2]


def transcribir_local(ruta_audio: str) -> str:
    """Transcripción con Whisper corriendo en tu máquina (gratis)."""
    import whisper
    print(f"  [Local] Cargando modelo Whisper...")
    model = whisper.load_model("small")  # small: rápido, bueno para español
    print(f"  [Local] Transcribiendo {Path(ruta_audio).name}...")
    result = model.transcribe(ruta_audio, language="es")
    return result["text"].strip()


def transcribir_openai(ruta_audio: str) -> str:
    """Transcripción con Whisper vía API de OpenAI."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️  Sin OPENAI_API_KEY en .env — saltando transcripción API"
    client = OpenAI(api_key=api_key)
    print(f"  [API]   Enviando a OpenAI Whisper API...")
    with open(ruta_audio, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="es"
        )
    return result.text.strip()


def main():
    resultados = {}

    for audio in AUDIOS:
        if not Path(audio).exists():
            print(f"❌ No encontré: {audio}")
            continue

        nombre = Path(audio).stem
        print(f"\n{'='*60}")
        print(f"Audio: {audio}")
        print(f"{'='*60}")

        print("\n→ Método 1: Whisper LOCAL")
        texto_local = transcribir_local(audio)
        print(f"  Resultado: {texto_local}")

        print("\n→ Método 2: OpenAI API")
        texto_api = transcribir_openai(audio)
        print(f"  Resultado: {texto_api}")

        resultados[nombre] = {
            "archivo": audio,
            "whisper_local": texto_local,
            "openai_api": texto_api,
        }

    # Guardar resultados para el siguiente paso
    with open("diccionario_transcripciones.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"\n\n✅ Transcripciones guardadas en diccionario_transcripciones.json")
    print("Revisa los resultados y dile a Claude cuál transcripción fue más precisa.")


if __name__ == "__main__":
    main()

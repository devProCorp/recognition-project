"""
Servidor web — Analista de Voz Fiore
"""
import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from pipeline import transcribir, interpretar, get_model
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Pre-cargar Whisper al arrancar el servidor (evita espera en la primera petición)
print("Pre-cargando modelo Whisper...", flush=True)
get_model()
print("Servidor listo.\n", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analizar", methods=["POST"])
def analizar():
    if "audio" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    archivo = request.files["audio"]
    if archivo.filename == "":
        return jsonify({"error": "Archivo vacío"}), 400

    # Guardar temporalmente
    ext = Path(archivo.filename).suffix or ".mp4"
    ruta = UPLOAD_FOLDER / f"{uuid.uuid4()}{ext}"
    archivo.save(ruta)

    try:
        transcripcion = transcribir(str(ruta))
        resultado = interpretar(transcripcion)
        return jsonify({
            "transcripcion": transcripcion,
            "interpretacion": resultado.get("interpretacion", ""),
            "confianza": resultado.get("confianza", "BAJO"),
            "nota": resultado.get("nota", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        ruta.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(debug=False, port=5050)

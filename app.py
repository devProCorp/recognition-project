"""
Analista de Voz Fiore — servidor web
Motor: DTW acústico personalizado (sin IA para el reconocimiento)
"""
import os, uuid
from typing import Optional
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from motor_acustico import (
    audio_a_numpy, extraer_huella, reconocer,
    agregar_muestra, estado_calibracion, VOCABULARIO,
    cargar_biblioteca, guardar_biblioteca,
)
from dotenv import load_dotenv
import imageio_ffmpeg, subprocess
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from flask import Response
from supabase import create_client, Client

load_dotenv()

_supabase: Client = create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL", ""),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
)

FFMPEG        = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE   = 16000
app           = Flask(__name__)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

_el_client  = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", ""))
JESSICA_ID  = "cgSgspJ2msm6clMCkdW9"  # Jessica — Playful, Bright, Warm

_historial:        list[dict] = []
_segmentos:        dict[str, dict] = {}   # seg_id → {frase, ruta}
_audios_respuesta: dict[str, bytes] = {}  # audio_id → mp3 bytes


def _generar_audio_respuesta(texto: str) -> Optional[str]:
    try:
        audio_gen = _el_client.text_to_speech.convert(
            voice_id=JESSICA_ID,
            text=texto,
            model_id="eleven_multilingual_v2",
        )
        audio_bytes = b"".join(audio_gen)
        audio_id = uuid.uuid4().hex[:10]
        _audios_respuesta[audio_id] = audio_bytes
        return audio_id
    except Exception as e:
        print(f"ElevenLabs error: {e}", flush=True)
        return None


def archivo_a_numpy(archivo) -> np.ndarray:
    ext  = Path(archivo.filename).suffix or ".mp4"
    ruta = UPLOAD_FOLDER / f"{uuid.uuid4()}{ext}"
    archivo.save(ruta)
    try:
        return audio_a_numpy(str(ruta))
    finally:
        ruta.unlink(missing_ok=True)


def _slug(frase: str) -> str:
    """Convierte frase a slug seguro para rutas de Storage (sin tildes ni espacios)."""
    import unicodedata
    s = unicodedata.normalize("NFD", frase)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace(" ", "_")


def _subir_audio_supabase(ruta_local: Path, frase: str) -> Optional[str]:
    """Convierte a WAV y sube a Supabase Storage. Devuelve la URL firmada."""
    try:
        wav_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.wav"
        subprocess.run(
            [FFMPEG, "-y", "-i", str(ruta_local),
             "-f", "wav", "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_path)],
            capture_output=True, check=True,
        )
        carpeta = _slug(frase)
        nombre  = f"{carpeta}/{uuid.uuid4().hex}.wav"
        with open(wav_path, "rb") as f:
            _supabase.storage.from_("calibraciones-audio").upload(
                nombre, f, {"content-type": "audio/wav"}
            )
        wav_path.unlink(missing_ok=True)
        res = _supabase.storage.from_("calibraciones-audio").create_signed_url(nombre, 3600 * 24 * 365)
        return res.get("signedURL")
    except Exception as e:
        print(f"Aviso Storage: {e}", flush=True)
        return None


# ─── Rutas ────────────────────────────────────────────────────────────────────

@app.route("/vocabulario")
def ruta_vocabulario():
    return jsonify(VOCABULARIO)


@app.route("/estado_calibracion")
def ruta_estado():
    return jsonify(estado_calibracion())


@app.route("/calibrar", methods=["POST"])
def ruta_calibrar():
    """Agrega una muestra de calibración para una frase."""
    frase = request.form.get("frase", "").strip()
    if not frase or frase not in VOCABULARIO:
        return jsonify({"error": "Frase no válida"}), 400
    if "audio" not in request.files:
        return jsonify({"error": "Sin audio"}), 400

    archivo = request.files["audio"]
    ext  = Path(archivo.filename).suffix or ".webm"
    ruta = UPLOAD_FOLDER / f"{uuid.uuid4().hex}{ext}"
    archivo.save(ruta)

    try:
        audio  = audio_a_numpy(str(ruta))
        huella = extraer_huella(audio)

        # Guardar huella en biblioteca local
        entries = cargar_biblioteca()
        entries.append({"frase": frase, "huella": huella})
        guardar_biblioteca(entries)
        n_muestras = sum(1 for e in entries if e["frase"] == frase)

        # Persistir en Supabase (huella + audio)
        audio_url = _subir_audio_supabase(ruta, frase)
        try:
            _supabase.table("calibraciones").insert({
                "frase":     frase,
                "muestras":  n_muestras,
                "fuente":    "manual",
                "audio_url": audio_url,
                "huella":    huella.tolist(),
            }).execute()
        except Exception as e:
            print(f"Aviso: no se pudo guardar huella en Supabase: {e}", flush=True)

        return jsonify({
            "ok": True,
            "frase": frase,
            "muestras": n_muestras,
            "listo": n_muestras >= 3,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        ruta.unlink(missing_ok=True)


@app.route("/analizar", methods=["POST"])
def ruta_analizar():
    """Analiza un audio usando el motor DTW."""
    if "audio" not in request.files:
        return jsonify({"error": "Sin audio"}), 400

    try:
        audio = archivo_a_numpy(request.files["audio"])

        # Reconocimiento acústico
        resultado = reconocer(audio)

        if resultado["frase"] is None:
            return jsonify({
                "no_reconocido": True,
                "mensaje": resultado["mensaje"],
                "distancia_dtw": resultado.get("distancia_dtw"),
            })

        frase   = resultado["frase"]
        entrada = VOCABULARIO.get(frase, {})

        texto_respuesta = entrada.get("respuesta", "—")
        audio_id = _generar_audio_respuesta(texto_respuesta)

        respuesta = {
            "frase_detectada":  frase,
            "significado":      entrada.get("significado", "—"),
            "urgencia":         entrada.get("urgencia", "BAJA"),
            "respuesta":        texto_respuesta,
            "confianza":        resultado["confianza"],
            "distancia_dtw":    resultado["distancia_dtw"],
            "segunda_opcion":   resultado.get("segunda_opcion"),
            "dist_segunda":     resultado.get("dist_segunda"),
            "audio_id":         audio_id,
        }

        try:
            _supabase.table("grabaciones").insert({
                "frase_detectada": frase,
                "significado":     entrada.get("significado", ""),
                "urgencia":        entrada.get("urgencia", "BAJA"),
                "respuesta":       entrada.get("respuesta", ""),
                "confianza":       resultado["confianza"],
                "distancia_dtw":   resultado["distancia_dtw"],
                "segunda_opcion":  resultado.get("segunda_opcion"),
                "dist_segunda":    resultado.get("dist_segunda"),
            }).execute()
        except Exception:
            pass  # no bloquear la respuesta si falla la BD

        _historial.append({
            "frase": frase,
            "significado": entrada.get("significado", ""),
            "urgencia": entrada.get("urgencia", "BAJA"),
        })
        if len(_historial) > 30:
            _historial.pop(0)

        return jsonify(respuesta)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/historial")
def ruta_historial():
    try:
        resp = (
            _supabase.table("grabaciones")
            .select("frase_detectada, significado, urgencia, created_at")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        rows = [
            {
                "frase":      r["frase_detectada"],
                "significado": r["significado"],
                "urgencia":    r["urgencia"],
                "created_at":  r["created_at"],
            }
            for r in resp.data
        ]
        return jsonify(rows)
    except Exception:
        return jsonify(list(reversed(_historial[-10:])))


@app.route("/historial", methods=["DELETE"])
def limpiar_historial():
    try:
        _supabase.table("grabaciones").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    except Exception:
        pass
    _historial.clear()
    return jsonify({"ok": True})


# ─── Segmentador automático ───────────────────────────────────────────────────
#
# El audio tiene el patrón:
#   [voz normal dice "frase X"]  →  [pausa 0-3 s]  →  [Fiore repite a su manera]
#   [voz normal dice "frase Y"]  →  [pausa 0-3 s]  →  [Fiore repite a su manera]
#
# Whisper detecta la voz normal con timestamps.
# Se extraen DOS segmentos por frase:
#   ref_id   → voz normal (referencia auditiva)
#   fiore_id → audio DESPUÉS de la voz normal (la versión de Fiore)
#
# El límite derecho del segmento de Fiore es el inicio de la siguiente frase
# detectada por Whisper, o VENTANA_FIORE segundos como máximo.

def _buscar_frases(palabras: list[dict]) -> list[dict]:
    encontrados = []
    frases_ord = sorted(VOCABULARIO.keys(), key=lambda f: len(f.split()), reverse=True)
    usados: set[int] = set()

    for frase in frases_ord:
        tokens = frase.lower().split()
        n = len(tokens)
        for i in range(len(palabras) - n + 1):
            if any(j in usados for j in range(i, i + n)):
                continue
            segmento = [p["word"].strip(".,¿?¡!\"'") for p in palabras[i: i + n]]
            if segmento == tokens:
                encontrados.append({
                    "frase":  frase,
                    "inicio": palabras[i]["start"],
                    "fin":    palabras[i + n - 1]["end"],
                })
                for j in range(i, i + n):
                    usados.add(j)

    encontrados.sort(key=lambda m: m["inicio"])
    return encontrados


def _cortar_wav(fuente: str, t_inicio: float, t_dur: float) -> tuple[str, Path]:
    seg_id   = uuid.uuid4().hex[:10]
    seg_ruta = UPLOAD_FOLDER / f"{seg_id}.wav"
    subprocess.run(
        [FFMPEG, "-y", "-i", fuente,
         "-ss", str(max(0.0, t_inicio)), "-t", str(max(0.1, t_dur)),
         "-f", "wav", "-ar", str(SAMPLE_RATE), "-ac", "1",
         str(seg_ruta)],
        capture_output=True,
    )
    return seg_id, seg_ruta


@app.route("/segmentar", methods=["POST"])
def ruta_segmentar():
    if "audio" not in request.files:
        return jsonify({"error": "Sin audio"}), 400

    # Parámetros ajustables desde el frontend
    VENTANA_FIORE = float(request.form.get("ventana", 4.5))  # máx. segundos para Fiore
    GAP_INICIO    = float(request.form.get("gap", 0.3))      # pausa mínima tras voz normal

    archivo = request.files["audio"]
    ext  = Path(archivo.filename).suffix or ".mp4"
    ruta = UPLOAD_FOLDER / f"{uuid.uuid4()}{ext}"
    archivo.save(ruta)

    try:
        cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        prompt_vocab = ", ".join(sorted(VOCABULARIO.keys()))

        with open(ruta, "rb") as f:
            resp = cliente.audio.transcriptions.create(
                model                   = "whisper-1",
                file                    = f,
                language                = "es",
                response_format         = "verbose_json",
                timestamp_granularities = ["word"],
                prompt                  = prompt_vocab,
            )

        palabras = []
        if hasattr(resp, "words") and resp.words:
            palabras = [
                {"word": w.word.strip().lower(), "start": w.start, "end": w.end}
                for w in resp.words
            ]

        matches = _buscar_frases(palabras)
        segmentos_out = []

        for i, m in enumerate(matches):
            entrada = VOCABULARIO.get(m["frase"], {})

            # ── Referencia: voz normal diciendo la frase ─────────────────────
            ref_dur = (m["fin"] - m["inicio"]) + 0.3
            ref_id, ref_ruta = _cortar_wav(str(ruta), m["inicio"] - 0.1, ref_dur)
            if ref_ruta.exists() and ref_ruta.stat().st_size > 100:
                _segmentos[ref_id] = {"frase": m["frase"], "ruta": str(ref_ruta)}
            else:
                ref_id = None

            # ── Fiore: audio que viene DESPUÉS de la voz normal ───────────────
            # Límite derecho = inicio de la siguiente frase detectada
            sig_inicio     = matches[i + 1]["inicio"] if i + 1 < len(matches) else m["fin"] + VENTANA_FIORE + 2
            fiore_inicio   = m["fin"] + GAP_INICIO
            fiore_fin      = min(sig_inicio - 0.2, fiore_inicio + VENTANA_FIORE)
            fiore_dur      = fiore_fin - fiore_inicio

            if fiore_dur < 0.3:
                continue

            fiore_id, fiore_ruta = _cortar_wav(str(ruta), fiore_inicio, fiore_dur)
            if not fiore_ruta.exists() or fiore_ruta.stat().st_size < 100:
                continue

            _segmentos[fiore_id] = {"frase": m["frase"], "ruta": str(fiore_ruta)}

            segmentos_out.append({
                "frase":          m["frase"],
                "inicio":         round(m["inicio"], 2),
                "fin":            round(m["fin"], 2),
                "ref_id":         ref_id,
                "fiore_id":       fiore_id,
                "fiore_inicio":   round(fiore_inicio, 2),
                "fiore_fin":      round(fiore_fin, 2),
                "duracion_fiore": round(fiore_dur, 2),
                "significado":    entrada.get("significado", ""),
                "urgencia":       entrada.get("urgencia", "BAJA"),
            })

        return jsonify({
            "texto":              resp.text[:1000],
            "total_palabras":     len(palabras),
            "frases_encontradas": len({s["frase"] for s in segmentos_out}),
            "segmentos":          segmentos_out,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        ruta.unlink(missing_ok=True)


@app.route("/calibrar/<frase>", methods=["DELETE"])
def borrar_calibracion(frase: str):
    """Elimina todas las muestras de una frase de la biblioteca DTW y Supabase."""
    from motor_acustico import cargar_biblioteca, guardar_biblioteca
    entries = cargar_biblioteca()
    nuevas = [e for e in entries if e["frase"] != frase]
    guardar_biblioteca(nuevas)
    try:
        _supabase.table("calibraciones").delete().eq("frase", frase).execute()
    except Exception:
        pass
    return jsonify({"ok": True, "frase": frase, "eliminadas": len(entries) - len(nuevas)})


@app.route("/segmento/<seg_id>")
def ruta_segmento_audio(seg_id: str):
    if seg_id not in _segmentos:
        return jsonify({"error": "Segmento no encontrado"}), 404
    return send_file(_segmentos[seg_id]["ruta"], mimetype="audio/wav")


@app.route("/segmento/<seg_id>/calibrar", methods=["POST"])
def ruta_segmento_calibrar(seg_id: str):
    if seg_id not in _segmentos:
        return jsonify({"error": "Segmento no encontrado"}), 404
    seg    = _segmentos[seg_id]
    frase  = seg["frase"]
    audio  = audio_a_numpy(seg["ruta"])
    huella = extraer_huella(audio)

    # Guardar huella en biblioteca local
    entries = cargar_biblioteca()
    entries.append({"frase": frase, "huella": huella})
    guardar_biblioteca(entries)
    n = sum(1 for e in entries if e["frase"] == frase)

    # Persistir en Supabase
    audio_url = _subir_audio_supabase(Path(seg["ruta"]), frase)
    try:
        _supabase.table("calibraciones").insert({
            "frase":     frase,
            "muestras":  n,
            "fuente":    "segmento",
            "audio_url": audio_url,
            "huella":    huella.tolist(),
        }).execute()
    except Exception as e:
        print(f"Aviso: no se pudo guardar huella en Supabase: {e}", flush=True)

    return jsonify({"ok": True, "frase": frase, "muestras": n, "listo": n >= 3})


def sincronizar_desde_supabase() -> int:
    """
    Carga las huellas directamente desde la tabla calibraciones de Supabase
    y reconstruye biblioteca_dtw.json local. Instantáneo — sin descargar audio.
    """
    try:
        resp = (
            _supabase.table("calibraciones")
            .select("frase, huella")
            .not_.is_("huella", "null")
            .execute()
        )
        if not resp.data:
            print("Sin huellas en Supabase — usando biblioteca local.", flush=True)
            return len(cargar_biblioteca())

        nuevas: list[dict] = []
        for row in resp.data:
            frase  = row["frase"]
            if frase not in VOCABULARIO:
                continue
            huella = np.array(row["huella"], dtype=np.float32)
            if huella.ndim == 2 and huella.shape[1] == 52:
                nuevas.append({"frase": frase, "huella": huella})

        guardar_biblioteca(nuevas)
        print(f"Biblioteca sincronizada: {len(nuevas)} huellas desde Supabase.", flush=True)
        return len(nuevas)

    except Exception as e:
        print(f"Error al sincronizar desde Supabase: {e}", flush=True)
        return len(cargar_biblioteca())


@app.route("/audio_respuesta/<audio_id>")
def ruta_audio_respuesta(audio_id: str):
    if audio_id not in _audios_respuesta:
        return jsonify({"error": "Audio no encontrado"}), 404
    return Response(_audios_respuesta[audio_id], mimetype="audio/mpeg")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path: str):
    dist = Path("static/react-dist")
    target = dist / path
    if path and target.exists():
        return send_from_directory(str(dist), path)
    return send_from_directory(str(dist), "index.html")


if __name__ == "__main__":
    total = sincronizar_desde_supabase()
    cal = estado_calibracion()
    calibradas = sum(1 for v in cal.values() if v >= 3)
    print(f"Servidor listo en http://localhost:5050", flush=True)
    print(f"Frases calibradas: {calibradas}/{len(VOCABULARIO)} · {total} huellas en biblioteca", flush=True)
    if calibradas == 0:
        print("→ Abre el navegador y ve a la pestaña CALIBRAR para empezar.", flush=True)
    app.run(debug=False, port=5050)

#!/bin/bash
# Setup rápido del Analista de Voz en un computador nuevo

echo "=== Analista de Voz — Setup ==="

# 1. Crear entorno virtual
python3 -m venv venv
echo "✅ Entorno virtual creado"

# 2. Instalar dependencias (torch y whisper son pesados, ~2GB)
echo "Instalando dependencias (puede tardar varios minutos)..."
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
echo "✅ Dependencias instaladas"

# 3. Crear .env si no existe
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "⚠️  Falta configurar las API keys en el archivo .env"
  echo "    Abre .env y agrega tus keys de OpenAI y ElevenLabs"
else
  echo "✅ .env ya existe"
fi

# 4. Crear carpeta de uploads
mkdir -p uploads
echo "✅ Carpeta uploads lista"

echo ""
echo "=== Todo listo ==="
echo "Para arrancar el servidor:"
echo "  venv/bin/python app.py"
echo ""
echo "Nota: la primera vez que corras el servidor descargará"
echo "el modelo Whisper large-v3 (~3GB). Solo ocurre una vez."

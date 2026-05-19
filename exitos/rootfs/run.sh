#!/usr/bin/with-contenv bashio

echo "Creem la carpeta (/share/exitos/) si no existeix, aqui guardarem fitxers persistents."
mkdir -p /share/exitos/

echo "Iniciant la connexió amb Open Remote..."
python3 openremote_service.py &

echo "Starting server.py..."
python3 server.py

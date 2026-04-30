#!/usr/bin/with-contenv bashio

echo "Creem la carpeta (/share/exitos/) si no existeix, aqui guardarem fitxers persistents."
mkdir -p /share/exitos/

echo "Instal·lant dependències de Python (paho-mqtt, requests)..."
python3 -m pip install paho-mqtt requests urllib3

echo "Iniciant la connexió amb Open Remote..."
python3 openremote_service.py &

echo "Starting server.py..."
python3 server.py

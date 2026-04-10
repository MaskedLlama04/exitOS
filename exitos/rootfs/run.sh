#!/usr/bin/with-contenv bashio

echo "Creem la carpeta (/share/exitos/) si no existeix, aqui guardarem fitxers persistents."
mkdir -p /share/exitos/

pip3 install paho-mqtt --break-system-packages > /dev/null 2>&1
echo "Starting server.py..."
python3 server.py

#!/usr/bin/with-contenv bashio

echo "Creem la carpeta (/share/exitos/) si no existeix, aqui guardarem fitxers persistents."
mkdir -p /share/exitos/

pip3 install paho-mqtt requests urllib3 --break-system-packages > /dev/null 2>&1
echo "Iniciant el Service Bridge d'OpenRemote en segon pla..."
python3 openremote_service.py &

echo "Starting server.py..."
python3 server.py

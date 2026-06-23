ARG BUILD_FROM
FROM ${BUILD_FROM}

# Instal·lem les dependències de Python a nivell de build per evitar instal·lar-les al runtime
RUN pip3 install --no-cache-dir paho-mqtt requests urllib3 --break-system-packages

#afegim els fitxers del servidor dins el container de docker
ADD ./rootfs/server.py .
ADD ./rootfs/openremote_service.py .
ADD ./rootfs/bottle.py .
ADD ./rootfs/sqlDB.py .
ADD ./rootfs/logging_config.py .
ADD ./rootfs/blockchain.py .

#Forecast i Optimizer
COPY ./rootfs/forecast /forecast/
COPY ./rootfs/optimization /optimization/
COPY ./rootfs/abstraction /abstraction/
COPY ./rootfs/abstraction/assets /assets/



#HTML and images
COPY ./rootfs/www /www/
COPY ./rootfs/images/ /images/
COPY ./rootfs/resources/ /resources/

#LLM chat
COPY ./rootfs/llm /llm/

COPY config.yaml config.yaml

# posem l'script d'inici i li donem permisos de execucio
COPY ./rootfs/run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]

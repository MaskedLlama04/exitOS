import os
import time
import requests
import logging
import urllib3

# Desactivar els warnings de certificats auto-signats HTTPS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuració del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OR-Service-Bridge")

# ======== CREDENCIALS I CONFIGURACIÓ ========
OPENREMOTE_HOST = "https://192.168.191.70"
REALM = "master"
CLIENT_ID = "exitos_ha_2"
CLIENT_SECRET = "mKNn6IXlZYunW3aDalvlulJVIg10VH9t"

SERVICE_ID = "exitos_dashboard"
SERVICE_LABEL = "Gestor eXiTOS"
SERVICE_ICON = "mdi-home-assistant"
HOMEPAGE_URL = "http://192.168.191.252:55023"

# Variable per reduir logs d'error repetitius
consecutive_token_errors = 0

def get_token():
    global consecutive_token_errors
    url = f"{OPENREMOTE_HOST}/auth/realms/{REALM}/protocol/openid-connect/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        response = requests.post(url, data=payload, verify=False, timeout=10)
        response.raise_for_status()
        if consecutive_token_errors > 0:
            logger.info("S'ha recuperat la connexió amb OpenRemote!")
            consecutive_token_errors = 0
        return response.json().get("access_token")
    except Exception as e:
        if consecutive_token_errors == 0:
            logger.error(f"Error obtenint token d'OpenRemote (revisa credencials o connectivitat): {e}")
        consecutive_token_errors += 1
        
        # Avisar només cada hora (cada 240 intents de 15 segons aprox)
        if consecutive_token_errors % 240 == 0:
             logger.error(f"Error persistent obtenint token d'OpenRemote: {e}")
        return None

def register_service(token):
    url = f"{OPENREMOTE_HOST}/api/{REALM}/service"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "serviceId": SERVICE_ID,
        "label": SERVICE_LABEL,
        "icon": SERVICE_ICON,
        "homepageUrl": HOMEPAGE_URL,
        "status": "AVAILABLE"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        instance_id = data.get("instanceId")
        logger.info(f"Servei 'Gestor eXiTOS' registrat correctament a OpenRemote! Instance ID: {instance_id}")
        return instance_id
    except Exception as e:
        logger.error(f"Error registrant el servei: {e}")
        return None

def send_heartbeat(token, instance_id):
    url = f"{OPENREMOTE_HOST}/api/{REALM}/service/{SERVICE_ID}/{instance_id}"
    headers = {
        "Authorization": f"Bearer {token}",
    }
    try:
        response = requests.put(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        logger.debug("💓 Heartbeat enviat correctament.")
        return True
    except Exception as e:
        logger.warning(f"Error al Heartbeat. Intentarem re-registrar el servei: {e}")
        return False

def main():
    logger.info("Iniciant el pont de connexió (Service Bridge) cap a OpenRemote...")
    instance_id = None
    
    while True:
        if not instance_id:
            token = get_token()
            if token:
                instance_id = register_service(token)
                if not instance_id:
                    time.sleep(15)
                    continue
            else:
                time.sleep(15)
                continue
                
        if instance_id:
            # Esperem 30 segons fins al proper heartbeat
            time.sleep(30)
            token = get_token()
            if not token or not send_heartbeat(token, instance_id):
                logger.info("S'ha perdut la connexió. Tornarem a registrar el servei en el proper cicle...")
                instance_id = None

if __name__ == "__main__":
    # Esperem una mica a que el servidor principal estigui a punt
    time.sleep(5)
    main()

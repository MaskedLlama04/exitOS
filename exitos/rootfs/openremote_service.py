import time
import requests
import logging
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OR-Service-Bridge")

# ======== CREDENCIALS I CONFIGURACIÓ ========
OPENREMOTE_HOST = "https://192.168.191.70"
REALM = "master"
CLIENT_ID = "exitos_ha_2"                   
CLIENT_SECRET = "mKNn6IXlZYunW3aDalvlulJVIg10VH9t"
SERVICE_ICON = "mdi-home-assistant"
HOMEPAGE_URL = "http://192.168.191.252:55023"

# Indicador per saber si ja hem avisat de la pèrdua de connexió (evita spam de logs)
_token_was_connected = False

def get_token():
    global _token_was_connected
    url = f"{OPENREMOTE_HOST}/auth/realms/{REALM}/protocol/openid-connect/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        response = requests.post(url, data=payload, verify=False, timeout=10)
        response.raise_for_status()
        if not _token_was_connected:
            logger.info("✅ Connexió amb OpenRemote establerta correctament.")
            _token_was_connected = True
        return response.json().get("access_token")
    except Exception:
        _token_was_connected = False
        return None

def update_remote_status(token):
    """Envia un heartbeat a OpenRemote per indicar que el servei està actiu."""
    url = f"{OPENREMOTE_HOST}/api/{REALM}/asset/service/status"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # Intentem registrar/actualitzar l'estat del servei
    payload = {
        "id": CLIENT_ID,
        "name": "Gestor eXiTOS",
        "type": "Service",
        "attributes": {
            "status": {"value": "RUNNING"},
            "icon": {"value": SERVICE_ICON},
            "url": {"value": HOMEPAGE_URL}
        }
    }
    try:
        # Enviem el heartbeat
        resp = requests.put(url, json=payload, headers=headers, verify=False, timeout=5)
        return resp.status_code in [200, 201, 204]
    except Exception as e:
        logger.debug(f"Error enviant heartbeat: {e}")
        return False

def main():
    logger.info("Iniciant el pont de connexió (Service Bridge) cap a OpenRemote...")
    retry_wait = 15
    MAX_RETRY_WAIT = 600
    
    while True:
        token = get_token()
        if not token:
            time.sleep(retry_wait)
            retry_wait = min(retry_wait * 2, MAX_RETRY_WAIT)
            continue
            
        # Si tenim token, intentem actualitzar l'estat a OpenRemote
        success = update_remote_status(token)
        if success:
            retry_wait = 15
            # Esperem 30 segons fins al proper heartbeat
            time.sleep(30)
        else:
            # Si falla el heartbeat, reintentem segons el backoff
            time.sleep(retry_wait)
            retry_wait = min(retry_wait * 2, MAX_RETRY_WAIT)

if __name__ == "__main__":
    main()

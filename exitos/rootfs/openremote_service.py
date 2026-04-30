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

    # Intentem registrar/actualitzar l'estat del servei a l'asset de la casa
    asset_id = "2ScVx3VqzFwG9PQPq4Q5b4"
    url = f"{OPENREMOTE_HOST}/api/{REALM}/asset/{asset_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # Obtenim l'asset primer per no sobreescriure-ho tot
    try:
        current_asset_resp = requests.get(url, headers=headers, verify=False, timeout=5)
        if current_asset_resp.status_code == 200:
            asset_data = current_asset_resp.json()
            # Només actualitzem l'atribut d'estat si existeix o el creem
            if "attributes" not in asset_data:
                asset_data["attributes"] = {}
            
            asset_data["attributes"]["service_status"] = {"value": "RUNNING"}
            asset_data["attributes"]["last_seen"] = {"value": int(time.time() * 1000)}
            
            resp = requests.put(url, json=asset_data, headers=headers, verify=False, timeout=5)
            return resp.status_code in [200, 204]
    except Exception as e:
        logger.debug(f"Error enviant heartbeat a l'asset: {e}")
        return False
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

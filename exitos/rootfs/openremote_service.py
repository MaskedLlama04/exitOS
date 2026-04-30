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
HOMEPAGE_URL = "http://192.168.191.252:8123/app/8e15d424_exitos"

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
    """S'assegura que l'asset del Servei existeixi a OpenRemote i l'actualitza perquè surti al menú."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 1. Busquem si el servei ja existeix
    query_url = f"{OPENREMOTE_HOST}/api/{REALM}/asset/query"
    query_payload = {
        "assetTypes": ["Service"]
    }
    
    service_asset_id = None
    try:
        query_resp = requests.post(query_url, json=query_payload, headers=headers, verify=False, timeout=5)
        if query_resp.status_code == 200:
            assets = query_resp.json()
            for asset in assets:
                if asset.get("name") == "Gestor eXiTOS":
                    service_asset_id = asset.get("id")
                    break
    except Exception as e:
        logger.debug(f"Error buscant el servei a OpenRemote: {e}")
        return False
        
    # 2. Si no existeix, el creem
    if not service_asset_id:
        create_url = f"{OPENREMOTE_HOST}/api/{REALM}/asset"
        create_payload = {
            "name": "Gestor eXiTOS",
            "type": "Service",
            "attributes": {
                "url": {"type": "text", "value": HOMEPAGE_URL},
                "icon": {"type": "text", "value": SERVICE_ICON},
                "service_status": {"type": "text", "value": "RUNNING"},
                "last_seen": {"type": "timestamp", "value": int(time.time() * 1000)}
            }
        }
        try:
            create_resp = requests.post(create_url, json=create_payload, headers=headers, verify=False, timeout=5)
            if create_resp.status_code in [200, 201]:
                logger.info("Servei 'Gestor eXiTOS' creat correctament a OpenRemote.")
                return True
        except Exception as e:
            logger.debug(f"Error creant el servei a OpenRemote: {e}")
            return False
    else:
        # 3. Si ja existeix, actualitzem el heartbeat
        update_url = f"{OPENREMOTE_HOST}/api/{REALM}/asset/{service_asset_id}"
        try:
            get_resp = requests.get(update_url, headers=headers, verify=False, timeout=5)
            if get_resp.status_code == 200:
                asset_data = get_resp.json()
                if "attributes" not in asset_data:
                    asset_data["attributes"] = {}
                
                # Assegurem que l'URL és el correcte
                asset_data["attributes"]["url"] = {"type": "text", "value": HOMEPAGE_URL}
                asset_data["attributes"]["icon"] = {"type": "text", "value": SERVICE_ICON}
                asset_data["attributes"]["service_status"] = {"type": "text", "value": "RUNNING"}
                asset_data["attributes"]["last_seen"] = {"type": "timestamp", "value": int(time.time() * 1000)}
                
                put_resp = requests.put(update_url, json=asset_data, headers=headers, verify=False, timeout=5)
                return put_resp.status_code in [200, 204]
        except Exception as e:
            logger.debug(f"Error actualitzant el heartbeat del servei: {e}")
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

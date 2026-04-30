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
    except Exception as e:
        # Silenciem els errors constants tal com es va demanar prèviament
        _token_was_connected = False
        return None

def main():
    logger.info("Iniciant el pont de connexió (Service Bridge) cap a OpenRemote...")
    retry_wait = 15
    MAX_RETRY_WAIT = 600  # Màxim 10 minuts d'espera
    
    while True:
        token = get_token()
        if not token:
            # Backoff exponencial silenciós
            time.sleep(retry_wait)
            retry_wait = min(retry_wait * 2, MAX_RETRY_WAIT)
        else:
            retry_wait = 15
            time.sleep(30)

if __name__ == "__main__":
    main()

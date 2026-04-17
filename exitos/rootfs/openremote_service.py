import os
import time
import requests
import logging
import urllib3

# Desactivar els warnings de certificats auto-signats HTTPS (per si l'OpenRemote en té)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuració del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OR-Service-Bridge")

# ======== CREDENCIALS I CONFIGURACIÓ ========
OPENREMOTE_HOST = "https://192.168.191.70"  # L'adreça on tens l'OpenRemote
REALM = "master"
CLIENT_ID = "exitos_ha_2"                   # El teu Service User
CLIENT_SECRET = "mKNn6IXlZYunW3aDalvlulJVIg10VH9t"

SERVICE_ID = "exitos_ha_dashboard"
SERVICE_LABEL = "Home Assistant"
SERVICE_ICON = "mdi-home-assistant"
HOMEPAGE_URL = "http://192.168.191.252:55023"
# ============================================

def get_token():
    url = f"{OPENREMOTE_HOST}/auth/realms/{REALM}/protocol/openid-connect/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        response = requests.post(url, data=payload, verify=False, timeout=10)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        logger.error(f"Error obtenint token d'OpenRemote (revisa credencials): {e}")
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
        logger.info(f"✅ Servei registrat correctament a OpenRemote! Instance ID: {instance_id}")
        return instance_id
    except Exception as e:
        logger.error(f"Error registrant el servei: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            logger.error(f"Detall del error d'OpenRemote: {response.text}")
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
                    time.sleep(15)  # Re-intentar si falla el registre
                    continue
            else:
                time.sleep(15)
                continue
                
        # Heartbeat loop
        if instance_id:
            time.sleep(30) # S'espera 30 segons (OpenRemote demana menys de 60s)
            
            # Demanem un token fresc just per estar segurs abans del heartbeat (per evitar que caduqui)
            token = get_token()
            if not token or not send_heartbeat(token, instance_id):
                logger.info("S'ha perdut la connexió. Tornarem a registrar el servei en el proper cicle...")
                instance_id = None # Força a tornar-se a registrar

if __name__ == "__main__":
    # Petit retard a l'arrencada per donar temps als altres serveis a pujar
    time.sleep(5)
    main()

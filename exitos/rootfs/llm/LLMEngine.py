import json
import traceback
import requests
import os
from bottle import template, request, response, request as bottle_request

# Global logger from parent (will be set in init_routes or imported if available)
logger = None

class LLMEngine:
    """
    Class to handle communication with Ollama LLM with conversation history.
    """
    def __init__(self, model=None, ollama_url=None):
        # Get model from environment variable or use default
        if model is None:
            model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        self.model = model
        # Get Ollama URL from environment variable or use default
        # Amb network_mode: host, localhost funciona directament
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        # Assegurar que la URL base no té /api/chat al final
        self.ollama_base_url = ollama_url.rstrip('/')
        self.api_url = f"{self.ollama_base_url}/api/chat"
        self.system_prompt = (
            "Ets un expert en gestió energètica de la plataforma eXiT. "
            "La teva missió és ajudar l'usuari a entendre la seva configuració d'autoconsum, "
            "optimització de bateries i generació solar. Respon de manera amable, clara i professional, "
            "preferiblement en català. Si l'usuari no coneix el tema, explica els conceptes de manera senzilla."
        )
        # Diccionari per guardar l'historial de cada sessió (per session_id)
        self.conversations = {}
        # Registre d'eines (tools)
        self.tools = {}
        
        if logger:
            logger.info(f"🔧 LLMEngine inicialitzat:")
            logger.info(f"   - Model: {self.model}")
            logger.info(f"   - URL base: {self.ollama_base_url}")
            logger.info(f"   - API endpoint: {self.api_url}")

    def register_tool(self, name, func, description, parameters):
        """
        Registra una nova eina que l'LLM pot utilitzar.
        :param name: Nom de la funció (per a l'LLM)
        :param func: Funció Python a executar
        :param description: Descripció del que fa l'eina
        :param parameters: Diccionari amb l'esquema dels paràmetres (JSON Schema)
        """
        tool_definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools[name] = {
            "definition": tool_definition,
            "func": func
        }
        if logger:
            logger.info(f"🛠️ Eina registrada: {name} - {description}")

    def get_response(self, user_input, session_id="default"):
        """
        Obté resposta d'Ollama mantenint l'historial de conversa per sessió i executant eines si cal.
        """
        try:
            if logger:
                logger.info(f"📨 Nova petició LLM per sessió: {session_id}")
                logger.info(f"   - Missatge usuari: {user_input[:50]}...")
            
            # Inicialitzar conversa si no existeix
            if session_id not in self.conversations:
                self.conversations[session_id] = [
                    {"role": "system", "content": self.system_prompt}
                ]
                if logger:
                    logger.info(f"   - Nova sessió creada amb system prompt")
            
            # Afegir missatge de l'usuari
            self.conversations[session_id].append({
                "role": "user", 
                "content": user_input
            })
            
            # Bucle per gestionar crides a eines (màxim 5 iteracions per evitar bucles infinits)
            for _ in range(5):
                # Preparar llista d'eines per a l'API
                available_tools = [t["definition"] for t in self.tools.values()] if self.tools else None

                # Preparar payload per Ollama API
                payload = {
                    "model": self.model,
                    "messages": self.conversations[session_id],
                    "stream": False
                }
                
                # Afegir eines si n'hi ha
                if available_tools:
                    payload["tools"] = available_tools

                if logger:
                    logger.info(f"🤖 Enviant petició a Ollama (Iteració eina):")
                    logger.info(f"   - Eines disponibles: {list(self.tools.keys())}")
                
                # Crida a l'API
                res = requests.post(self.api_url, json=payload, timeout=120)
                res.raise_for_status()
                data = res.json()
                
                # Processar resposta
                message = data.get("message", {})
                assistant_content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                
                # Afegir resposta (encara que sigui buida si hi ha tool_calls) a l'historial
                self.conversations[session_id].append(message)

                # Si no hi ha crides a eines, és la resposta final
                if not tool_calls:
                    if logger:
                        logger.info(f"✅ Resposta final rebuda: {assistant_content[:100]}...")
                    return assistant_content

                if logger:
                    logger.info(f"🛠️ L'LLM vol executar {len(tool_calls)} eines...")

                # Executar cada eina sol·licitada
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    
                    if function_name in self.tools:
                        if logger:
                            logger.info(f"   ▶️ Executant {function_name} amb args: {arguments}")
                        
                        try:
                            # Executar la funció
                            func = self.tools[function_name]["func"]
                            result = func(**arguments)
                            result_str = str(result)
                        except Exception as e:
                            logger.error(f"   ❌ Error executant {function_name}: {e}")
                            result_str = f"Error executing tool {function_name}: {str(e)}"
                        
                        if logger:
                            logger.info(f"   ◀️ Resultat: {result_str[:100]}...")

                        # Afegir resultat a l'historial
                        self.conversations[session_id].append({
                            "role": "tool",
                            "content": result_str,
                            # "name": function_name # Nota: Ollama a vegades no necessita 'name' en role:tool, però ajuda
                        })
                    else:
                        logger.warning(f"   ⚠️ Eina {function_name} no trobada!")
                        self.conversations[session_id].append({
                            "role": "tool",
                            "content": f"Error: Tool '{function_name}' not found."
                        })

            return "He assolit el límit d'iteracions d'eines sense resposta final."

        except requests.exceptions.ConnectionError as e:
            if logger: 
                logger.error(f"❌ Error de connexió amb Ollama a {self.api_url}: {e}")
                logger.error(f"   - Detalls: {traceback.format_exc()}")
            return f"❌ No puc connectar amb Ollama a {self.ollama_base_url}. Verifica que Ollama està executant-se i que la URL és correcta."
        except requests.exceptions.HTTPError as e:
            if logger: 
                logger.error(f"❌ Error HTTP {e.response.status_code} d'Ollama: {e}")
                logger.error(f"   - URL: {self.api_url}")
                logger.error(f"   - Response text: {e.response.text}")
                logger.error(f"   - Detalls: {traceback.format_exc()}")
            if e.response.status_code == 404:
                return f"❌ Model '{self.model}' no trobat. Assegura't que el model està descarregat a Ollama."
            elif e.response.status_code == 405:
                return f"❌ Error 405: Endpoint incorrecte. Verifica que Ollama està actualitzat i suporta /api/chat"
            else:
                return f"Error HTTP {e.response.status_code}: {e.response.text if hasattr(e.response, 'text') else str(e)}"
        except requests.exceptions.Timeout:
            if logger:
                logger.error(f"❌ Timeout esperant resposta d'Ollama")
            return "⏱️ El servidor Ollama està trigant massa. Pot ser que el model sigui massa gran o el servidor estigui ocupat."
        except requests.exceptions.RequestException as e:
            if logger: 
                logger.error(f"❌ Error connectant amb Ollama: {e}")
                logger.error(f"   - Detalls: {traceback.format_exc()}")
            return "Ho sento, no puc connectar amb el servidor Ollama. Verifica la configuració."
        except Exception as e:
            if logger: 
                logger.error(f"❌ Error inesperat al LLM: {e}")
                logger.error(traceback.format_exc())
            return "Hi ha hagut un error inesperat processant la teva consulta."
    
    def clear_conversation(self, session_id="default"):
        """
        Esborra l'historial de conversa d'una sessió.
        """
        if session_id in self.conversations:
            self.conversations[session_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
            return True
        return False

# Instància global
llm_engine = LLMEngine()

def init_routes(app, external_logger):
    global logger
    logger = external_logger
    
    if logger:
        logger.info("🔌 Inicialitzant rutes LLM...")
    
    @app.route('/llmChat')
    def llm_chat_page():
        if logger:
            logger.info("📄 Servint pàgina llmChat")
        return template('./www/llmChat.html')

    @app.route('/llm_response', method='POST')
    def llm_response():
        if logger:
            logger.info("🔵 Endpoint /llm_response cridat")
        try:
            data = request.json
            if logger:
                logger.info(f"   - Dades rebudes: {data}")
            
            if not data:
                response.status = 400
                return json.dumps({'status': 'error', 'message': 'Dades buides'})
            
            user_message = data.get('message', '')
            if not user_message:
                return json.dumps({'status': 'error', 'message': 'El missatge està buit'})
            
            # Obtenir session_id (pots usar IP, cookie o generar un ID únic)
            session_id = bottle_request.environ.get('REMOTE_ADDR', 'default')
            
            # Cridem el LLM amb historial
            response_text = llm_engine.get_response(user_message, session_id)
            
            result = json.dumps({
                'status': 'ok',
                'response': response_text
            })
            
            # Afegir headers CORS per si de cas
            response.content_type = 'application/json'
            response.headers['Access-Control-Allow-Origin'] = '*'
            
            return result
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Error en LLM response: {e}")
                logger.error(traceback.format_exc())
            return json.dumps({
                'status': 'error', 
                'message': 'Ho sento, hi ha hagut un error sol·licitant la resposta.'
            })
    
    @app.route('/llm_clear', method='POST')
    def llm_clear():
        """
        Endpoint per esborrar l'historial de conversa.
        """
        try:
            session_id = bottle_request.environ.get('REMOTE_ADDR', 'default')
            llm_engine.clear_conversation(session_id)
            return json.dumps({'status': 'ok', 'message': 'Conversa esborrada'})
        except Exception as e:
            if logger: logger.error(f" Error esborrant conversa: {e}")
            return json.dumps({'status': 'error', 'message': 'Error esborrant conversa'})
    
    # Endpoint de test per verificar connectivitat
    @app.route('/llm_test', method='GET')
    def llm_test():
        if logger:
            logger.info("🧪 Test endpoint cridat")
        return json.dumps({
            'status': 'ok',
            'message': 'LLM routes are working!',
            'ollama_url': llm_engine.ollama_base_url,
            'model': llm_engine.model
        })
    
    if logger:
        logger.info("✅ Rutes LLM registrades:")
        logger.info("   - GET  /llmChat")
        logger.info("   - POST /llm_response")
        logger.info("   - POST /llm_clear")
        logger.info("   - GET  /llm_test")


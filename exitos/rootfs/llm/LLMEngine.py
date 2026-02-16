import json
import traceback
import requests
import os
from bottle import template, request, response, request as bottle_request

# Global logger from parent (will be set in init_routes or imported if available)
logger = None

class LLMEngine:
    """
    Class to handle communication with OpenAI LLM (ChatGPT) with conversation history and tools.
    """
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        # Get API Key from env or arg
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Default model
        self.model = model
        
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
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
            logger.info(f"🔧 LLMEngine (OpenAI) inicialitzat:")
            logger.info(f"   - Model: {self.model}")
            if not self.api_key:
                logger.warning("⚠️  OPENAI_API_KEY no trobada! Les peticions fallaran.")

    def register_tool(self, name, func, description, parameters):
        """
        Registra una nova eina que l'LLM pot utilitzar.
        :param name: Nom de la funció
        :param func: Funció Python a executar
        :param description: Descripció del que fa l'eina
        :param parameters: Esquema JSON dels paràmetres
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
            logger.info(f"🛠️ Eina registrada: {name}")

    def get_response(self, user_input, session_id="default"):
        """
        Obté resposta d'OpenAI executant eines si cal.
        """
        if not self.api_key:
            return "❌ Error: Manca l'API Key d'OpenAI. Configura la variable d'entorn OPENAI_API_KEY."

        try:
            # Inicialitzar conversa
            if session_id not in self.conversations:
                self.conversations[session_id] = [
                    {"role": "system", "content": self.system_prompt}
                ]
            
            # Afegir missatge usuari
            self.conversations[session_id].append({
                "role": "user", 
                "content": user_input
            })
            
            # Bucle d'execució d'eines (màx 5 iteracions)
            for _ in range(5):
                available_tools = [t["definition"] for t in self.tools.values()] if self.tools else None
                
                payload = {
                    "model": self.model,
                    "messages": self.conversations[session_id],
                    "stream": False
                }
                if available_tools:
                    payload["tools"] = available_tools
                    payload["tool_choice"] = "auto"

                if logger:
                    logger.info(f"🤖 Enviant petició a OpenAI...")

                res = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                
                if res.status_code != 200:
                    error_msg = f"Error OpenAI {res.status_code}: {res.text}"
                    if logger: logger.error(f"❌ {error_msg}")
                    return f"❌ {error_msg}"

                data = res.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                # Afegim la resposta (o petició d'eina) a l'historial
                self.conversations[session_id].append(message)
                
                tool_calls = message.get("tool_calls")
                
                # Si no demana eines, hem acabat
                if not tool_calls:
                    content = message.get("content", "")
                    if logger: logger.info(f"✅ Resposta final: {content[:50]}...")
                    return content
                
                # Si demana eines, les executem
                if logger: logger.info(f"🛠️ Executant {len(tool_calls)} eines...")
                
                for tool_call in tool_calls:
                    fn_name = tool_call["function"]["name"]
                    fn_args = json.loads(tool_call["function"]["arguments"])
                    
                    if fn_name in self.tools:
                        if logger: logger.info(f"   ▶️ {fn_name}({fn_args})")
                        try:
                            result = self.tools[fn_name]["func"](**fn_args)
                            result_str = str(result)
                        except Exception as e:
                            result_str = f"Error: {e}"
                            if logger: logger.error(f"   ❌ Error: {e}")
                        
                        self.conversations[session_id].append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": fn_name,
                            "content": result_str
                        })
                    else:
                        self.conversations[session_id].append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": fn_name,
                            "content": f"Error: Tool {fn_name} not found"
                        })

            return "⚠️ Límit d'iteracions d'eines superat."

        except Exception as e:
            if logger: logger.error(f"❌ Error inesperat: {e}\n{traceback.format_exc()}")
            return f"Error inesperat: {e}"
    
    def clear_conversation(self, session_id="default"):
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


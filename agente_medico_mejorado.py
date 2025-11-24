import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import logging
import json
import hashlib
from functools import lru_cache
import re

# LangChain - Core
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# LangChain - Agentes y Herramientas 
from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_classic.tools import Tool
from langchain_classic import hub

# LangChain - Memoria Avanzada 
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferMemory  
from langchain_core.documents import Document  


# Utilidades
from math import sqrt
import sys

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# IL3.1 - CONFIGURACIÓN DE OBSERVABILIDAD (Logging Estructurado)
# ═══════════════════════════════════════════════════════════════

# Configurar logger con formato JSON estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MedicalAgent')

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE API - GITHUB MODELS
# ═══════════════════════════════════════════════════════════════

github_token = os.getenv("GITHUB_TOKEN")
github_base_url = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

if not github_token:
    raise ValueError("❌ GITHUB_TOKEN no configurado en .env")

# Configurar variables de entorno para LangChain
os.environ["OPENAI_API_KEY"] = github_token
os.environ["OPENAI_API_BASE"] = github_base_url

print(f"✅ API configurada: {github_base_url}\n")

# ═══════════════════════════════════════════════════════════════
# FUNCIÓN AUXILIAR - SIMILITUD COSENO (sin numpy/sklearn)
# ═══════════════════════════════════════════════════════════════

def cosine_similarity_native(vec1: List[float], vec2: List[float]) -> float:
    """Calcula la similitud coseno entre dos vectores"""
    if len(vec1) != len(vec2):
        raise ValueError("Los vectores deben tener la misma longitud")
    
    # Producto punto (dot product)
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes (normas)
    magnitude1 = sqrt(sum(a * a for a in vec1))
    magnitude2 = sqrt(sum(b * b for b in vec2))
    
    # Evitar división por cero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Similitud coseno
    return dot_product / (magnitude1 * magnitude2)


class MedicalAgentImproved:
    # Síntomas críticos que requieren atención inmediata
    CRITICAL_SYMPTOMS = [
        "dolor de pecho", "dolor en el pecho", "opresión torácica",
        "dificultad para respirar", "falta de aire", "disnea severa",
        "pérdida de conciencia", "desmayo", "síncope",
        "convulsiones", "ataque epiléptico",
        "sangrado severo", "hemorragia abundante",
        "dolor de cabeza súbito e intenso", "cefalea thunderclap",
        "parálisis facial", "pérdida de fuerza en extremidades",
        "confusión súbita", "desorientación aguda"
    ]
    
    # Palabras prohibidas
    PROHIBITED_KEYWORDS = [
        "hackear", "falsificar", "engañar", "fraude",
        "certificado falso", "receta falsa"
    ]
    
    # Disclaimer médico 
    MEDICAL_DISCLAIMER = "\n\n⚠️ **AVISO IMPORTANTE**: Esta información es solo educativa y NO reemplaza la consulta con un profesional médico. Ante cualquier duda o síntoma preocupante, consulta con tu médico."
    
    def __init__(
        self,
        pdf_directory: str,
        model: str = "gpt-4o-mini",
        use_summary_memory: bool = True,
        verbose: bool = True
    ):
        print("═══════════════════════════════════════════════════")
        print("AGENTE MÉDICO")
        print("═══════════════════════════════════════════════════")
        
        self.verbose = verbose
        self.consultation_count = 0
        self.quality_scores = []
        
        # Métricas de rendimiento
        self.metrics = {
            "consultas_totales": 0,
            "tiempo_total": 0,
            "errores": 0,
            "uso_herramientas": {},
            "alertas_emergencia": 0,
            "validaciones_seguridad": 0,
            "tiempos_respuesta": []  # Para detectar anomalías
        }
        
        # Historial de decisiones
        self.trace_history = []
        self.current_trace = []
        
        # Cache de búsquedas
        self.search_cache = {}
        self.token_usage = {"total": 0, "by_operation": {}}
        
        logger.info("Inicializando MedicalAgentImproved con RA3 integrado")
        
        # ═══════════════════════════════════════════════════
        # CONFIGURACIÓN DEL LLM
        # ═══════════════════════════════════════════════════
        print("\n[1/5] Configurando modelo de lenguaje...")
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  
            base_url=github_base_url,  
            api_key=github_token,       
            model_kwargs={"top_p": 0.9}
        )
        print(f"Modelo: {model} (temp=0.1)")
        
        # ═══════════════════════════════════════════════════
        # SISTEMA RAG - Base de Conocimiento
        # ═══════════════════════════════════════════════════
        print("\n[2/5] Cargando base de conocimiento médico...")
        self.vectorstore = self._setup_knowledge_base(pdf_directory)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # ═══════════════════════════════════════════════════
        # MEMORIA AVANZADA 
        # ═══════════════════════════════════════════════════
        print("\n[3/5] Configurando sistema de memoria...")
        if use_summary_memory:
            # ConversationSummaryMemory: Ahorra tokens, ideal para consultas largas
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
                human_prefix="Paciente",
                ai_prefix="Dr. AI"
            )
            print("Tipo: ConversationSummaryMemory (óptimo para tokens)")
        else:
            # ConversationBufferMemory: Historial completo
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            print("Tipo: ConversationBufferMemory (historial completo)")
        
        # ═══════════════════════════════════════════════════
        # HERRAMIENTAS ESPECIALIZADAS 
        # ═══════════════════════════════════════════════════
        print("\n🔧 [4/5] Creando herramientas médicas especializadas...")
        tools = self._create_advanced_medical_tools()
        print(f"{len(tools)} herramientas especializadas activas")
        
        # ═══════════════════════════════════════════════════
        # AGENTE CON PLANIFICACIÓN 
        # ═══════════════════════════════════════════════════
        print("\n[5/5] Ensamblando agente con capacidades avanzadas...")
        
        # Prompt especializado con instrucciones de planificación
        prompt = hub.pull("hwchase17/openai-functions-agent")
        
        # Personalización del sistema
        prompt.messages[0].prompt.template = self._create_system_prompt()
        
        # Crear agente con function calling
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Executor con memoria y manejo de errores
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=7,  # Permite análisis más profundos
            max_execution_time=60,  # Timeout de 60 segundos
            return_intermediate_steps=True  # Para evaluación de calidad
        )
        
        print("\n═══════════════════════════════════════════════════")
        print("✅ Sistema médico inteligente listo ✅")
        print("═══════════════════════════════════════════════════\n")
    
    # ═══════════════════════════════════════════════════════════
    # TRAZABILIDAD: Tracking de decisiones
    # ═══════════════════════════════════════════════════════════
    
    def _track_step(self, step_type: str, input_data: str, output: str, tool_used: str = None):
        """Registra cada paso del razonamiento del agente"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,  # "retrieval", "reasoning", "tool_call", "validation"
            "input": input_data[:200],  # Limitar tamaño
            "output": output[:200] if output else "",
            "tool": tool_used
        }
        self.current_trace.append(trace_entry)
        logger.debug(f"Trace step: {step_type} - Tool: {tool_used}")
    
    def _get_decision_tree(self) -> Dict:
        """Genera árbol de decisión de la última consulta"""
        return {
            "total_steps": len(self.current_trace),
            "tools_used": [t["tool"] for t in self.current_trace if t["tool"]],
            "flow": self.current_trace
        }
    
    # ═══════════════════════════════════════════════════════════
    # SEGURIDAD Y ÉTICA: Validación y sanitización
    # ═══════════════════════════════════════════════════════════
    
    def _validate_input(self, question: str) -> Tuple[bool, str]:
        """Valida que la pregunta sea apropiada y segura"""
        self.metrics["validaciones_seguridad"] += 1
        
        # Detectar palabras prohibidas
        question_lower = question.lower()
        for keyword in self.PROHIBITED_KEYWORDS:
            if keyword in question_lower:
                logger.warning(f"Input rechazado por palabra prohibida: {keyword}")
                return False, f"❌ Solicitud no permitida. Este sistema es solo para consultas médicas educativas."
        
        # Detectar patrones de datos sensibles 
        if re.search(r'\b\d{16}\b', question):  # Posible número de tarjeta
            logger.warning("Input rechazado por datos sensibles detectados")
            return False, "❌ Por seguridad, no compartas números de tarjetas u otros datos financieros."
        
        # Validar longitud razonable
        if len(question) > 2000:
            return False, "❌ La consulta es demasiado larga. Por favor, sé más conciso."
        
        if len(question.strip()) < 5:
            return False, "❌ La consulta es demasiado corta. Por favor, proporciona más detalles."
        
        self._track_step("validation", question, "Input válido", "SecurityValidator")
        return True, "OK"
    
    def _sanitize_response(self, response: str) -> str:
        """Filtra y mejora la respuesta con consideraciones éticas"""
        # Reemplazar lenguaje de certeza médica
        response = response.replace("es definitivamente", "podría ser")
        response = response.replace("tienes", "podrías tener")
        response = response.replace("sufres de", "podrías estar experimentando")
        response = response.replace("diagnóstico: ", "posible diagnóstico: ")
        
        # Agregar disclaimer si no existe
        if "AVISO IMPORTANTE" not in response and "⚠️" not in response:
            response += self.MEDICAL_DISCLAIMER
        
        self._track_step("sanitization", "raw_response", "sanitized_response", "EthicsFilter")
        return response
    
    # ═══════════════════════════════════════════════════════════
    # ESCALABILIDAD: Caching y optimización
    # ═══════════════════════════════════════════════════════════
    
    def _get_cache_key(self, query: str, search_type: str) -> str:
        """Genera clave única para cache"""
        content = f"{query}_{search_type}".lower().strip()
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cached_search(self, query: str, search_type: str, k: int = 4) -> List:
        """Búsqueda con cache para reducir costos"""
        cache_key = self._get_cache_key(query, search_type)
        
        # Verificar cache
        if cache_key in self.search_cache:
            logger.info(f"Cache HIT para {search_type}: {query[:50]}")
            self._track_step("retrieval", query, "Cache hit", f"CachedSearch_{search_type}")
            return self.search_cache[cache_key]
        
        # Realizar búsqueda real
        logger.info(f"Cache MISS para {search_type}: {query[:50]}")
        search_query = f"{search_type} {query}"
        docs = self.vectorstore.similarity_search(search_query, k=k)
        
        # Guardar en cache (limitar a 100 entradas)
        if len(self.search_cache) > 100:
            # Eliminar entrada más antigua (FIFO simple)
            self.search_cache.pop(next(iter(self.search_cache)))
        
        self.search_cache[cache_key] = docs
        self._track_step("retrieval", query, f"{len(docs)} docs found", f"VectorSearch_{search_type}")
        return docs
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimación aproximada de tokens (1 token ≈ 4 caracteres)"""
        return len(text) // 4
    
    def _log_token_usage(self, operation: str, text: str):
        """Registra uso de tokens por operación"""
        tokens = self._estimate_tokens(text)
        self.token_usage["total"] += tokens
        
        if operation not in self.token_usage["by_operation"]:
            self.token_usage["by_operation"][operation] = 0
        self.token_usage["by_operation"][operation] += tokens
    
    def _create_system_prompt(self) -> str:
        """Crea el prompt del sistema con instrucciones avanzadas"""
        return """Eres el "AI Assistant", un asistente médico especializado con capacidades avanzadas.

Tus FUNCIONES PRINCIPALES son:

1. ANÁLISIS CLÍNICO ESTRUCTURADO:
   - Descompón problemas médicos complejos en pasos lógicos
   - Usa un enfoque jerárquico: Síntomas → Síndromes → Diagnósticos
   - Realiza diagnóstico diferencial sistemático

2. BÚSQUEDA INTELIGENTE:
   - Usa herramientas especializadas para cada tipo de consulta
   - Combina múltiples fuentes antes de responder
   - Valida la calidad y relevancia de la información

3. MEMORIA CONTEXTUAL:
   - Recuerda el historial completo del paciente
   - Personaliza respuestas según conversaciones previas
   - Identifica patrones en síntomas recurrentes

4. SEGURIDAD Y ÉTICA:

PASO 1 - TRIAJE (Prioridad):
    DETECTA síntomas de emergencia (dolor de pecho, dificultad 
    respiratoria, pérdida de conciencia, etc.)
    Si detectas emergencia: ALERTA INMEDIATA de acudir a urgencias

PASO 2 - RECOPILACIÓN (Contexto):
   Identifica qué información necesitas
   Usa las herramientas apropiadas para buscar datos
   Si falta información, pregunta al paciente

PASO 3 - ANÁLISIS (Razonamiento):
   Realiza diagnóstico diferencial cuando sea apropiado
   Considera múltiples posibilidades
   Evalúa probabilidades basadas en evidencia

PASO 4 - RESPUESTA (Comunicación):
   Explica en términos simples pero precisos
   Estructura: Diagnóstico → Explicación → Tratamiento → Prevención
   Cita fuentes cuando sea posible (páginas de documentos)

RECUERDA SIEMPRE:

✓ SIEMPRE menciona que NO reemplazas atención médica profesional
✓ USA terminología médica PERO explícala en lenguaje claro
✓ SÉ empático y profesional, evita alarmar innecesariamente
✓ CITA fuentes de la base de conocimiento (páginas, documentos)
✓ Si NO tienes información suficiente, admítelo honestamente
✓ NUNCA inventes información médica o estadísticas


Tienes acceso a herramientas especializadas. Úsalas estratégicamente:
- Para síntomas: BuscarSintomas
- Para tratamientos: BuscarTratamientos
- Para medicamentos: BuscarMedicamentos y ValidarInteracciones
- Para diagnóstico complejo: AnalisisDiferencial
- Para emergencias: DetectarEmergencia
- Para evaluar calidad: EvaluarRelevancia

Recuerda: La calidad de tu respuesta depende de usar las herramientas correctas."""

    def _setup_knowledge_base(self, pdf_directory: str) -> Chroma:
        """Configura la base de conocimiento médica desde PDFs"""
        if not os.path.exists(pdf_directory):
            raise FileNotFoundError(f"Directorio no encontrado: {pdf_directory}")
        
        # Cargar documentos PDF
        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No se encontraron documentos PDF en el directorio")
        
        print(f"Cargados: {len(documents)} documentos médicos")
        
        # Chunking optimizado para documentos médicos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Chunks medianos para contexto
            chunk_overlap=150,  # Overlap generoso para continuidad
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Prioriza párrafos completos
            is_separator_regex=False
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"Fragmentos: {len(splits)} chunks procesados")
        
        # Crear embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Procesar en LOTES para evitar exceder límite de tokens
        BATCH_SIZE = 100  # Procesar 100 chunks a la vez
        
        print(f"Creando vectorstore en lotes de {BATCH_SIZE} chunks...")
        
        # Crear vectorstore con el primer lote
        first_batch = splits[:BATCH_SIZE]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            collection_name="medical_knowledge_improved"
        )
        
        print(f"   Lote 1/{(len(splits) // BATCH_SIZE) + 1} procesado ({len(first_batch)} chunks)")
        
        # Agregar los lotes restantes
        for i in range(BATCH_SIZE, len(splits), BATCH_SIZE):
            batch = splits[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(splits) // BATCH_SIZE) + 1
            print(f"   Lote {batch_num}/{total_batches} procesado ({len(batch)} chunks)")
        
        print(f"Vectorstore: Base de conocimiento indexada completamente")
        
        return vectorstore
    
    def _create_advanced_medical_tools(self) -> List[Tool]:
        tools = [
            # ═══════════════════════════════════════════════
            # HERRAMIENTAS DE BÚSQUEDA MÉDICA
            # ═══════════════════════════════════════════════
            Tool(
                name="BuscarSintomas",
                func=self._search_symptoms,
                description="""Busca información detallada sobre SÍNTOMAS y signos clínicos.
                
                Cuándo usar:
                - El paciente describe síntomas específicos
                - Necesitas información sobre manifestaciones de enfermedades
                - Quieres entender cómo se presenta una condición
                
                Input: nombre de enfermedad o síntoma
                Ejemplo: "diabetes tipo 2" o "cefalea tensional"
                
                Output: Síntomas, signos, manifestaciones clínicas con fuentes"""
            ),
            
            Tool(
                name="BuscarTratamientos",
                func=self._search_treatments,
                description="""Busca TRATAMIENTOS, terapias y manejo clínico.
                
                Cuándo usar:
                - Necesitas opciones terapéuticas para una condición
                - Quieres información sobre manejo clínico
                - El paciente pregunta sobre tratamientos disponibles
                
                Input: nombre de enfermedad o condición
                Ejemplo: "hipertensión arterial" o "gastritis crónica"
                
                Output: Tratamientos farmacológicos, no farmacológicos, manejo"""
            ),
            
            Tool(
                name="BuscarMedicamentos",
                func=self._search_medications,
                description="""Busca información completa sobre MEDICAMENTOS específicos.
                
                Cuándo usar:
                - Preguntas sobre un medicamento en particular
                - Necesitas dosis, efectos secundarios, contraindicaciones
                - Quieres información sobre uso correcto
                
                Input: nombre del medicamento
                Ejemplo: "metformina" o "omeprazol"
                
                Output: Dosis, indicaciones, efectos adversos, precauciones"""
            ),
            
            # ═══════════════════════════════════════════════
            # HERRAMIENTAS DE ANÁLISIS AVANZADO 
            # ═══════════════════════════════════════════════
            Tool(
                name="AnalisisDiferencial",
                func=self._differential_analysis,
                description="""Realiza DIAGNÓSTICO DIFERENCIAL sistemático.
                
                Cuándo usar:
                - Múltiples síntomas que podrían indicar varias patologías
                - Necesitas evaluar probabilidades diagnósticas
                - Situación clínica compleja o ambigua
                
                Input: lista de síntomas separados por comas
                Ejemplo: "fiebre, tos seca, fatiga, pérdida de olfato"
                
                Output: Diagnósticos posibles ordenados por probabilidad, 
                        características distintivas, exámenes recomendados"""
            ),
            
            Tool(
                name="ValidarInteracciones",
                func=self._validate_drug_interactions,
                description="""Valida INTERACCIONES entre medicamentos.
                
                Cuándo usar:
                - Paciente toma múltiples medicamentos
                - Necesitas verificar compatibilidad
                - Antes de recomendar un nuevo tratamiento
                
                Input: lista de medicamentos separados por comas
                Ejemplo: "warfarina, aspirina, omeprazol"
                
                Output: Interacciones conocidas, nivel de gravedad, precauciones"""
            ),
            
            # ═══════════════════════════════════════════════
            # HERRAMIENTAS DE SEGURIDAD
            # ═══════════════════════════════════════════════
            Tool(
                name="DetectarEmergencia",
                func=self._detect_emergency,
                description="""Detecta SÍNTOMAS DE EMERGENCIA que requieren atención inmediata.
                
                Cuándo usar:
                - Al inicio de TODA consulta (como triaje)
                - Síntomas potencialmente graves
                - Cuando haya duda sobre la urgencia
                
                Input: descripción de síntomas del paciente
                Ejemplo: "dolor intenso en el pecho y sudoración"
                
                Output: Nivel de urgencia (CRÍTICO/URGENTE/NORMAL) y recomendaciones"""
            ),
            
            # ═══════════════════════════════════════════════
            # HERRAMIENTA DE EVALUACIÓN DE CALIDAD
            # ═══════════════════════════════════════════════
            Tool(
                name="EvaluarRelevancia",
                func=self._evaluate_relevance,
                description="""Evalúa la RELEVANCIA y calidad de información encontrada.
                
                Cuándo usar:
                - Después de buscar información en la base de conocimiento
                - Para verificar que los resultados son pertinentes
                - Antes de dar una respuesta final importante
                
                Input: pregunta original y contexto encontrado (separados por |)
                Ejemplo: "diabetes tipo 2 | <contexto_encontrado>"
                
                Output: Score de relevancia (0-10) y recomendaciones"""
            )
        ]
        
        return tools
    
    # ═══════════════════════════════════════════════════════════
    # IMPLEMENTACIÓN DE HERRAMIENTAS
    # ═══════════════════════════════════════════════════════════
    
    def _search_symptoms(self, query: str) -> str:
        """Busca síntomas en la base de conocimiento con evaluación de calidad"""
        # Usar búsqueda con cache
        docs = self._cached_search(query, "síntomas signos manifestaciones clínicas de", k=4)
        
        # Registrar uso de herramienta
        if "BuscarSintomas" not in self.metrics["uso_herramientas"]:
            self.metrics["uso_herramientas"]["BuscarSintomas"] = 0
        self.metrics["uso_herramientas"]["BuscarSintomas"] += 1
        
        if not docs:
            return f"No se encontró información sobre síntomas de '{query}' en la base de conocimiento."
        
        # Evaluar relevancia
        relevance_score = self._calculate_relevance(query, docs)
        
        result = f"SÍNTOMAS DE '{query.upper()}':\n"
        result += f"   (Relevancia: {relevance_score:.1f}/10)\n\n"
        
        for i, doc in enumerate(docs, 1):
            result += f"─── Fuente {i} ───\n"
            result += f"{doc.page_content}\n"
            metadata = doc.metadata
            result += f"Documento: {metadata.get('source', 'N/A')} | Página: {metadata.get('page', 'N/A')}\n\n"
        
        return result
    
    def _search_treatments(self, query: str) -> str:
        """Busca tratamientos médicos"""
        # IL3.4 - Usar búsqueda con cache
        docs = self._cached_search(query, "tratamiento terapia manejo clínico de", k=4)
        
        # IL3.1 - Registrar uso de herramienta
        if "BuscarTratamientos" not in self.metrics["uso_herramientas"]:
            self.metrics["uso_herramientas"]["BuscarTratamientos"] = 0
        self.metrics["uso_herramientas"]["BuscarTratamientos"] += 1
        if "BuscarTratamientos" not in self.metrics["uso_herramientas"]:
            self.metrics["uso_herramientas"]["BuscarTratamientos"] = 0
        self.metrics["uso_herramientas"]["BuscarTratamientos"] += 1
        
        if not docs:
            return f"No se encontró información sobre tratamientos para '{query}'."
        
        relevance_score = self._calculate_relevance(query, docs)
        
        result = f"TRATAMIENTOS PARA '{query.upper()}':\n"
        result += f"   (Relevancia: {relevance_score:.1f}/10)\n\n"
        
        for i, doc in enumerate(docs, 1):
            result += f"─── Opción {i} ───\n"
            result += f"{doc.page_content}\n"
            metadata = doc.metadata
            result += f"Fuente: {metadata.get('source', 'N/A')} | Pág: {metadata.get('page', 'N/A')}\n\n"
        
        return result
    
    def _search_medications(self, query: str) -> str:
        """Busca información detallada de medicamentos"""
        # IL3.4 - Usar búsqueda con cache
        docs = self._cached_search(query, "medicamento fármaco dosis efectos secundarios contraindicaciones", k=3)
        
        # IL3.1 - Registrar uso de herramienta
        if "BuscarMedicamentos" not in self.metrics["uso_herramientas"]:
            self.metrics["uso_herramientas"]["BuscarMedicamentos"] = 0
        self.metrics["uso_herramientas"]["BuscarMedicamentos"] += 1
        
        if not docs:
            return f"No se encontró información sobre el medicamento '{query}'."
        
        result = f"INFORMACIÓN DEL MEDICAMENTO '{query.upper()}':\n\n"
        
        for i, doc in enumerate(docs, 1):
            result += f"─── Referencia {i} ───\n"
            result += f"{doc.page_content}\n"
            metadata = doc.metadata
            result += f"Fuente: {metadata.get('source', 'N/A')} | Pág: {metadata.get('page', 'N/A')}\n\n"
        
        return result
    
    def _differential_analysis(self, symptoms: str) -> str:
        """Realiza un análisis diferencial sistemático"""
        # Búsqueda amplia en la base de conocimiento
        search_query = f"diagnóstico diferencial síntomas: {symptoms}"
        docs = self.vectorstore.similarity_search(search_query, k=6)
        
        if not docs:
            return "Información insuficiente para realizar análisis diferencial."
        
        # Contexto médico
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # Prompt estructurado con planificación jerárquica
        analysis_prompt = f"""Realiza un ANÁLISIS DIFERENCIAL SISTEMÁTICO usando el siguiente enfoque jerárquico:

═══════════════════════════════════════════════════════════
SÍNTOMAS PRESENTADOS:
{symptoms}

CONTEXTO MÉDICO DISPONIBLE:
{context}
═══════════════════════════════════════════════════════════

ESTRUCTURA TU ANÁLISIS EN 3 NIVELES:

NIVEL 1 - AGRUPACIÓN SINDRÓMICA:
- Agrupa los síntomas en síndromes o sistemas afectados
- Identifica el patrón clínico dominante

NIVEL 2 - DIAGNÓSTICOS DIFERENCIALES:
- Lista 3-5 diagnósticos posibles ordenados por probabilidad
- Para cada uno, indica:
  * Características que apoyan el diagnóstico
  * Características que lo hacen menos probable
  * Prevalencia/frecuencia

NIVEL 3 - PLAN DE ACCIÓN:
- Exámenes complementarios recomendados (en orden de prioridad)
- Signos de alarma a vigilar
- Cuándo buscar atención inmediata

Sé conciso, claro y basado en evidencia."""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": "Eres un médico experto en diagnóstico diferencial y medicina interna."},
                {"role": "user", "content": analysis_prompt}
            ])
            
            return f"ANÁLISIS DIFERENCIAL:\n\n{response.content}"
            
        except Exception as e:
            return f"Error en análisis diferencial: {str(e)}"
    
    def _validate_drug_interactions(self, medications: str) -> str:
        """Valida interacciones entre medicamentos"""
        meds_list = [m.strip() for m in medications.split(",")]
        
        if len(meds_list) < 2:
            return "Se necesitan al menos 2 medicamentos para evaluar interacciones."
        
        # Buscar información sobre interacciones
        search_query = f"interacciones medicamentosas {' '.join(meds_list)}"
        docs = self.vectorstore.similarity_search(search_query, k=3)
        
        context = "\n---\n".join([doc.page_content for doc in docs]) if docs else "Sin información específica"
        
        prompt = f"""Analiza posibles INTERACCIONES MEDICAMENTOSAS entre:
{', '.join(meds_list)}

INFORMACIÓN DISPONIBLE:
{context}

Proporciona:
1. Interacciones conocidas (si las hay)
2. Nivel de gravedad (LEVE/MODERADA/GRAVE)
3. Mecanismo de interacción
4. Recomendaciones de manejo

Si no tienes información específica, indícalo claramente y sugiere consultar con un farmacéutico."""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": "Eres un farmacéutico clínico experto en interacciones medicamentosas."},
                {"role": "user", "content": prompt}
            ])
            
            return f"ANÁLISIS DE INTERACCIONES:\n\n{response.content}"
            
        except Exception as e:
            return f"Error en análisis de interacciones: {str(e)}"
    
    def _detect_emergency(self, symptoms: str) -> str:
        """
        Detecta síntomas de emergencia - SISTEMA DE TRIAJE
        """
        # Registrar uso de herramienta
        if "DetectarEmergencia" not in self.metrics["uso_herramientas"]:
            self.metrics["uso_herramientas"]["DetectarEmergencia"] = 0
        self.metrics["uso_herramientas"]["DetectarEmergencia"] += 1
        
        symptoms_lower = symptoms.lower()
        
        # Buscar síntomas críticos
        critical_found = []
        for critical in self.CRITICAL_SYMPTOMS:
            if critical in symptoms_lower:
                critical_found.append(critical)
        
        if critical_found:
            # Registrar alerta de emergencia
            self.metrics["alertas_emergencia"] += 1
            logger.warning(f"🚨 EMERGENCIA DETECTADA: {critical_found}")
            
            alert = "="*60 + "\n"
            alert += "   ¡¡¡ ALERTA DE EMERGENCIA MÉDICA !!!\n"
            alert += "="*64 + "\n\n"
            alert += f"Se detectaron síntomas CRÍTICOS:\n"
            for symptom in critical_found:
                alert += f"{symptom.upper()}\n"
            alert += "\n ACCIÓN REQUERIDA:\n"
            alert += "   → Acudir INMEDIATAMENTE a urgencias\n"
            alert += "   → NO esperar consulta programada\n"
            alert += "   → Si es necesario, llamar al 911 o emergencias\n"
            alert += "\n" + "="*64
            
            return alert
        
        # Búsqueda de señales de alarma en la base de conocimiento
        search_query = f"emergencia urgencia síntomas graves {symptoms}"
        docs = self.vectorstore.similarity_search(search_query, k=2)
        
        if docs:
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Evalúa el nivel de URGENCIA de estos síntomas:
{symptoms}

CONTEXTO MÉDICO:
{context}

Clasifica como:
- CRÍTICO: Requiere atención de emergencia inmediata
- URGENTE: Requiere atención médica en 24 horas
- NORMAL: Puede esperar consulta programada

Justifica brevemente tu clasificación."""

            try:
                response = self.llm.invoke([
                    {"role": "system", "content": "Eres un médico de urgencias experto en triaje."},
                    {"role": "user", "content": prompt}
                ])
                
                return f"EVALUACIÓN DE URGENCIA:\n\n{response.content}"
                
            except Exception as e:
                return f"No se detectaron emergencias evidentes, pero ante cualquier duda, consulte con un profesional."
        
        return "No se detectaron síntomas de emergencia evidente. Sin embargo, si los síntomas empeoran, busque atención médica."
    
    def _evaluate_relevance(self, query_and_context: str) -> str:
        """Evalúa la relevancia de la información encontrada"""
        try:
            parts = query_and_context.split("|")
            if len(parts) != 2:
                return "Formato incorrecto. Use: pregunta | contexto"
            
            query = parts[0].strip()
            context = parts[1].strip()
            
            # Evaluar con embeddings
            query_emb = self.embeddings.embed_query(query)
            context_emb = self.embeddings.embed_query(context[:1000])  # Limitar tamaño
            
            # Similitud coseno (sin numpy/sklearn)
            similarity = cosine_similarity_native(query_emb, context_emb)
            
            score = similarity * 10  # Escala 0-10
            
            if score >= 7:
                assessment = "EXCELENTE - Información muy relevante"
            elif score >= 5:
                assessment = "BUENA - Información moderadamente relevante"
            elif score >= 3:
                assessment = "REGULAR - Relevancia limitada"
            else:
                assessment = "BAJA - Considerar búsqueda adicional"
            
            return f"EVALUACIÓN DE RELEVANCIA:\n   Score: {score:.1f}/10\n   Valoración: {assessment}"
            
        except Exception as e:
            return f"Error en evaluación: {str(e)}"
    
    def _calculate_relevance(self, query: str, docs: List[Document]) -> float:
        """Calcula score de relevancia promedio"""
        if not docs:
            return 0.0
        
        try:
            query_emb = self.embeddings.embed_query(query)
            scores = []
            
            for doc in docs[:3]:  # Evaluar primeros 3 documentos
                doc_emb = self.embeddings.embed_query(doc.page_content[:500])
                similarity = cosine_similarity_native(query_emb, doc_emb)
                scores.append(similarity * 10)
            
            return sum(scores) / len(scores)
            
        except Exception:
            return 5.0  # Score neutral si hay error
    
    # ═══════════════════════════════════════════════════════════
    # INTERFAZ PRINCIPAL 
    # ═══════════════════════════════════════════════════════════
    
    def consult(self, question: str) -> Dict:
        """Procesa consulta médica con observabilidad, seguridad y trazabilidad completa"""
        start_time = datetime.now()
        self.current_trace = []  # Resetear trace para nueva consulta
        
        # OBSERVABILIDAD: Logging estructurado
        logger.info(f"Nueva consulta recibida: {question[:100]}...")
        self.metrics["consultas_totales"] += 1
        self.consultation_count += 1  # Mantener sincronizado
        self.consultation_count += 1  # Mantener sincronizado
        
        # SEGURIDAD: Validar entrada
        is_valid, validation_msg = self._validate_input(question)
        if not is_valid:
            logger.warning(f"Consulta rechazada: {validation_msg}")
            self.metrics["errores"] += 1
            return {
                "respuesta": validation_msg,
                "estado": "rechazado",
                "tiempo_respuesta": "0s",
                "calidad": 0,
                "herramientas_usadas": 0,
                "trace": self._get_decision_tree()
            }
        
        # ESCALABILIDAD: Log de tokens
        self._log_token_usage("input", question)
        
        try:
            # Procesar con el agente
            self._track_step("reasoning", question, "Iniciando procesamiento", "AgentExecutor")
            resultado = self.agent_executor.invoke({"input": question})
            
            respuesta = resultado.get("output", "Lo siento, no pude procesar tu consulta.")
            intermediate_steps = resultado.get("intermediate_steps", [])
            
            # SEGURIDAD: Sanitizar respuesta
            respuesta = self._sanitize_response(respuesta)
            
            # ESCALABILIDAD: Log de tokens de salida
            self._log_token_usage("output", respuesta)
            
            # Calcular métricas
            tiempo_respuesta = (datetime.now() - start_time).total_seconds()
            self.metrics["tiempo_total"] += tiempo_respuesta
            self.metrics["tiempos_respuesta"].append(tiempo_respuesta)  # Para detección de anomalías
            
            calidad = self._evaluate_response_quality(question, respuesta, intermediate_steps)
            self.quality_scores.append(calidad)
            
            herramientas_usadas = len(intermediate_steps)
            
            # TRAZABILIDAD: Guardar trace completo
            self.trace_history.append({
                "consulta": question,
                "timestamp": start_time.isoformat(),
                "trace": self.current_trace.copy(),
                "calidad": calidad
            })
            
            # OBSERVABILIDAD: Log estructurado completo
            log_entry = {
                "timestamp": start_time.isoformat(),
                "question_length": len(question),
                "response_length": len(respuesta),
                "tiempo_respuesta": tiempo_respuesta,
                "calidad": calidad,
                "herramientas_usadas": herramientas_usadas,
                "tools": [step[0].tool for step in intermediate_steps],
                "estado": "exitoso",
                "tokens_estimados": self._estimate_tokens(question + respuesta)
            }
            logger.info(f"Consulta completada: {json.dumps(log_entry)}")
            
            return {
                "respuesta": respuesta,
                "tiempo_respuesta": f"{tiempo_respuesta:.2f}s",
                "calidad": round(calidad, 1),
                "herramientas_usadas": herramientas_usadas,
                "estado": "exitoso",
                "trace": self._get_decision_tree(),  # Árbol de decisión
                "tokens_estimados": self._estimate_tokens(question + respuesta)  
            }
            
        except Exception as e:
            # OBSERVABILIDAD: Log de errores
            logger.error(f"Error en consulta: {str(e)}", exc_info=True)
            self.metrics["errores"] += 1
            
            error_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "respuesta": f"❌ Error al procesar la consulta: {str(e)}\n\nPor favor, intenta reformular tu pregunta.",
                "tiempo_respuesta": f"{error_time:.2f}s",
                "calidad": 0,
                "herramientas_usadas": 0,
                "estado": "error",
                "trace": self._get_decision_tree(),
                "error": str(e)
            }
    
    # Mantener método original como fallback
    def _consult_original(self, question: str) -> Dict:
        """
        Realiza una consulta al agente médico
        
        Args:
            question: Pregunta o consulta del paciente
            
        Returns:
            Dict con respuesta, metadatos y evaluación de calidad
        """
        self.consultation_count += 1
        start_time = datetime.now()
        
        try:
            # Ejecutar agente
            result = self.agent_executor.invoke({"input": question})
            
            # Calcular tiempo de respuesta
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluar calidad de la respuesta
            quality_score = self._evaluate_response_quality(
                question,
                result.get("output", ""),
                result.get("intermediate_steps", [])
            )
            
            self.quality_scores.append(quality_score)
            
            
            
            return {
                "respuesta": result["output"],
                "estado": "exitoso",
                "tiempo_respuesta": f"{response_time:.2f}s",
                "calidad": quality_score,
                "num_consulta": self.consultation_count,
                "herramientas_usadas": len(result.get("intermediate_steps", [])),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "respuesta": f"Error al procesar la consulta: {str(e)}",
                "estado": "error",
                "tiempo_respuesta": f"{(datetime.now() - start_time).total_seconds():.2f}s",
                "error_detalle": str(e)
            }
    
    def _evaluate_response_quality(
        self,
        question: str,
        response: str,
        intermediate_steps: List
    ) -> float:
        """
        Evalúa la calidad de la respuesta del agente
        
        Criterios:
        - Relevancia de la respuesta
        - Uso apropiado de herramientas
        - Completitud de la información
        """
        score = 5.0  # Score base
        
        # +2 puntos si usó herramientas (no solo LLM)
        if intermediate_steps:
            score += 2.0
        
        # +1 punto si la respuesta es suficientemente detallada
        if len(response) > 200:
            score += 1.0
        
        # +1 punto si menciona fuentes
        if "fuente" in response.lower() or "página" in response.lower():
            score += 1.0
        
        # +1 punto si incluye disclaimer/precaución
        if "importante" in response.lower() or "precaución" in response.lower():
            score += 1.0
        
        return min(score, 10.0)  # Máximo 10
    
    def get_memory_summary(self) -> str:
        """Obtiene resumen de la memoria del agente"""
        try:
            if isinstance(self.memory, ConversationSummaryMemory):
                return f"Memoria (Resumen):\n{self.memory.buffer}"
            else:
                messages = self.memory.chat_memory.messages
                return f"Memoria ({len(messages)} mensajes guardados)"
        except Exception:
            return "Memoria no disponible"
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de uso con métricas de RA3 completas (IE1, IE2, IE4)"""
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        consultas = self.metrics["consultas_totales"]
        avg_time = self.metrics["tiempo_total"] / consultas if consultas > 0 else 0
        
        # Obtener estadísticas de memoria/contexto
        memory_stats = self._get_memory_stats()
        
        # Detectar anomalías
        anomaly_report = self.get_anomalies_report()
        
        base_stats = {
            "total_consultas": consultas,
            "calidad_promedio": f"{avg_quality:.1f}/10",
            "tipo_memoria": memory_stats["tipo_memoria"],
            "modelo": self.llm.model_name,
            # OBSERVABILIDAD
            "tiempo_promedio": f"{avg_time:.2f}s",
            "errores": self.metrics["errores"],
            "tasa_exito": f"{((consultas - self.metrics['errores']) / max(1, consultas) * 100):.1f}%",
            "herramientas_mas_usadas": dict(sorted(self.metrics["uso_herramientas"].items(), key=lambda x: x[1], reverse=True)[:3]),
            "alertas_emergencia": self.metrics["alertas_emergencia"],
            # MÉTRICAS DE MEMORIA Y CONTEXTO 
            "memoria_tamano_kb": memory_stats["tamano_memoria_kb"],
            "memoria_elementos": memory_stats["elementos_en_memoria"],
            "memoria_eficiencia": memory_stats["memoria_eficiencia"],
            # ESCALABILIDAD
            "cache_size": len(self.search_cache),
            "tokens_totales": self.token_usage["total"],
            "tokens_por_consulta": self.token_usage["total"] // max(1, consultas),
            "costo_estimado_usd": f"${(self.token_usage['total'] * 0.00015 / 1000):.4f}",
            # DETECCIÓN DE ANOMALÍAS 
            "anomalias_detectadas": anomaly_report["anomalias_detectadas"],
            "estado_sistema": anomaly_report["estado_sistema"],
            "anomalias": anomaly_report["anomalias"],
            "recomendaciones": anomaly_report["recomendaciones"]
        }
        
        return base_stats
    
    def get_trace_history(self, last_n: int = 5) -> List[Dict]:
        """IL3.2 - Retorna historial de trazabilidad de las últimas consultas"""
        return self.trace_history[-last_n:] if self.trace_history else []
    
    def _detect_anomalies(self) -> List[str]:
        """IL3.1/IL3.4 - Detecta patrones anómalos en las métricas del sistema"""
        anomalies = []
        consultas = self.metrics["consultas_totales"]
        
        if consultas == 0:
            return anomalies
        
        # Calcular estadísticas
        avg_time = self.metrics["tiempo_total"] / consultas
        error_rate = self.metrics["errores"] / consultas
        tiempos = self.metrics.get("tiempos_respuesta", [])
        
        # ANOMALÍA 1: Latencia alta (tiempo promedio > 10 segundos)
        if avg_time > 10:
            anomalies.append(f"⚠️ Latencia alta detectada: {avg_time:.1f}s (normal < 10s)")
        
        # ANOMALÍA 2: Tasa de error elevada (> 20%)
        if error_rate > 0.2:
            anomalies.append(f"⚠️ Tasa de error elevada: {error_rate*100:.1f}% (normal < 20%)")
        
        # ANOMALÍA 3: Picos de latencia (alguna consulta > 15s)
        if tiempos and max(tiempos) > 15:
            anomalies.append(f"⚠️ Pico de latencia: {max(tiempos):.1f}s en última consulta")
        
        # ANOMALÍA 4: Cache ineficiente (tamaño cercano al límite)
        cache_usage = len(self.search_cache) / 100  # Límite es 100
        if cache_usage > 0.9:
            anomalies.append(f"⚠️ Cache casi lleno: {len(self.search_cache)}/100 entradas (considerar limpiar)")
        
        # ANOMALÍA 5: Uso excesivo de tokens (> 50k tokens)
        if self.token_usage["total"] > 50000:
            anomalies.append(f"⚠️ Uso alto de tokens: {self.token_usage['total']:,} tokens (costo aumentando)")
        
        # ANOMALÍA 6: Alertas de emergencia frecuentes (> 30%)
        if consultas > 5 and (self.metrics["alertas_emergencia"] / consultas) > 0.3:
            anomalies.append(f"⚠️ Alta frecuencia de emergencias: {self.metrics['alertas_emergencia']} de {consultas} consultas")
        
        return anomalies
    
    def _get_memory_stats(self) -> Dict:
        """IE2 - Obtiene estadísticas detalladas de uso de memoria y contexto"""
        try:
            # Calcular tamaño de la memoria
            if isinstance(self.memory, ConversationSummaryMemory):
                memory_size_bytes = sys.getsizeof(self.memory.buffer)
                buffer_length = len(str(self.memory.buffer))
                memory_type = "Summary (optimizada)"
            else:
                messages = self.memory.chat_memory.messages
                memory_size_bytes = sum(sys.getsizeof(str(msg)) for msg in messages)
                buffer_length = len(messages)
                memory_type = "Buffer (completa)"
            
            return {
                "tipo_memoria": memory_type,
                "tamano_memoria_kb": round(memory_size_bytes / 1024, 2),
                "elementos_en_memoria": buffer_length,
                "memoria_eficiencia": "Alta" if memory_size_bytes < 50000 else "Media" if memory_size_bytes < 100000 else "Baja"
            }
        except Exception as e:
            logger.warning(f"Error calculando estadísticas de memoria: {e}")
            return {
                "tipo_memoria": type(self.memory).__name__,
                "tamano_memoria_kb": 0,
                "elementos_en_memoria": 0,
                "memoria_eficiencia": "N/A"
            }
    
    def get_anomalies_report(self) -> Dict:
        """IE4 - Genera reporte completo de anomalías detectadas con recomendaciones"""
        anomalies = self._detect_anomalies()
        
        recommendations = []
        if any("Latencia alta" in a for a in anomalies):
            recommendations.append("💡 Considera: Optimizar búsquedas, reducir chunk_size, o usar cache más agresivo")
        
        if any("Tasa de error" in a for a in anomalies):
            recommendations.append("💡 Considera: Revisar logs, validar PDFs de entrada, o ajustar timeout")
        
        if any("Cache casi lleno" in a for a in anomalies):
            recommendations.append("💡 Acción: Ejecutar clear_cache() o aumentar límite de cache")
        
        if any("Uso alto de tokens" in a for a in anomalies):
            recommendations.append("💡 Considera: Usar ConversationSummaryMemory, reducir k en búsquedas, o limpiar memoria")
        
        return {
            "anomalias_detectadas": len(anomalies),
            "anomalias": anomalies,
            "recomendaciones": recommendations,
            "estado_sistema": "CRÍTICO" if len(anomalies) >= 3 else "ATENCIÓN" if len(anomalies) >= 1 else "NORMAL"
        }
    
    def clear_cache(self):
        """IL3.4 - Limpia el cache de búsquedas"""
        cache_size = len(self.search_cache)
        self.search_cache.clear()
        logger.info(f"Cache limpiado: {cache_size} entradas eliminadas")
        return cache_size
    
    def reset_memory(self):
        """Limpia la memoria del agente"""
        self.memory.clear()
        print("🧹 Memoria limpiada. Iniciando nueva sesión.")


# ═══════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" DEMO - Agente Médico Mejorado (Integración RA2)")
    print("="*70 + "\n")
    
    # Configuración
    PDF_DIRECTORY = "./medical_pdfs"  # Ajustar ruta según tu estructura
    
    try:
        # Inicializar agente
        agent = MedicalAgentImproved(
            pdf_directory=PDF_DIRECTORY,
            model="gpt-4o-mini",
            use_summary_memory=True,  # Usar memoria con resumen (ahorro de tokens)
            verbose=True
        )
        
        # Consulta de ejemplo
        print("\n" + "─"*70)
        print("CONSULTA 1: Síntomas generales")
        print("─"*70)
        
        resultado = agent.consult(
            "Tengo sed excesiva, orino mucho y me siento muy cansado. ¿Qué podría ser?"
        )
        
        print(f"\nRespuesta:\n{resultado['respuesta']}")
        print(f"\nMetadatos:")
        print(f"   - Tiempo: {resultado.get('tiempo_respuesta')}")
        print(f"   - Calidad: {resultado.get('calidad')}/10")
        print(f"   - Herramientas usadas: {resultado.get('herramientas_usadas')}")
        
        # Segunda consulta (con memoria)
        print("\n" + "─"*70)
        print("CONSULTA 2: Seguimiento (usa memoria)")
        print("─"*70)
        
        resultado2 = agent.consult(
            "¿Y qué tratamientos existen para eso?"
        )
        
        print(f"\nRespuesta:\n{resultado2['respuesta']}")
        
        # Estadísticas
        print("\n" + "="*70)
        print("ESTADÍSTICAS DEL AGENTE")
        print("="*70)
        stats = agent.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n" + agent.get_memory_summary())
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Asegúrate de crear una carpeta 'medical_pdfs' con documentos médicos en PDF")
    except Exception as e:
        print(f"\nError: {e}")

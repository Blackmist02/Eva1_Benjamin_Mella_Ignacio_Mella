import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# LangChain - Core
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# LangChain - Agentes y Herramientas (IL2.1)
from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_classic.tools import Tool
from langchain_classic import hub

# LangChain - Memoria Avanzada (IL2.2)
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferMemory  
from langchain_core.documents import Document  


# Utilidades
from math import sqrt

load_dotenv()

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
        
        # ═══════════════════════════════════════════════════
        # CONFIGURACIÓN DEL LLM
        # ═══════════════════════════════════════════════════
        print("\n[1/5] Configurando modelo de lenguaje...")
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # Baja temperatura = precisión médica
            base_url=github_base_url,  # AGREGAR ESTO
            api_key=github_token,       # AGREGAR ESTO
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
        # MEMORIA AVANZADA (IL2.2)
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
        # HERRAMIENTAS ESPECIALIZADAS (IL2.1)
        # ═══════════════════════════════════════════════════
        print("\n🔧 [4/5] Creando herramientas médicas especializadas...")
        tools = self._create_advanced_medical_tools()
        print(f"{len(tools)} herramientas especializadas activas")
        
        # ═══════════════════════════════════════════════════
        # AGENTE CON PLANIFICACIÓN (IL2.3)
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
        print("Sistema médico inteligente listo")
        print("═══════════════════════════════════════════════════\n")
    
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
        
        # Crear vectorstore con embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="medical_knowledge_improved"
        )
        
        print(f"Vectorstore: Base de conocimiento indexada")
        
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
            # HERRAMIENTAS DE ANÁLISIS AVANZADO (IL2.3)
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
        search_query = f"síntomas signos manifestaciones clínicas de {query}"
        docs = self.vectorstore.similarity_search(search_query, k=4)
        
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
        search_query = f"tratamiento terapia manejo clínico de {query}"
        docs = self.vectorstore.similarity_search(search_query, k=4)
        
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
        search_query = f"medicamento fármaco {query} dosis efectos secundarios contraindicaciones"
        docs = self.vectorstore.similarity_search(search_query, k=3)
        
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
        symptoms_lower = symptoms.lower()
        
        # Buscar síntomas críticos
        critical_found = []
        for critical in self.CRITICAL_SYMPTOMS:
            if critical in symptoms_lower:
                critical_found.append(critical)
        
        if critical_found:
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
        """Obtiene estadísticas del agente"""
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        
        return {
            "total_consultas": self.consultation_count,
            "calidad_promedio": f"{avg_quality:.1f}/10",
            "tipo_memoria": type(self.memory).__name__,
            "modelo": self.llm.model_name
        }
    
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

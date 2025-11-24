import streamlit as st
import os
from datetime import datetime
from agente_medico_mejorado import MedicalAgentImproved

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA PÁGINA
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Assistant - Agente Médico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# ESTILOS CSS PERSONALIZADOS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 2rem;
    }
    
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #0a0a0a;
        border-left-color: #2196F3;
    }
    
    .assistant-message {
        background-color: #0a0a0a;
        border-left-color: #4CAF50;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    
    .emergency-alert {
        background-color: #0a0a0a;
        border: 2px solid #F44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# INICIALIZACIÓN DEL ESTADO
# ═══════════════════════════════════════════════════════════════

def initialize_session_state():
    """Inicializa variables de sesión"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'consultation_count' not in st.session_state:
        st.session_state.consultation_count = 0
    
    if 'quality_scores' not in st.session_state:
        st.session_state.quality_scores = []
    
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False

initialize_session_state()

# ═══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════

def initialize_agent():
    """Inicializa el agente médico"""
    pdf_directory = st.session_state.pdf_directory
    model = st.session_state.selected_model
    use_summary = st.session_state.use_summary_memory
    
    try:
        with st.spinner("Inicializando agente médico..."):
            agent = MedicalAgentImproved(
                pdf_directory=pdf_directory,
                model=model,
                use_summary_memory=use_summary,
                verbose=False  # Sin verbose para interfaz limpia
            )
        
        st.session_state.agent = agent
        st.session_state.agent_initialized = True
        st.success("Agente médico listo para consultas!")
        
    except FileNotFoundError:
        st.error(f"Directorio no encontrado: {pdf_directory}")
        st.info("Crea la carpeta y agrega documentos médicos en PDF")
        st.session_state.agent_initialized = False
        
    except Exception as e:
        st.error(f"Error al inicializar: {str(e)}")
        st.session_state.agent_initialized = False

def process_question(question: str):
    """Procesa una consulta del usuario"""
    if not st.session_state.agent_initialized:
        st.warning("Primero debes inicializar el agente en la barra lateral")
        return
    
    # Agregar pregunta al historial
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now()
    })
    
    # Procesar con el agente
    with st.spinner("Analizando consulta..."):
        resultado = st.session_state.agent.consult(question)
    
    # Agregar respuesta al historial
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": resultado["respuesta"],
        "metadata": {
            "tiempo": resultado.get("tiempo_respuesta", "N/A"),
            "calidad": resultado.get("calidad", 0),
            "herramientas": resultado.get("herramientas_usadas", 0),
            "estado": resultado.get("estado", "desconocido"),
            "tokens_estimados": resultado.get("tokens_estimados", 0),  
            "trace": resultado.get("trace", {})  
        },
        "timestamp": datetime.now()
    })
    
    # Actualizar métricas
    st.session_state.consultation_count += 1
    st.session_state.quality_scores.append(resultado.get("calidad", 0))

def render_chat_message(message):
    """Renderiza un mensaje del chat"""
    role = message["role"]
    content = message["content"]
    timestamp = message["timestamp"].strftime("%H:%M:%S")
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong> Paciente</strong> <small>({timestamp})</small><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        metadata = message.get("metadata", {})
        calidad = metadata.get("calidad", 0)
        tiempo = metadata.get("tiempo", "N/A")
        herramientas = metadata.get("herramientas", 0)
        tokens = metadata.get("tokens_estimados", 0)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong> AI Assistant</strong> <small>({timestamp})</small><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar métricas de la respuesta 
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tiempo", tiempo)
        with col2:
            st.metric("Calidad", f"{calidad:.1f}/10")
        with col3:
            st.metric("Herramientas", herramientas)
        with col4:
            st.metric("Tokens", f"~{tokens}")
        
        


# ═══════════════════════════════════════════════════════════════
# BARRA LATERAL - CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("Configuración")
    st.markdown("---")
    
    # Configuración del agente
    st.markdown("Configuración del Agente")
    
    pdf_directory = st.text_input(
        "Directorio de PDFs",
        value="./medical_pdfs",
        help="Carpeta con documentos médicos en PDF"
    )
    st.session_state.pdf_directory = pdf_directory
    
    selected_model = st.selectbox(
        "Modelo LLM",
        options=["gpt-4o-mini"],
        index=0,
        help="Modelo de lenguaje a utilizar"
    )
    st.session_state.selected_model = selected_model
    
    use_summary_memory = st.checkbox(
        "Usar memoria con resumen",
        value=True,
        help="Ahorra tokens manteniendo solo resumen de conversaciones largas"
    )
    st.session_state.use_summary_memory = use_summary_memory
    
    # Botón de inicialización
    if st.button("Inicializar Agente", type="primary", use_container_width=True):
        initialize_agent()
    
    st.markdown("---")
    
    # Estado del agente
    if st.session_state.agent_initialized:
        st.success("Agente activo")
        
        # Estadísticas
        st.markdown("### Estadísticas")
        
        stats = st.session_state.agent.get_statistics()
        
        st.metric("Consultas", stats["total_consultas"])
        st.metric("Calidad Promedio", stats["calidad_promedio"])
        st.metric("Tipo Memoria", stats["tipo_memoria"])
        
        # Métricas
        st.markdown("---")
        st.markdown("### Métricas")
        
        # Observabilidad
        st.metric("Tiempo Promedio", stats.get("tiempo_promedio", "N/A"))
        st.metric("Tasa de Éxito", stats.get("tasa_exito", "N/A"))
        st.metric("Alertas Emergencia", stats.get("alertas_emergencia", 0))
        
        # Memoria y Contexto 
        st.markdown("---")
        st.markdown("### Memoria/Contexto")
        st.metric("Tamaño", f"{stats.get('memoria_tamano_kb', 0)} KB")
        st.metric("Elementos", stats.get("memoria_elementos", 0))
        st.metric("Eficiencia", stats.get("memoria_eficiencia", "N/A"))
        
        # Escalabilidad
        st.markdown("---")
        st.markdown("### Escalabilidad")
        st.metric("Cache Size", stats.get("cache_size", 0))
        st.metric("Tokens Totales", stats.get("tokens_totales", 0))
        st.metric("Costo Estimado", stats.get("costo_estimado_usd", "$0.00"))
        
        # Botones de acción
        st.markdown("---")
        st.markdown("### Acciones")
        
        if st.button("Limpiar Memoria", use_container_width=True):
            st.session_state.agent.reset_memory()
            st.success("Memoria limpiada")
        
        if st.button("Limpiar Cache", use_container_width=True):
            cache_size = st.session_state.agent.clear_cache()
            st.success(f"Cache limpiado: {cache_size} entradas")
        
        if st.button("Limpiar Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.consultation_count = 0
            st.session_state.quality_scores = []
            st.success("Chat limpiado")
    else:
        st.warning("Agente no inicializado")
    
    
    
    st.markdown("---")
    st.markdown("""
    <small>
    Creado por Ignacio Mella & Benjamín Mella - UwU
    </small>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONTENIDO PRINCIPAL
# ═══════════════════════════════════════════════════════════════

# Encabezado
st.markdown('<h1 class="main-header">AI Assistant - Agente Médico</h1>', unsafe_allow_html=True)

# Tabs para organizar la interfaz
tab1, tab2, tab3 = st.tabs(["Chat Médico", "Estadísticas", "Ayuda"])

# ─────────────────────────────────────────────────────────────
# TAB 1: CHAT MÉDICO
# ─────────────────────────────────────────────────────────────

with tab1:
    if not st.session_state.agent_initialized:
        st.warning("**Agente no inicializado**")
        st.info("Configura y inicializa el agente en la barra lateral para comenzar")
    else:
        # Área de chat
        st.markdown("### Conversación Médica")
        
        # Contenedor del historial
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    render_chat_message(message)
            else:
                st.info("¡Hola! Soy AI Assistant. ¿En qué puedo ayudarte hoy?")
        
        # Separador
        st.markdown("---")
        
        # Input de consulta
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_area(
                "Tu consulta médica:",
                height=100,
                placeholder="Ejemplo: Tengo dolor de cabeza y fiebre, ¿qué podría ser?",
                key="question_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Enviar", type="primary", use_container_width=True):
                if question.strip():
                    process_question(question)
                    st.rerun()
                else:
                    st.warning("Escribe una consulta primero")

# ─────────────────────────────────────────────────────────────
# TAB 2: ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Métricas de Rendimiento ")
    
    if st.session_state.agent_initialized and st.session_state.consultation_count > 0:
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Consultas",
                st.session_state.consultation_count
            )
        
        with col2:
            avg_quality = sum(st.session_state.quality_scores) / len(st.session_state.quality_scores)
            st.metric(
                "Calidad Promedio",
                f"{avg_quality:.1f}/10"
            )
        
        with col3:
            stats = st.session_state.agent.get_statistics()
            st.metric(
                "Tipo Memoria",
                stats["tipo_memoria"].replace("Conversation", "").replace("Memory", "")
            )
        
        with col4:
            st.metric(
                "Modelo",
                stats["modelo"]
            )
        
        # Métricas avanzadas
        st.markdown("---")
        st.markdown("### Métricas Detalladas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Observabilidad**")
            st.metric("Tiempo Promedio", stats.get("tiempo_promedio", "N/A"))
            st.metric("Tasa de Éxito", stats.get("tasa_exito", "N/A"))
            st.metric("Errores", stats.get("errores", 0))
        
        with col2:
            st.markdown("**Memoria/Contexto**")
            st.metric("Tamaño Memoria", f"{stats.get('memoria_tamano_kb', 0)} KB")
            st.metric("Elementos", stats.get("memoria_elementos", 0))
            st.metric("Eficiencia", stats.get("memoria_eficiencia", "N/A"))
        
        with col3:
            st.markdown("**Seguridad**")
            st.metric("Alertas Emergencia", stats.get("alertas_emergencia", 0))
            st.metric("Validaciones", stats.get("total_consultas", 0))
            
        with col4:
            st.markdown("**Escalabilidad**")
            st.metric("Tamaño Cache", stats.get("cache_size", 0))
            st.metric("Tokens Totales", stats.get("tokens_totales", 0))
        
        # Detección de Anomalías 
        st.markdown("---")
        st.markdown("### Detección de Anomalías")
        
        estado_sistema = stats.get("estado_sistema", "NORMAL")
        anomalias = stats.get("anomalias", [])
        
        
        # Badge de estado del sistema
        if estado_sistema == "CRÍTICO":
            st.error(f"🚨 **Estado del Sistema: {estado_sistema}**")
        elif estado_sistema == "ATENCIÓN":
            st.warning(f"⚠️ **Estado del Sistema: {estado_sistema}**")
        else:
            st.success(f"✅ **Estado del Sistema: {estado_sistema}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Anomalías Detectadas:**")
            if anomalias:
                for anomalia in anomalias:
                    st.warning(anomalia)
            else:
                st.info("No se detectaron anomalías")
        
        
        
        # Herramientas más usadas
        st.markdown("---")
        st.markdown("### Herramientas Más Utilizadas")
        
        if stats.get("herramientas_mas_usadas"):
            total_consultas = stats.get("total_consultas", 1)
            for tool, count in stats["herramientas_mas_usadas"].items():
                st.progress(min(count / max(total_consultas, 1), 1.0), text=f"{tool}: {count} veces")
        
        # Gráfico de calidad
        st.markdown("---")
        st.markdown("### Evolución de Calidad de Respuestas")
        
        if len(st.session_state.quality_scores) > 0:
            import plotly.graph_objects as go
            
            # Gráfico de líneas con marcadores
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.quality_scores) + 1)),
                y=st.session_state.quality_scores,
                mode='lines+markers',
                name='Calidad',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=10, color='#1E88E5', symbol='circle',
                           line=dict(color='white', width=2))
            ))
            
            fig.update_layout(
                title="Calidad de Respuestas por Consulta",
                xaxis_title="Número de Consulta",
                yaxis_title="Calidad (0-10)",
                yaxis=dict(range=[0, 10]),
                hovermode='x unified',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trazabilidad 
        st.markdown("---")
        st.markdown("### Historial de Trazabilidad")
        
        trace_history = st.session_state.agent.get_trace_history(last_n=3)
        if trace_history:
            for i, trace in enumerate(reversed(trace_history), 1):
                with st.expander(f"Consulta {len(trace_history) - i + 1}: {trace['consulta'][:50]}..."):
                    st.write(f"**Timestamp:** {trace['timestamp']}")
                    st.write(f"**Calidad:** {trace['calidad']:.1f}/10")
                    st.write(f"**Pasos ejecutados:** {len(trace['trace'])}")
                    
                    if trace['trace']:
                        st.write("**Flujo de ejecución:**")
                        for j, step in enumerate(trace['trace'], 1):
                            st.text(f"{j}. [{step['step_type']}] Tool: {step.get('tool', 'N/A')}")
        else:
            st.info("No hay historial de trazabilidad disponible")
        
        # Resumen de memoria
        st.markdown("---")
        st.markdown("### Estado de la Memoria")
        
        memory_summary = st.session_state.agent.get_memory_summary()
        st.text_area("Resumen", memory_summary, height=200, disabled=True)
        
    else:
        st.info("Las estadísticas aparecerán después de realizar consultas")

# ─────────────────────────────────────────────────────────────
# TAB 3: AYUDA
# ─────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### Guía de Uso")
    
    st.markdown("""
    ## Cómo Usar el Agente Médico 
    
    ### Inicialización
    1. Configura el **directorio de PDFs** con documentos médicos
    2. Selecciona el **modelo LLM** (gpt-4o-mini recomendado)
    3. Activa/desactiva **memoria con resumen** según necesites
    4. Haz clic en **Inicializar Agente**
    
    ### Realizar Consultas
    - Escribe tu pregunta médica en el área de texto
    - Sé específico con los síntomas
    - Puedes hacer preguntas de seguimiento (el agente recuerda el contexto)
    
    
    #### Observabilidad
    - **Logging estructurado**: Todas las consultas se registran en `medical_agent.log`
    - **Métricas de rendimiento**: Tiempo de respuesta, tasa de éxito, errores
    - **Alertas de emergencia**: Contador de síntomas críticos detectados
    - **Uso de herramientas**: Tracking de qué herramientas se usan más
    
    #### Trazabilidad
    - **Árbol de decisiones**: Cada respuesta incluye el flujo de razonamiento
    - **Historial de traces**: Revisa cómo el agente llegó a sus conclusiones
    - **Pasos intermedios**: Visualiza qué herramientas se usaron y cuándo
    - Ver en "Ver Trazabilidad" bajo cada respuesta del agente
    
    #### Seguridad y Ética
    - **Validación de entrada**: Rechaza consultas inapropiadas o datos sensibles
    - **Sanitización de respuestas**: Evita lenguaje de certeza médica absoluta
    - **Disclaimers automáticos**: Todas las respuestas incluyen aviso legal
    - **Detección de emergencias**: Sistema de triaje automático prioritario
    
    #### Escalabilidad y Sostenibilidad
    - **Caching inteligente**: Las búsquedas repetidas se sirven desde cache
    - **Optimización de tokens**: Estimación y tracking de uso de tokens
    - **Estimación de costos**: Calcula el costo aproximado de cada consulta
    - **Límite de cache**: Máximo 100 entradas para evitar sobrecarga de memoria
    
    #### Métricas de Memoria y Contexto 
    - **Tracking de memoria**: Monitorea tamaño y eficiencia de la memoria del agente
    - **Elementos en contexto**: Cuenta mensajes o resumen almacenado
    - **Eficiencia calculada**: Clasifica uso de memoria como Alta/Media/Baja
    - **Optimización automática**: Usa ConversationSummaryMemory para ahorrar memoria
    
    #### Detección de Anomalías 
    - **6 tipos de anomalías detectadas**: Latencia, errores, cache, tokens, emergencias, picos
    - **Alertas automáticas**: El sistema te avisa cuando detecta patrones inusuales
    - **Estado del sistema**: CRÍTICO/ATENCIÓN/NORMAL según anomalías encontradas
    - Ver en pestaña "Estadísticas" > "Detección de Anomalías"
    
    ### Tipos de Consultas Soportadas
    
    #### Búsqueda de Síntomas
    *"¿Cuáles son los síntomas de la diabetes?"*
    
    #### Información de Tratamientos
    *"¿Qué tratamientos existen para la hipertensión?"*
    
    #### Datos de Medicamentos
    *"¿Para qué sirve la metformina y qué efectos secundarios tiene?"*
    
    #### Análisis Diferencial
    *"Tengo fiebre, tos y dolor de cabeza. ¿Qué diagnósticos debo considerar?"*
    
    #### Interacciones Medicamentosas
    *"Tomo warfarina y aspirina, ¿hay alguna interacción?"*
    
    ### Interpretando Resultados
    
    Cada respuesta incluye:
    - **Tiempo de respuesta**: Velocidad de procesamiento
    - **Calidad**: Score de 0-10 basado en relevancia y completitud
    - **Herramientas usadas**: Número de búsquedas realizadas
    - **Tokens**: Estimación de tokens consumidos 
    - **Trazabilidad**: Árbol de decisión expandible 
    
    ### ⚠️ Importante
    
    - Este sistema es **solo educativo**
    - **NO reemplaza** consulta médica profesional
    - Ante síntomas graves, acude a **urgencias inmediatamente**
    - El agente detecta emergencias y te alertará si es necesario 
    ### Mantenimiento
    
    - **Limpiar Memoria**: Borra el contexto de conversación 
    - **Limpiar Cache**: Elimina búsquedas cacheadas 
    - **Limpiar Chat**: Reinicia completamente el historial visual
    
    ### Base de Conocimiento
    
    El agente funciona mejor con:
    - Guías clínicas actualizadas
    - Vademécums de medicamentos
    - Protocolos de tratamiento
    - Literatura médica revisada por pares
    
    Agrega PDFs médicos al directorio configurado para mejorar respuestas.
    
    ### Monitoreo y Métricas
    
    Revisa la pestaña **Estadísticas** para ver:
    - Métricas de observabilidad 
    - Historial de trazabilidad 
    - Estadísticas de seguridad 
    - Métricas de escalabilidad y costos 
    """)
    
    # Información técnica
    st.markdown("---")
    st.markdown("### Información Técnica")

    
    with st.expander("Herramientas Disponibles"):
        st.markdown("""
        1. **BuscarSintomas**: Información sobre manifestaciones clínicas 
        2. **BuscarTratamientos**: Opciones terapéuticas disponibles 
        3. **BuscarMedicamentos**: Detalles de fármacos específicos 
        4. **AnalisisDiferencial**: Diagnóstico diferencial sistemático
        5. **ValidarInteracciones**: Verificación de interacciones medicamentosas
        6. **DetectarEmergencia**: Sistema de triaje automático 
        7. **EvaluarRelevancia**: Evaluación de calidad de información
        
        **Todas las herramientas incluyen:**
        - Tracking de uso 
        - Trazabilidad de ejecución 
        - Optimización con cache 
        """)

# ═══════════════════════════════════════════════════════════════
# PIE DE PÁGINA
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <small>
    Creado por Ignacio Mella & Benjamín Mella - UwU
    </small>
</div>
""", unsafe_allow_html=True)
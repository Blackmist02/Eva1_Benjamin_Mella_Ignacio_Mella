# Arquitectura del Agente Médico Inteligente

## 📐 Visión General

Este sistema implementa un **agente médico conversacional avanzado** que combina:
- **Retrieval-Augmented Generation (RAG)** para acceso a conocimiento médico
- **Arquitectura de Agentes con LangChain** para razonamiento estructurado
- **Herramientas especializadas** para diferentes tipos de consultas
- **Sistema de memoria avanzado** para contexto conversacional
- **Evaluación automática de calidad** en cada respuesta

---

## 🏗️ Diagrama de Arquitectura
┌─────────────────────────────────────────────────────────────────┐
│ INTERFAZ WEB (Streamlit) │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Chat Médico │ │ Estadísticas │ │ Ayuda │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ AGENTE MÉDICO (LangChain Agent) │
│ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ SISTEMA DE PLANIFICACIÓN │ │
│ │ • OpenAI Functions Agent │ │
│ │ • Prompt especializado con instrucciones jerárquicas │ │
│ │ • AgentExecutor con manejo de errores │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ SISTEMA DE MEMORIA │ │
│ │ │ │
│ │ ┌────────────────────┐ ┌─────────────────────┐ │ │
│ │ │ ConversationSummary│ OR │ ConversationBuffer │ │ │
│ │ │ Memory │ │ Memory │ │ │
│ │ │ (Resumen auto) │ │ (Historial completo)│ │ │
│ │ └────────────────────┘ └─────────────────────┘ │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ SELECTOR DE HERRAMIENTAS (Function Calling) │ │
│ └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
┌────────────────────┼────────────────────┐
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ HERRAMIENTAS │ │ HERRAMIENTAS │ │ HERRAMIENTAS │
│ DE BÚSQUEDA │ │ DE ANÁLISIS │ │ DE SEGURIDAD │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • BuscarSintomas│ │ • Análisis │ │ • Detectar │
│ • Buscar │ │ Diferencial │ │ Emergencia │
│ Tratamientos │ │ • Validar │ │ • Evaluar │
│ • Buscar │ │ Interacciones │ │ Relevancia │
│ Medicamentos │ │ │ │ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │ │
└────────────────────┼────────────────────┘
▼
┌───────────────────────────────┐
│ SISTEMA RAG │
│ │
│ ┌─────────────────────────┐ │
│ │ Vector Store (Chroma) │ │
│ │ • Embeddings OpenAI │ │
│ │ • Búsqueda semántica │ │
│ └─────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────┐ │
│ │ Base de Conocimiento │ │
│ │ • PDFs médicos │ │
│ │ • Chunking optimizado │ │
│ │ • Metadata tracking │ │
│ └─────────────────────────┘ │
└───────────────────────────────┘
│
▼
┌───────────────────────────────┐
│ MODELO DE LENGUAJE │
│ GPT-4o-mini (Azure OpenAI) │
│ • Temperature: 0.1 │
│ • Function calling nativo │
└───────────────────────────────┘


---

## 🔧 Componentes Detallados

### 1. Interfaz Web (app_agente_mejorado.py)

**Tecnología**: Streamlit

**Responsabilidades**:
- Renderizado de interfaz de usuario
- Gestión de estado de sesión
- Visualización de métricas y estadísticas
- Exportación de datos

**Estructura de pestañas**:

#### Pestaña 1: Chat Médico
python
- Input de consulta del usuario
- Visualización de historial con formato
- Métricas por mensaje (tiempo, calidad, herramientas)
- Botones de acción (enviar, limpiar)

#### Pestaña 2: Estadísticas
- Métricas acumuladas (total consultas, calidad promedio)
- Gráfico de evolución de calidad
- Estado de memoria
- Información del modelO

#### Pestaña 3: Ayuda
- Guía de uso completa
- Ejemplos de consultas
- Información técnica de módulos
- Consideraciones de seguridad

#### Barra lateral
- Configuración del agente
- Selector de modelo LLM
- Tipo de memoria
- Botones de mantenimiento (limpiar memoria, limpiar chat)
- Estadísticas en tiempo real


#### Sistema RAG

PDFs médicos
    │
    ▼
PyPDFDirectoryLoader (LangChain)
    │
    ▼
RecursiveCharacterTextSplitter
    • chunk_size: 600
    • chunk_overlap: 150
    • separators: ["\n\n", "\n", ". ", " "]
    │
    ▼
Chunks con metadata
    │
    ▼
OpenAIEmbeddings (text-embedding-3-small)
    │
    ▼
Chroma Vector Store

#### Sistema de Memoria
Conversación actual:
    Usuario: "Tengo sed excesiva"
    AI: "Podría ser diabetes, requiere examen"
    Usuario: "¿Qué exámenes necesito?"
    
    ↓ (Cuando alcanza límite de tokens)
    
Resumen generado:
    "Paciente reporta sed excesiva (polidipsia). 
    Se sugirió posible diabetes mellitus. 
    Paciente pregunta sobre exámenes diagnósticos."
    
    ↓ (Continúa conversación con resumen)
    
Nuevo contexto:
    [RESUMEN PREVIO] + Conversación reciente

#### Flujo de Ejecución Completo

1. USUARIO: "Tengo sed excesiva y orino mucho, ¿qué podría ser?"
   │
   ▼
2. INTERFAZ WEB (Streamlit)
   • Captura input
   • Llama a agent.consult(question)
   │
   ▼
3. AGENT EXECUTOR
   • Lee memoria (contexto previo)
   • Analiza pregunta con LLM
   • Decide qué herramientas usar
   │
   ▼
4. PLANIFICACIÓN (GPT-4o-mini)
   "El usuario describe poliuria y polidipsia. 
   Primero debo verificar si es emergencia, 
   luego buscar síntomas relacionados."
   │
   ▼
5. EJECUCIÓN DE HERRAMIENTAS
   
   a) DetectarEmergencia("sed excesiva, orino mucho")
      → No crítico, continuar
   
   b) BuscarSintomas("poliuria polidipsia")
      → Vector search en base de conocimiento
      → Recupera 4 documentos sobre diabetes
      → Relevancia: 8.5/10
   
   c) BuscarTratamientos("diabetes")
      → Recupera información sobre manejo
   │
   ▼
6. GENERACIÓN DE RESPUESTA (GPT-4o-mini)
   • Contexto: Documentos recuperados + memoria
   • Prompt: Instrucciones del sistema + contexto + pregunta
   • Output: Respuesta estructurada
   │
   ▼
7. EVALUACIÓN AUTOMÁTICA
   • Score de calidad: 8.5/10
      - Usó herramientas ✓ (+2)
      - Respuesta detallada ✓ (+1)
      - Cita fuentes ✓ (+1)
      - Incluye precauciones ✓ (+1)
   • Tiempo: 3.2s
   • Herramientas: 3
   │
   ▼
8. ACTUALIZACIÓN DE MEMORIA
   ConversationSummaryMemory:
   "Paciente reporta sed excesiva (polidipsia) y 
   micción frecuente (poliuria). Se sugirió posible 
   diabetes mellitus tipo 2."
   │
   ▼
9. RENDERIZADO EN INTERFAZ
   • Respuesta formateada en chat
   • Métricas visualizadas (tiempo, calidad, herramientas)
   • Historial actualizado
   │
   ▼
10. LOGGING
    • Almacenamiento en session_state
    • Actualización de estadísticas
    • Preparación para exportación

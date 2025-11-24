# AI Assistant - Agente Médico Inteligente

Sistema avanzado de asistencia médica basado en IA que combina Retrieval-Augmented Generation (RAG), agentes inteligentes con herramientas especializadas y memoria conversacional para proporcionar consultas médicas informadas.

## 🎯 Características Principales

### 🤖 Arquitectura de Agentes (LangChain)
- **7 herramientas especializadas** para diferentes tipos de consultas médicas
- **Planificación jerárquica** con análisis diferencial sistemático
- **Function calling** nativo de OpenAI para selección inteligente de herramientas
- **Ejecución controlada** con timeout y manejo de errores robusto

### 🧠 Sistema de Memoria Avanzado
- **ConversationSummaryMemory**: Resúmenes automáticos para conversaciones largas (ahorro de tokens)
- **ConversationBufferMemory**: Historial completo para sesiones cortas
- **Contexto persistente**: El agente recuerda consultas previas del paciente

### 📚 Base de Conocimiento Médica (RAG)
- Procesamiento de documentos PDF médicos
- **Embeddings** con `text-embedding-3-small` de OpenAI
- **ChromaDB** como vector store
- Chunking optimizado con overlap para continuidad de contexto

### 📊 OBSERVABILIDAD 
- **Logging estructurado** en archivo `medical_agent.log`
- **Métricas de rendimiento**: Tiempo promedio, tasa de éxito, errores
- **Tracking de herramientas**: Contador de uso por tipo
- **Alertas de emergencia**: Sistema de detección automática
- **Visualización en tiempo real** en interfaz Streamlit

### 🔍 TRAZABILIDAD 
- **Árbol de decisiones**: Flujo completo de razonamiento del agente
- **Tracking de pasos**: Retrieval, reasoning, tool_call, validation
- **Historial de traces**: Últimas N consultas con metadata
- **Visualización expandible**: En cada respuesta del chat
- **Timestamp detallado**: Para auditoría y análisis

### 🔒 SEGURIDAD Y ÉTICA 
- **Validación de entrada**: Palabras prohibidas, datos sensibles
- **Detector de emergencias**: Triaje automático de síntomas críticos
- **Sanitización de respuestas**: Lenguaje médico apropiado
- **Disclaimers obligatorios**: En todas las respuestas
- **Protección de datos**: Detección de tarjetas, DNI
- **Validación de interacciones medicamentosas**

### ⚡ ESCALABILIDAD Y SOSTENIBILIDAD 
- **Sistema de cache**: LRU con máximo 100 entradas
- **Estimación de tokens**: Tracking de consumo en tiempo real
- **Cálculo de costos**: USD estimado por consulta
- **Optimización de búsquedas**: Reducción de consultas redundantes
- **Métricas de sostenibilidad**: Cache hit/miss ratio

### 🧠 MÉTRICAS DE MEMORIA Y CONTEXTO 
- **Tracking de memoria**: Tamaño en KB, elementos almacenados
- **Clasificación de eficiencia**: Alta/Media/Baja automática
- **Compatible con ambos tipos**: Summary y Buffer Memory
- **Visualización detallada**: En sidebar y estadísticas

### 🚨 DETECCIÓN DE ANOMALÍAS 
- **6 tipos de anomalías**: Latencia, errores, cache, tokens, emergencias, picos
- **Estado del sistema**: CRÍTICO/ATENCIÓN/NORMAL
- **Recomendaciones inteligentes**: Específicas por tipo de anomalía
- **Alertas automáticas**: Cuando se detectan patrones inusuales

### 🎨 Interfaz Web Completa (Streamlit)
- **3 pestañas**: Chat Médico, Estadísticas, Ayuda
- **Visualización de métricas RA3** en tiempo real
- **Gráficos interactivos** con Plotly (evolución de calidad)
- **Historial de conversación** con metadata completa
- **Dashboard de anomalías** con estado del sistema

---

## 🚀 Inicio Rápido

### Requisitos Previos
- Python 3.9+
- Token de GitHub con acceso a Azure OpenAI Models

### Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repo>
cd Eva1_Benjamin_Mella_Ignacio_Mella
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
cp .env_example .env
# Editar .env con tu GITHUB_TOKEN
```

5. **Preparar base de conocimiento**
```bash
mkdir medical_pdfs
# Copiar tus documentos médicos en PDF a esta carpeta
```

6. **Ejecutar la aplicación**
```bash
streamlit run app_agente_mejorado.py
```

La aplicación estará disponible en `http://localhost:8501`

---

## 📁 Estructura del Proyecto

```
Eva1_Benjamin_Mella_Ignacio_Mella/
│
├── app_agente_mejorado.py          # Interfaz web Streamlit
├── agente_medico_mejorado.py       # Lógica del agente con LangChain
├── ARCHITECTURE.md                  # Documentación detallada de arquitectura
├── README.md                        # Este archivo
├── .env_example                     # Plantilla de configuración
├── .gitignore                       # Archivos ignorados por Git
├── requirements.txt                 # Dependencias del proyecto
│
└── medical_pdfs/                    # Base de conocimiento (no incluida)
    ├── diabetes.pdf
    ├── hipertension.pdf
    └── ...
```

---

## 🛠️ Tecnologías Utilizadas

### Core
- **Python 3.9+**: Lenguaje principal
- **Streamlit**: Framework para interfaz web
- **LangChain**: Framework para agentes y RAG
- **OpenAI GPT-4o-mini**: Modelo de lenguaje

### LangChain Components
- `langchain-openai`: Integración con OpenAI
- `langchain-chroma`: Vector store
- `langchain-community`: Cargadores de documentos
- `langchain-text-splitters`: Procesamiento de texto

### Utilidades
- `python-dotenv`: Gestión de variables de entorno
- `PyPDF`: Procesamiento de documentos PDF

---

## 🧩 Componentes Principales

### 1. Herramientas Médicas Especializadas

#### `BuscarSintomas`
Busca información sobre síntomas y manifestaciones clínicas de enfermedades.

**Ejemplo de uso:**
```python
"¿Cuáles son los síntomas de la diabetes tipo 2?"
```

#### `BuscarTratamientos`
Recupera opciones terapéuticas para condiciones médicas.

**Ejemplo de uso:**
```python
"¿Qué tratamientos existen para la hipertensión arterial?"
```

#### `BuscarMedicamentos`
Proporciona información detallada sobre medicamentos específicos.

**Ejemplo de uso:**
```python
"¿Para qué sirve la metformina y qué efectos secundarios tiene?"
```

#### `AnalisisDiferencial`
Realiza diagnóstico diferencial sistemático basado en síntomas.

**Ejemplo de uso:**
```python
"Tengo fiebre, tos seca y fatiga. ¿Qué diagnósticos debo considerar?"
```

#### `ValidarInteracciones`
Verifica interacciones entre múltiples medicamentos.

**Ejemplo de uso:**
```python
"Tomo warfarina y aspirina, ¿hay alguna interacción?"
```

#### `DetectarEmergencia`
Sistema de triaje que identifica síntomas críticos.

**Ejemplo de uso:**
```python
"Tengo dolor en el pecho y dificultad para respirar"
# → ALERTA DE EMERGENCIA
```

#### `EvaluarRelevancia`
Evalúa la calidad de información recuperada de la base de conocimiento.

---

### 2. Sistema de Memoria

**ConversationSummaryMemory** (Recomendado)
- Genera resúmenes automáticos de conversaciones largas
- Ahorra tokens manteniendo contexto esencial
- Ideal para consultas médicas con múltiples seguimientos

**ConversationBufferMemory**
- Almacena historial completo
- Mayor consumo de tokens
- Útil para sesiones cortas

---

### 3. Evaluación de Calidad

Cada respuesta es evaluada con múltiples criterios:

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Uso de herramientas | +2 pts | Utilizó búsqueda en base de conocimiento |
| Detalle de respuesta | +1 pt | Respuesta > 200 caracteres |
| Citas de fuentes | +1 pt | Menciona documentos/páginas |
| Precauciones | +1 pt | Incluye disclaimers médicos |

**Score final: 0-10**

---

## 📊 Métricas Monitoreadas

### Rendimiento y Observabilidad
- ⏱️ **Tiempo de respuesta**: Velocidad de procesamiento (promedio y por consulta)
- 🔧 **Herramientas usadas**: Número y tipo de búsquedas realizadas
- 📈 **Consultas totales**: Contador de sesión acumulativo
- ✅ **Tasa de éxito**: Porcentaje de consultas exitosas vs errores
- 🚨 **Alertas de emergencia**: Contador de síntomas críticos detectados

### Trazabilidad
- 🔍 **Árbol de decisión**: Flujo completo paso a paso
- 📝 **Historial de traces**: Últimas 5 consultas con detalle
- 🕐 **Timestamps**: Marca temporal de cada operación
- 🛠️ **Herramientas por paso**: Qué tool se usó en cada momento

### Escalabilidad y Costos
- 💾 **Cache size**: Número de búsquedas cacheadas
- 🎫 **Tokens totales**: Consumo acumulado de tokens
- 💰 **Costo estimado**: USD ($0.00015 por 1K tokens)
- 📊 **Tokens por consulta**: Promedio de consumo

### Memoria y Contexto
- 🧠 **Tamaño de memoria**: KB utilizados por el agente
- 📦 **Elementos en memoria**: Número de mensajes/caracteres
- 🚀 **Eficiencia**: Clasificación Alta/Media/Baja

### Anomalías del Sistema
- ⚠️ **Anomalías detectadas**: Número y tipo
- 🎯 **Estado del sistema**: CRÍTICO/ATENCIÓN/NORMAL
- 💡 **Recomendaciones**: Acciones sugeridas automáticamente

### Calidad 
- ⭐ **Score de calidad**: Evaluación automática (0-10)
- 🎯 **Relevancia**: Similitud coseno con consulta
- 📚 **Cobertura**: Documentos utilizados

---

##  Uso de la Interfaz Web

### Pestaña 1: Chat Médico

1. **Inicializar agente** (barra lateral):
   - Configurar directorio de PDFs
   - Seleccionar modelo LLM (gpt-4o-mini)
   - Activar/desactivar memoria con resumen (recomendado: activado)
   - Hacer clic en "Inicializar Agente"

2. **Realizar consulta**:
   - Escribir pregunta médica en el área de texto
   - Hacer clic en "Enviar"
   - Ver respuesta con métricas:
     * ⏱️ Tiempo de respuesta
     * ⭐ Calidad (0-10)
     * 🔧 Herramientas usadas
     * 🎫 Tokens estimados

3. **Seguimiento**:
   - El agente recuerda conversaciones previas
   - Puedes hacer preguntas de contexto sin repetir información



### Pestaña 2: Estadísticas

**Métricas principales:**
- Total de consultas realizadas
- Calidad promedio de respuestas
- Tipo de memoria utilizada
- Modelo LLM activo

**IL3.1 - Observabilidad:**
- ⏱️ Tiempo promedio de respuesta
- ✅ Tasa de éxito (%)
- ❌ Número de errores

**Memoria/Contexto:**
- 💾 Tamaño de memoria (KB)
- 📊 Elementos almacenados
- 🚀 Eficiencia calculada

**Escalabilidad:**
- 💾 Tamaño del cache
- 🎫 Tokens totales consumidos
- 💰 Costo estimado en USD

**Detección de Anomalías:**
- 🚨 Estado del sistema (CRÍTICO/ATENCIÓN/NORMAL)
- ⚠️ Anomalías detectadas con descripción detallada
- 💡 Recomendaciones automáticas para solucionar problemas

**Visualizaciones:**
- 📈 Gráfico de evolución de calidad (líneas con marcadores)
- 🔧 Barras de progreso de herramientas más usadas
- 🔍 Historial de trazabilidad (últimas 3 consultas)

### Pestaña 3: Ayuda

- Guía completa de uso del sistema
- Tipos de consultas soportadas
- Información técnica sobre módulos 
- Ejemplos de preguntas
- Documentación de funcionalidades :
  * Observabilidad y logging
  * Trazabilidad de decisiones
  * Seguridad y ética
  * Escalabilidad y caching
  * Tracking de memoria/contexto
  * Detección de anomalías

### Barra Lateral

**Métricas en tiempo real:**
- Consultas totales
- Calidad promedio
- Tipo de memoria
- Tiempo promedio
- Tasa de éxito
- Alertas de emergencia
- Tamaño de memoria (KB)
- Elementos en memoria
- Eficiencia de memoria
- Cache size
- Tokens totales
- Costo estimado

**Acciones disponibles:**
- Limpiar Memoria
- Limpiar Cache
- Limpiar Chat

---

## Casos de Uso

### 1. Consulta Básica 

**Pregunta:** *"¿Cuáles son los síntomas de la diabetes tipo 2?"*

**Flujo del sistema:**
1. **Recepción** Log de consulta con timestamp
2. **Planificación** Agente identifica que necesita buscar información
3. **Herramienta RAG** Busca en PDFs médicos sobre diabetes
4. **Trazabilidad** Registra cada paso (pensamiento → herramienta → resultado)
5. **Respuesta**: Genera respuesta basada en fuentes
6. **Memoria** Calcula tamaño, elementos y eficiencia del contexto
7. **Observabilidad** Métricas de tiempo, tokens, calidad

**Resultado visible:**
- Respuesta completa con métricas
- Trazabilidad expandible en interfaz
- Memoria actualizada en sidebar

---

### 2. Conversación Multi-Turn 

**Secuencia:**
1. *"¿Qué es la hipertensión?"*
2. *"¿Cuáles son sus tratamientos?"*
3. *"¿Hay contraindicaciones?"*

**Flujo del sistema:**
1. **Primera consulta**: Inicializa memoria conversacional
2. **Segunda consulta**: Usa contexto previo sin repetir búsqueda de hipertensión
3. **Tercera consulta**: Mantiene todo el hilo conversacional
4. **IE2**: Tracking continuo del crecimiento de memoria
5. **IE4**: Detección de anomalías si memoria crece excesivamente

**Resultado visible:**
- Respuestas coherentes sin repetir contexto
- Métricas de memoria creciendo gradualmente
- Alertas si se detectan anomalías en el sistema

---

### 3. Detección de Anomalías 

**Escenario:** Sistema bajo carga con múltiples consultas

**Anomalías detectadas automáticamente:**
1. **Latencia alta**: Respuesta > 10 segundos
2. **Tasa de errores**: > 30% de consultas fallidas
3. **Picos de respuesta**: Respuestas > 15 segundos
4. **Cache ineficiente**: < 30% de hits
5. **Consumo alto de tokens**: > 50,000 tokens
6. **Alertas frecuentes**: > 3 alertas de emergencia

**Resultado visible:**
- Estado del sistema: 🟢 NORMAL / 🟡 ATENCIÓN / 🔴 CRÍTICO
- Lista de anomalías específicas detectadas
- Recomendaciones automáticas:
  * "Optimizar consultas para reducir latencia"
  * "Revisar configuración de cache"
  * "Verificar calidad de datos médicos"

---

### 4. Optimización de Recursos 

**Escenario:** Uso del sistema de cache y tracking de memoria

**Flujo del sistema:**
1. **Primera consulta**: Respuesta lenta (sin cache)
2. **Segunda consulta similar**: Respuesta rápida (hit en cache)

**Resultado visible:**
- Cache size creciente en sidebar
- Tokens consumidos por consulta
- Costo estimado en USD
- Eficiencia de memoria (Alta/Media/Baja)

---

### 5. Trazabilidad Completa 

**Flujo paso a paso de una consulta:**

```
PENSAMIENTO 1
"Necesito buscar información sobre hipertensión arterial"

HERRAMIENTA: busqueda_rag
   Input: {"query": "hipertensión arterial definición causas"}

RESULTADO
"Encontrados 4 documentos relevantes sobre hipertensión..."

PENSAMIENTO 2
"Tengo suficiente información para responder"

RESPUESTA FINAL
"La hipertensión arterial es..."
```

**Resultado visible:**
- Expandible en cada respuesta del chat
- Útil para debugging y auditoría
- Transparencia en decisiones del agente

---

### 6. Seguridad y Ética 

**Validaciones automáticas:**

1. **Consultas sensibles:**
   - *"¿Puedo autodiagnosticarme cáncer?"*
   - **Sistema**: Alerta ética + advertencia
   - **Respuesta**: "No puedo proporcionar diagnósticos. Consulta a un médico profesional."

2. **Contenido inapropiado:**
   - Detecta palabras clave peligrosas
   - Registra alertas en estadísticas
   - Muestra advertencia en interfaz

3. **Datos personales:**
   - No almacena información sensible
   - Advertencia sobre privacidad en ayuda

**Resultado visible:**
- Contador de alertas de emergencia en sidebar
- Estadísticas de seguridad en pestaña de Estadísticas

---

### Ejemplos adicionales de casos:

**Consulta con emergencia:**
```
Usuario: "Tengo sed excesiva y orino mucho, ¿qué podría ser?"

Agente:
1. Ejecuta DetectarEmergencia (no crítico)
2. Usa BuscarSintomas("poliuria polidipsia")
3. Genera respuesta basada en contexto
4. Incluye disclaimer médico

Respuesta: "Los síntomas que describes (poliuria y polidipsia) 
son característicos de la diabetes mellitus. [Más detalles...]
Es importante que consultes con un médico para confirmar..."
```

**Análisis Diferencial:**
```
Usuario: "Tengo fiebre, tos, dolor de cabeza y fatiga"

Agente:
1. Ejecuta DetectarEmergencia
2. Usa AnalisisDiferencial(síntomas)
3. Estructura respuesta jerárquica:
   - Nivel 1: Síndrome respiratorio febril
   - Nivel 2: Diagnósticos posibles (COVID-19, gripe, neumonía)
   - Nivel 3: Exámenes recomendados
```

**Validación de Medicamentos:**
```
Usuario: "Tomo warfarina y quiero tomar ibuprofeno"

Agente:
1. Usa BuscarMedicamentos("warfarina")
2. Usa BuscarMedicamentos("ibuprofeno")
3. Ejecuta ValidarInteracciones("warfarina, ibuprofeno")
4. ALERTA: Interacción GRAVE detectada
5. Recomienda alternativas
```

---


## Evaluación y Testing

### Dataset de Evaluación Incluido

El archivo [`agente_medico_mejorado.py`](agente_medico_mejorado.py) incluye un script de ejemplo:

```python
python agente_medico_mejorado.py
```

Esto ejecutará:
- Inicialización del agente
- 2 consultas de ejemplo
- Evaluación de calidad
- Estadísticas de rendimiento


Creado por:
- **Ignacio Mella**
- **Benjamín Mella**

---
Para documentación técnica un poco mas detallada [`Arquitectura.md`](Arquitectura.md)

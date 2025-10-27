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

### 🔒 Sistema de Seguridad
- **Detector de emergencias** con triaje automático
- Alertas críticas para síntomas que requieren atención inmediata
- Validación de interacciones medicamentosas
- Disclaimers automáticos en cada respuesta

### 📊 Evaluación de Calidad
- **Scoring automático** de respuestas (0-10)
- Métricas de relevancia con similitud coseno
- Tracking de herramientas utilizadas
- Medición de tiempos de respuesta

### 🎨 Interfaz Web Completa (Streamlit)
- **3 pestañas**: Chat Médico, Estadísticas, Ayuda
- Visualización de métricas en tiempo real
- Historial de conversación con metadata
- Exportación de estadísticas

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

### Rendimiento
- ⏱️ **Tiempo de respuesta**: Velocidad de procesamiento
- 🔧 **Herramientas usadas**: Número de búsquedas realizadas
- 📈 **Consultas totales**: Contador de sesión

### Calidad
- ⭐ **Score de calidad**: Evaluación automática (0-10)
- 🎯 **Relevancia**: Similitud coseno con consulta
- 📚 **Cobertura**: Documentos utilizados

---

##  Uso de la Interfaz Web

### Pestaña 1: Chat Médico

1. **Inicializar agente** (barra lateral):
   - Configurar directorio de PDFs
   - Seleccionar modelo LLM
   - Activar/desactivar memoria con resumen
   - Hacer clic en "Inicializar Agente"

2. **Realizar consulta**:
   - Escribir pregunta médica en el área de texto
   - Hacer clic en "Enviar"
   - Ver respuesta con métricas

3. **Seguimiento**:
   - El agente recuerda conversaciones previas
   - Puedes hacer preguntas de contexto sin repetir información

### Pestaña 2: Estadísticas

- Visualiza métricas acumuladas
- Gráfico de evolución de calidad
- Estado de la memoria
- Estadísticas del modelo

### Pestaña 3: Ayuda

- Guía completa de uso
- Tipos de consultas soportadas
- Información técnica
- Ejemplos de preguntas

---

## Casos de Uso

### Consulta Básica
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

### Análisis Diferencial
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

### Validación de Medicamentos
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


## 🧪 Evaluación y Testing

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

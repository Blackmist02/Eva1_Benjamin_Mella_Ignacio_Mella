# Eva1_Benjamin_Mella_Ignacio_Mella

## Sistema RAG (Retrieval-Augmented Generation) con Evaluación Médica

Este proyecto implementa un sistema RAG completo con capacidades de evaluación y monitoreo para consultas médicas, utilizando Streamlit, OpenAI GPT-4o, y LangChain.

### Funcionalidades Principales

#### 1. Sistema RAG Híbrido
- Combina búsqueda semántica (70%) y por palabras clave (30%)
- Utiliza embeddings de OpenAI (`text-embedding-3-small`) via LangChain
- Base de conocimiento médico con 8 documentos sobre condiciones comunes

#### 2. Evaluación Automática
- **Faithfulness**: Evalúa si la respuesta es fiel al contexto (1-10)
- **Relevance**: Mide qué tan relevante es la respuesta (1-10)
- **Context Precision**: Calcula precisión de documentos recuperados

#### 3. Interfaz Web Completa
5 pestañas principales:
- 🔍 **Consulta**: Interfaz para hacer preguntas médicas
- 📄 **Documentos**: Gestión de base de conocimiento
- 📊 **Métricas**: Dashboard con visualizaciones
- 🧪 **Evaluación**: Tests sistemáticos con dataset predefinido
- 📈 **Analytics**: Exportación de datos y estadísticas

### 🔧 Tecnologías Utilizadas

- **Streamlit**: Framework para la interfaz web
- **OpenAI GPT-4o**: Modelo de lenguaje para generación y evaluación
- **LangChain**: Framework para embeddings y manejo de documentos
- **scikit-learn**: Cálculo de similitud coseno
- **Plotly**: Visualización de datos interactiva
- **Pandas/NumPy**: Manipulación y análisis de datos

### Requisitos

```bash
pip install streamlit openai langchain-openai scikit-learn plotly pandas numpy python-dotenv
```

### Configuración

1. Crear archivo `.env`:
```env
GITHUB_TOKEN=your_github_token_here
GITHUB_BASE_URL=https://models.inference.ai.azure.com
```

2. Ejecutar la aplicación:
```bash
streamlit run rag.py
```

### Arquitectura del Sistema

#### Documentos Médicos Incluidos
- Diabetes tipo 2
- Hipertensión arterial
- Anemia
- Antibióticos
- Vacunación
- Infarto de miocardio
- Neumonía
- Depresión

### Documentación de Funciones

#### Inicialización y Configuración

**`initialize_client()`**
- Inicializa el cliente OpenAI con token de GitHub
- Configura la URL base para Azure OpenAI
- Retorna cliente configurado o None si hay error

**`initialize_embeddings()`**
- Inicializa el modelo de embeddings de LangChain
- Utiliza `OpenAIEmbeddings` con modelo `text-embedding-3-small`
- Maneja errores de autenticación y conexión

#### Generación de Embeddings

**`get_embeddings_langchain(embeddings_model, texts)`**
- Convierte lista de textos a embeddings vectoriales
- Transforma textos a objetos `Document` de LangChain
- Retorna array NumPy con embeddings o None si hay error

**`get_query_embedding_langchain(embeddings_model, query)`**
- Genera embedding para una consulta específica
- Optimizado para búsqueda semántica
- Maneja excepciones y retorna vector de consulta

#### Sistema de Búsqueda

**`hybrid_search_with_metrics(query, documents, embeddings, embeddings_model, client, top_k=5)`**
- Implementa búsqueda híbrida: semántica (70%) + palabras clave (30%)
- Calcula similitud coseno para búsqueda semántica
- Implementa scoring por intersección de palabras
- Retorna documentos rankeados con scores y tiempo de recuperación

#### Generación de Respuestas

**`generate_response_with_metrics(client, query, context_docs)`**
- Genera respuestas usando GPT-4o basándose en contexto recuperado
- Combina múltiples documentos como contexto unificado
- Mide tiempo de generación y maneja errores
- Retorna respuesta y métricas de tiempo

#### Sistema de Evaluación

**`evaluate_faithfulness(client, query, context, response)`**
- Evalúa si la respuesta es fiel al contexto proporcionado
- Utiliza GPT-4o como evaluador automático
- Escala 1-10 donde 7-10 indica alta fidelidad
- Prompt especializado para evaluación médica

**`evaluate_relevance(client, query, response)`**
- Mide qué tan relevante es la respuesta para la consulta
- Evaluación automática con GPT-4o
- Escala 1-10 para relevancia de contenido médico
- Considera utilidad práctica de la respuesta

**`evaluate_context_precision(client, query, retrieved_docs)`**
- Calcula precisión de documentos recuperados
- Evalúa cuántos documentos son realmente relevantes
- Retorna ratio de documentos útiles vs total recuperado
- Utiliza evaluación automática por documento

#### Dataset de Evaluación

**`create_evaluation_dataset()`**
- Define conjunto de pruebas médicas predefinidas
- 5 casos de prueba sobre condiciones comunes
- Estructura: query, expected_context, ground_truth
- Implementa few-shot prompting para evaluación

#### Logging y Exportación

**`log_interaction(query, response, metrics, context_docs)`**
- Registra interacciones del usuario con timestamps
- Almacena métricas completas de rendimiento
- Genera IDs únicos para tracking
- Mantiene historial en sesión de Streamlit

**`export_langsmith_format(logs)`**
- Convierte logs a formato compatible con LangSmith
- Estructura datos para análisis externo
- Incluye inputs, outputs, métricas y metadata
- Facilita integración con herramientas de MLOps

#### Función Principal

**`main()`**
- Punto de entrada de la aplicación Streamlit
- Gestiona estado de sesión y configuración
- Implementa interfaz de 5 pestañas
- Maneja flujo completo: consulta → búsqueda → generación → evaluación

### Métricas Monitoreadas

#### Rendimiento
- Tiempo de recuperación de documentos
- Tiempo de generación de respuestas
- Tiempo total de procesamiento
- Número de documentos recuperados

#### Calidad
- Faithfulness (fidelidad al contexto)
- Relevance (relevancia de la respuesta)
- Context Precision (precisión del contexto)
- Score promedio de relevancia

### Flujo de Trabajo

1. **Carga inicial**: Documentos médicos predefinidos
2. **Generación de embeddings**: Vectorización con LangChain
3. **Consulta del usuario**: Pregunta médica
4. **Búsqueda híbrida**: Recuperación de documentos relevantes
5. **Generación**: Respuesta basada en contexto
6. **Evaluación**: Métricas automáticas de calidad
7. **Logging**: Registro para análisis posterior

### Visualizaciones Disponibles

- Distribución de tiempos de respuesta
- Histogramas de scores de fidelidad
- Scatter plots de métricas de calidad
- Estadísticas generales del sistema
- Análisis de longitud de documentos

### Casos de Uso

- Consultas médicas básicas
- Evaluación de sistemas RAG médicos
- Análisis de rendimiento de embeddings
- Benchmarking de modelos de lenguaje
- Investigación en IA médica

### Consideraciones de Seguridad

- Tokens de API manejados via variables de entorno
- Validación de inputs del usuario
- Manejo seguro de errores y excepciones
- Logs sin información sensible

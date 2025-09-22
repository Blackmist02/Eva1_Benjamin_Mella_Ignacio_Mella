# Eva1_Benjamin_Mella_Ignacio_Mella

## Sistema RAG con Evaluación de Rendimiento

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) con capacidades de evaluación utilizando Streamlit, OpenAI, y LangChain.

### Funcionalidades Principales

#### 1. Configuración y Cliente
- Configuración de cliente OpenAI usando tokens de GitHub
- Inicialización de modelos de embeddings de LangChain (`text-embedding-3-small`)
- Manejo de variables de entorno con soporte para archivos `.env`

#### 2. Generación de Embeddings
- **`get_embeddings_langchain()`**: Genera embeddings para documentos usando LangChain
- **`get_query_embedding_langchain()`**: Genera embeddings para consultas
- Conversión automática de texto a objetos Document de LangChain

#### 3. Búsqueda Híbrida
- **`hybrid_search_with_metrics()`**: Combina búsqueda semántica y por palabras clave
- Utiliza similitud coseno para búsqueda semántica (70% del peso)
- Implementa búsqueda por palabras clave basada en intersección (30% del peso)
- Retorna documentos rankeados con métricas de tiempo

#### 4. Generación de Respuestas
- **`generate_response_with_metrics()`**: Genera respuestas usando GPT-4o
- Combina múltiples documentos recuperados como contexto
- Mide tiempo de generación de respuestas

#### 5. Sistema de Evaluación
- **`evaluate_faithfulness()`**: Evalúa si la respuesta es fiel al contexto (escala 1-10)
- **`evaluate_relevance()`**: Evalúa la relevancia de la respuesta a la consulta (escala 1-10)
- **`evaluate_context_precision()`**: Calcula la precisión del contexto recuperado
- Todas las evaluaciones utilizan GPT-4o como juez automático

### Tecnologías Utilizadas
- **Streamlit**: Framework para la interfaz web
- **OpenAI GPT-4o**: Modelo de lenguaje para generación y evaluación
- **LangChain**: Framework para embeddings y manejo de documentos
- **scikit-learn**: Cálculo de similitud coseno
- **NumPy/Pandas**: Manipulación de datos
- **Plotly**: Visualización de datos

### Requisitos
- Token de GitHub válido para acceso a modelos
- Variables de entorno: `GITHUB_TOKEN`, `GITHUB_BASE_URL`
- Dependencias: streamlit, openai, langchain-openai, sklearn, plotly, python-dotenv

### Métricas de Evaluación
El sistema proporciona métricas completas de rendimiento:
- Tiempo de recuperación de documentos
- Tiempo de generación de respuestas
- Puntuaciones de fidelidad y relevancia
- Precisión del contexto recuperado

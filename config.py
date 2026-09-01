"""
Configuración centralizada del sistema multiagente:
- Modelos de LLM y embeddings (LangChain).
- Cliente y callback handler de Langfuse para trazabilidad end-to-end.
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

load_dotenv()

# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------
# Un LLM "rápido y barato" para la tarea de clasificación del Orquestador,
# y otro para la generación de respuestas de los Agentes RAG.
CLASSIFIER_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

GENERATION_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


def get_langfuse_handler(session_id: str, user_id: str = "usuario_demo", tags=None) -> CallbackHandler:
    """
    Crea un CallbackHandler de Langfuse para una ejecución concreta del grafo.

    Al pasar `session_id`, todas las invocaciones de una misma conversación
    quedan agrupadas en Langfuse bajo la misma sesión, y cada nodo del grafo
    (clasificación, retrieval, generación) aparece como un span anidado
    dentro del trace raíz "Orquestador".
    """
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
    )

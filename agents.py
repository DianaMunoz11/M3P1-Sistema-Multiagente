"""
Define:
  1. El estado compartido del grafo (AgentState).
  2. El Agente Orquestador: clasifica la intención del usuario en
     "rr_hh", "tecnologia" u "otro" mediante salida estructurada (Pydantic).
  3. Los Agentes RAG especializados (RR. HH. y Tecnología): cada uno
     con su propio retriever y su propio prompt de generación
     contextualmente fundamentada (grounded generation).

Cada función de nodo está decorada con @observe de Langfuse para que,
además de las trazas automáticas de los componentes LangChain, cada
"agente" aparezca como un span nombrado explícitamente en el trace.
"""

from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langfuse import observe

from config import CLASSIFIER_LLM, GENERATION_LLM
from vectorstore_builder import get_hr_retriever, get_tech_retriever


# ---------------------------------------------------------------------------
# Estado compartido del grafo
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    query: str                 # Consulta original del usuario
    intent: str                # "rr_hh" | "tecnologia" | "otro"
    confidence: float          # Confianza de la clasificación
    context_docs: List[Document]  # Documentos recuperados por el Agente RAG
    answer: str                 # Respuesta final generada
    agent_used: str              # Qué agente RAG atendió la consulta


# ---------------------------------------------------------------------------
# 1) AGENTE ORQUESTADOR — Clasificación de intención
# ---------------------------------------------------------------------------
class IntentClassification(BaseModel):
    intent: Literal["rr_hh", "tecnologia", "otro"] = Field(
        description=(
            "Categoría de la consulta: 'rr_hh' para temas de Recursos Humanos "
            "(vacaciones, licencias, beneficios, onboarding, desempeño, nómina); "
            "'tecnologia' para temas de IT (accesos, VPN, contraseñas, software, "
            "infraestructura, soporte técnico); 'otro' si no corresponde a ninguna."
        )
    )
    confidence: float = Field(description="Confianza de 0 a 1 en la clasificación")
    justificacion: str = Field(description="Breve razón de la clasificación, en una frase")


_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el Agente Orquestador de un sistema multiagente empresarial. "
     "Tu única tarea es clasificar la intención de la consulta del usuario "
     "en una de estas categorías: 'rr_hh', 'tecnologia' u 'otro'. "
     "No respondas la consulta, solo clasifícala."),
    ("human", "Consulta del usuario:\n{query}"),
])

_classifier_chain = _CLASSIFIER_PROMPT | CLASSIFIER_LLM.with_structured_output(
    IntentClassification
)


@observe(name="orquestador.clasificar_intencion")
def classify_intent_node(state: AgentState) -> dict:
    """Nodo del grafo: clasifica la intención de la consulta del usuario."""
    result: IntentClassification = _classifier_chain.invoke({"query": state["query"]})
    return {"intent": result.intent, "confidence": result.confidence}


# ---------------------------------------------------------------------------
# 2) AGENTES RAG ESPECIALIZADOS
# ---------------------------------------------------------------------------
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente especializado del área de {area}. "
     "Responde ÚNICAMENTE con base en el siguiente contexto recuperado de la "
     "base de conocimiento oficial. Si el contexto no contiene la respuesta, "
     "indica explícitamente que no cuentas con esa información y sugiere "
     "contactar al área correspondiente. No inventes datos.\n\n"
     "Contexto:\n{context}"),
    ("human", "{query}"),
])

_generation_chain = _RAG_PROMPT | GENERATION_LLM | StrOutputParser()


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


@observe(name="agente_rag.rr_hh")
def rr_hh_rag_node(state: AgentState) -> dict:
    """Agente RAG especializado en Recursos Humanos."""
    retriever = get_hr_retriever()
    docs = retriever.invoke(state["query"])
    answer = _generation_chain.invoke({
        "area": "Recursos Humanos",
        "context": _format_docs(docs),
        "query": state["query"],
    })
    return {"context_docs": docs, "answer": answer, "agent_used": "Agente RAG - RR. HH."}


@observe(name="agente_rag.tecnologia")
def tecnologia_rag_node(state: AgentState) -> dict:
    """Agente RAG especializado en Tecnología / IT."""
    retriever = get_tech_retriever()
    docs = retriever.invoke(state["query"])
    answer = _generation_chain.invoke({
        "area": "Tecnología (IT)",
        "context": _format_docs(docs),
        "query": state["query"],
    })
    return {"context_docs": docs, "answer": answer, "agent_used": "Agente RAG - Tecnología"}


@observe(name="agente_fallback.otro")
def fallback_node(state: AgentState) -> dict:
    """Se activa cuando la intención no corresponde a ningún dominio conocido."""
    answer = (
        "Tu consulta no parece corresponder a los dominios de Recursos Humanos "
        "o Tecnología que puedo atender. ¿Podrías reformularla o indicar a qué "
        "área pertenece?"
    )
    return {"context_docs": [], "answer": answer, "agent_used": "Fallback"}

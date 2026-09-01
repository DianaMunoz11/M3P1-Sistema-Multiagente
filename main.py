"""
Punto de entrada del sistema multiagente.

Cada llamada a `responder()` invoca el grafo completo (Orquestador -> Agente
RAG especializado) y adjunta el CallbackHandler de Langfuse, de modo que en
el dashboard de Langfuse se ve un trace raíz con spans anidados:

    Trace: "consulta_usuario"
      └── orquestador.clasificar_intencion   (LLM call de clasificación)
      └── agente_rag.rr_hh  (o .tecnologia / fallback)
            ├── retriever (similarity_search sobre FAISS)
            └── ChatOpenAI (generación de la respuesta final)

Uso:
    python main.py
"""

import uuid

from config import get_langfuse_handler, langfuse_client
from graph import multiagent_app


def responder(query: str, session_id: str) -> dict:
    """
    Ejecuta el flujo completo del sistema multiagente para una consulta,
    con trazabilidad completa en Langfuse.
    """
    langfuse_handler = get_langfuse_handler(
        session_id=session_id,
        tags=["multiagente", "rag", "orquestador"],
    )

    result = multiagent_app.invoke(
        {"query": query},
        config={
            "callbacks": [langfuse_handler],
            "run_name": "consulta_usuario",
            "metadata": {"langfuse_session_id": session_id},
        },
    )

    return result


if __name__ == "__main__":
    session_id = str(uuid.uuid4())

    preguntas_demo = [
        "¿Cuántos días de vacaciones tengo al año y cómo las solicito?",
        "Necesito resetear mi contraseña corporativa, ¿cuál es el proceso?",
        "¿Cuál es la capital de Francia?",
    ]

    for pregunta in preguntas_demo:
        print(f"\n{'='*70}\nPregunta: {pregunta}")
        resultado = responder(pregunta, session_id=session_id)
        print(f"Intención detectada: {resultado['intent']} "
              f"(confianza: {resultado.get('confidence', 'N/A')})")
        print(f"Agente utilizado: {resultado['agent_used']}")
        print(f"Respuesta: {resultado['answer']}")

    # Asegura el envío de todos los eventos pendientes a Langfuse antes de salir
    langfuse_client.flush()

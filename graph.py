from langgraph.graph import StateGraph, END

from agents import (
    AgentState,
    classify_intent_node,
    rr_hh_rag_node,
    tecnologia_rag_node,
    fallback_node,
)


def _route_by_intent(state: AgentState) -> str:
    """
    Función de enrutamiento condicional: lee la intención clasificada por el
    Agente Orquestador y decide a qué nodo (Agente RAG) delegar la consulta.
    """
    return {
        "rr_hh": "rag_rr_hh",
        "tecnologia": "rag_tecnologia",
    }.get(state["intent"], "fallback")


def build_graph():
    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("orquestador", classify_intent_node)
    workflow.add_node("rag_rr_hh", rr_hh_rag_node)
    workflow.add_node("rag_tecnologia", tecnologia_rag_node)
    workflow.add_node("fallback", fallback_node)

    # Punto de entrada
    workflow.set_entry_point("orquestador")

    # Enrutamiento condicional desde el Orquestador
    workflow.add_conditional_edges(
        "orquestador",
        _route_by_intent,
        {
            "rag_rr_hh": "rag_rr_hh",
            "rag_tecnologia": "rag_tecnologia",
            "fallback": "fallback",
        },
    )

    # Todos los agentes RAG terminan el flujo
    workflow.add_edge("rag_rr_hh", END)
    workflow.add_edge("rag_tecnologia", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


# Grafo compilado, listo para invocarse
multiagent_app = build_graph()

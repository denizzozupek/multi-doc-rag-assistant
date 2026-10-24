import logging
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from src.config import DEFAULT_LLM, DEFAULT_RERANK_TOP_K
from src.graph.nodes import (
    create_generation_node,
    create_grade_documents_node,
    create_retrieve_and_rerank_node,
    create_rewrite_query_node,
    create_web_search_node,
)
from src.graph.state import State

logger = logging.getLogger(__name__)


# A conditional edge function to determine if web search is needed based on the state
def web_search_needed_edge(state: State) -> str:
    """Determine if web search is needed based on the state."""

    web_search_needed = state.get("web_search_needed", False)
    if web_search_needed:
        logger.info("Web search is needed based on the state.")
        return "web_search"
    else:
        logger.info("Web search is not needed based on the state.")
        return "generate_response"



def graph_pipeline(
    llm: BaseChatModel = DEFAULT_LLM,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> StateGraph:
    """Creates and returns an uncompiled StateGraph for the RAG workflow."""
    workflow = StateGraph(State)

    # Add processing nodes to the graph
    workflow.add_node("rewrite_query", create_rewrite_query_node(llm=llm))
    workflow.add_node(
        "retrieve_and_rerank",
        create_retrieve_and_rerank_node(top_k=top_k),
    )
    workflow.add_node("generate_response", create_generation_node(llm=llm))
    workflow.add_node("grade_documents", create_grade_documents_node(llm=llm))
    workflow.add_node("web_search", create_web_search_node(llm=llm))
    
    

    # Define linear execution edges
    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve_and_rerank")
    workflow.add_edge("retrieve_and_rerank", "grade_documents")
    workflow.add_conditional_edges("grade_documents",
                                   web_search_needed_edge,
                                   {"web_search": "web_search", "generate_response": "generate_response"})
    workflow.add_edge("web_search", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow
import logging
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from src.config import DEFAULT_LLM, DEFAULT_RERANK_TOP_K
from src.graph.nodes import (
    create_generation_node,
    create_retrieve_and_rerank_node,
    create_rewrite_query_node,
)
from src.graph.state import State

logger = logging.getLogger(__name__)


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

    # Define linear execution edges
    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve_and_rerank")
    workflow.add_edge("retrieve_and_rerank", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow
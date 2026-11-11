import logging
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import DEFAULT_LLM, DEFAULT_RERANK_TOP_K
from src.graph.nodes import (
    create_generation_node,
    create_grade_documents_node,
    create_retrieve_and_rerank_node,
    create_rewrite_query_node,
    fallback_agent_node,
)
from src.graph.state import State
from src.agent.tools import search_docs, search_web

logger = logging.getLogger(__name__)


def agent_needed_edge(state: State) -> str:
    """Determine if grader approved documents are sufficient not to proceed with response generation, if not, escalate to agent node for further processing."""

    agent_needed = state.get("web_search_needed", False)
    if agent_needed:
        logger.info("Agent is needed based on the state.")
        return "fallback_agent"
    else:
        logger.info("Agent is not needed based on the state.")
        return "generate_response"


def escalate_graph_pipeline(
    llm: BaseChatModel = DEFAULT_LLM,
    top_k: int = DEFAULT_RERANK_TOP_K,
    crag_enable: bool = True,
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

    if crag_enable:
        tool_node = ToolNode([search_docs, search_web])
        workflow.add_node("tools", tool_node)

        workflow.add_node("grade_documents", create_grade_documents_node(llm=llm))
        workflow.add_node("fallback_agent", fallback_agent_node(llm=llm))

        workflow.add_edge("retrieve_and_rerank", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            agent_needed_edge,
            {
                "fallback_agent": "fallback_agent",
                "generate_response": "generate_response",
            },
        )
        workflow.add_conditional_edges("fallback_agent", tools_condition)
        workflow.add_edge("tools", "fallback_agent")

    else:
        workflow.add_edge("retrieve_and_rerank", "generate_response")

    workflow.add_edge("generate_response", END)
    return workflow

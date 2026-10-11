from src.graph.nodes import create_rewrite_query_node, create_retrieve_and_rerank_node, create_generation_node
from src.graph.state import State

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever

DEFAULT_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=3,
)

def graph_pipeline(retriever: BaseRetriever, llm: BaseChatModel = DEFAULT_LLM, top_k: int = 3) -> StateGraph:
    """Creates and returns a StateGraph for the workflow."""

    if retriever is None:
        raise ValueError("Retriever is not initialized. Please ensure the retriever is set up correctly.")
    
    workflow = StateGraph(State)

    # Add nodes to the workflow
    workflow.add_node("rewrite_query", create_rewrite_query_node(llm=llm))
    workflow.add_node("retrieve_and_rerank", create_retrieve_and_rerank_node(retriever=retriever, model=llm, top_k=top_k))
    workflow.add_node("generate_response", create_generation_node(llm=llm))

    # Define the workflow edges
    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve_and_rerank")
    workflow.add_edge("retrieve_and_rerank", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow



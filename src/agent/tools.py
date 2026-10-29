import logging
from typing import List
from functools import lru_cache
from typing import List

from langchain_core.tools import tool

from src.retrieval.retriever import get_retriever
from src.retrieval.reranker import reranker
from src.config import DEFAULT_RERANK_TOP_K

logger = logging.getLogger(__name__)

query_cache = dict[str, str] = {}

@lru_cache(maxsize=1)
def get_cached_retriever(selected_files: List[str] | None = None):
    if selected_files is not None:
        return get_retriever(selected_files=selected_files)
    return get_retriever()

@tool
def search_internal_docs(query: str, selected_files: List[str] | None = None) -> str:
    """Searches the internal knowledge base, technical documentation, and indexed research papers.
    
    Use this tool whenever you need to retrieve verified facts, algorithms, formulas, or specific
    domain knowledge from the local document repository. You can optionally provide `selected_files`
    to restrict the search to specific document filenames.
    """
    normalized_query = query.strip().lower()
    files_key = ",".join(sorted(selected_files)) if selected_files else "all"
    try:
        if selected_files:
            retriever = get_cached_retriever(selected_files=selected_files)
        else:
            retriever = get_cached_retriever()
    except Exception as e:
        return f"Error occurred while initializing the retriever: {str(e)}"
    
    if retriever is None:
        return "No retriever available. Please ensure the vector database is set up correctly."
    
    docs = retriever.invoke(query)
    final_docs = reranker.invoke(docs, query, top_k=DEFAULT_RERANK_TOP_K)
    
    if not docs:
        return "No relevant documents found."
import logging
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from src.retrieval.bm25 import get_bm25_retriever
from src.config import (
    BM25_PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)
from src.retrieval.hybrid_search import HybridRetriever
from src.utils import build_filter

logger = logging.getLogger(__name__)


def get_dense_retriever(
    k: int = 3,
    fetch_k: int = 20,
    selected_files: list[str] | None = None,
    embedding_model: Embeddings | None = None,
    persist_directory: str | None = None,
    search_type: str = "similarity",
) -> BaseRetriever | None:
    """Returns an LCEL Chroma vector store retriever."""
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )

    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        logger.info("Vector database not found on disk.")
        return None

    try:
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )
    except Exception as e:
        logger.error(f"Error occurred while loading vector database: {e}", exc_info=True)
        raise

    search_kwargs: dict[str, Any] = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k

    filter_dict = build_filter(selected_files)
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    # Return the retriever with the specified search type and parameters as a BaseRetriever instance. This allows for flexible retrieval strategies (similarity or MMR) based on the provided search_type.
    return vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def get_retriever(
    k: int = 3,
    fetch_k: int = 20,
    selected_files: list[str] | None = None,
    embedding_model: Embeddings | None = None,
    persist_directory: str | None = None,
    bm25_persist_dir: str | None = None,
    search_type: str = "similarity",
) -> BaseRetriever | None:
    """Unified entry point returning Dense, BM25, or Hybrid retriever."""
    if search_type == "bm25":
        return get_bm25_retriever(
            bm25_persist_dir=bm25_persist_dir,
            k=k,
            selected_files=selected_files,
        )

    if search_type == "hybrid":
        dense = get_dense_retriever(
            k=fetch_k,
            selected_files=selected_files,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
        )
        bm25 = get_bm25_retriever(
            bm25_persist_dir=bm25_persist_dir,
            k=fetch_k,
            selected_files=selected_files,
        )

        if not dense and not bm25:
            return None
        if not dense:
            return bm25
        if not bm25:
            return dense

        return HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, k=k)

    # Default: Dense search (similarity / mmr)
    return get_dense_retriever(
        k=k,
        fetch_k=fetch_k,
        selected_files=selected_files,
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        search_type=search_type,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hybrid = get_retriever(search_type="hybrid", k=2, selected_files=["arxiv1.pdf"])
    if hybrid:
        test_docs = hybrid.invoke("Nash equilibrium")
        print(f"Retrieved {len(test_docs)} hybrid documents.")
        for d in test_docs:
            print(f"- Source: {d.metadata.get('source')} | Text: {d.page_content[:80]}...")
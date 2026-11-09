import asyncio
import logging
from typing import Any

from src.config import (
    DEFAULT_RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVER_SEARCH_TYPE,
)
from src.retrieval.reranker import reranker, get_rerank_semaphore
from src.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)


async def retrieve_and_rerank(
    query: str | None,
    selected_files: list[str] | None,
    model: Any = None,
    candidate_k: int = RETRIEVAL_CANDIDATE_K,
    top_k: int = DEFAULT_RERANK_TOP_K,
    search_type: str = RETRIEVER_SEARCH_TYPE,
) -> list:
    """Retrieves candidates from the selected files and reranks them.

    Shared by the CRAG node (mode 1), the agent tool (mode 2).
    """
    if not selected_files:
        logger.warning(
            "No selected files provided. Using all available documents for retrieval."
        )

    if not query or not query.strip():
        logger.warning("No query provided. Returning empty document list.")
        return []

    retriever = get_retriever(
        k=candidate_k,
        selected_files=selected_files,
        search_type=search_type,
    )
    if retriever is None:
        logger.warning("Retriever could not be created. Returning empty document list.")
        return []

    raw_docs = await retriever.ainvoke(query)
    logger.info("Retrieved documents count: %s", len(raw_docs))

    if not raw_docs:
        logger.warning("No documents retrieved.")
        return []

    # Cross-encoder is sync and CPU-heavy: run in a thread, throttled by semaphore
    async with get_rerank_semaphore():
        return await asyncio.to_thread(
            reranker,
            query=query,
            documents=raw_docs,
            model=model,
            top_k=top_k,
        )


def format_docs(docs: list) -> str:
    """Text returned to the agent: source + page + content per passage."""
    return "\n\n".join(
        f"[{d.metadata.get('source')} | p.{d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )
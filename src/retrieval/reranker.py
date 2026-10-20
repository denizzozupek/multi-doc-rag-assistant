import os
import logging
import asyncio
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.config import CROSS_ENCODER_MODEL_NAME, DEFAULT_RERANK_TOP_K, MAX_RERANK_CONCURRENCY
from src.utils import get_optimal_device

logger = logging.getLogger(__name__)

_cross_encoder_model = None
_rerank_semaphore = None
_rerank_semaphore_loop = None


def get_rerank_semaphore(max_concurrency: int = MAX_RERANK_CONCURRENCY) -> asyncio.Semaphore:
    """
    Returns a singleton asyncio.Semaphore for controlling concurrency of reranking operations.
    Decreases reranking latency by limiting the number of concurrent reranking tasks.
    """
    global _rerank_semaphore, _rerank_semaphore_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _rerank_semaphore is None or _rerank_semaphore_loop is not current_loop:
        _rerank_semaphore = asyncio.Semaphore(max_concurrency)
        _rerank_semaphore_loop = current_loop

    return _rerank_semaphore


def get_cross_encoder_model() -> CrossEncoder:
    """
    Singleton factory returning the CrossEncoder model.
    Automatically detects and selects the optimal compute accelerator (CUDA/MPS/XPU/CPU).
    """
    global _cross_encoder_model
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    try:
        device = get_optimal_device()
        logger.info(f"Loading CrossEncoder model '{CROSS_ENCODER_MODEL_NAME}' on {device}...")
        _cross_encoder_model = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME, max_length=512, device=device
        )
        return _cross_encoder_model
    except Exception as e:
        logger.error(f"Error loading CrossEncoder model: {e}", exc_info=True)
        raise

def reranker(
    query: str,
    documents: list[Document],
    model: CrossEncoder | None = None,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[Document]:
    """
    Reranks documents using a CrossEncoder model.

    """
    if not documents:
        return []

    # Prepare pairs of (query, document) for scoring
    query_doc_pairs = [(query, doc.page_content) for doc in documents]

    if model is None:
        model = get_cross_encoder_model()

    # Get relevance scores from the cross-encoder model
    scores = model.predict(query_doc_pairs)

    scored_docs = []
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k documents
    return [doc for doc, _ in scored_docs[:top_k]]

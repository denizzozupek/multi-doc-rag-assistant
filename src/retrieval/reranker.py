import os
import logging
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.config import CROSS_ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)

_cross_encoder_model = None


def get_cross_encoder_model() -> CrossEncoder:
    global _cross_encoder_model
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    else:
        try:
            device = (
                "cuda"
                if os.environ.get("USE_CUDA", "false").lower() == "true"
                else "cpu"
            )
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
    top_k: int = 2,
) -> list[Document]:
    """Reranks documents using a cross-encoder model."""
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
        scored_docs.append((doc, score)) # Store the score in the document's metadata

    scored_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order

    # Return the top_k documents based on the scores
    top_docs = [doc for doc, _ in scored_docs[:top_k]]
    return top_docs


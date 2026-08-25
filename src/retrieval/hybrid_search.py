import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    doc_lists: list[list[Document]],
    k: int = 3,
    rrf_k: int = 60,
) -> list[Document]:
    """Combines multiple document rankings using Reciprocal Rank Fusion (RRF)."""
    combined_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            doc_key = doc.page_content
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

            combined_scores[doc_key] = (
                combined_scores.get(doc_key, 0.0) + 1.0 / (rrf_k + rank + 1)
            )

    sorted_keys = sorted(
        combined_scores.keys(),
        key=lambda x: combined_scores[x],
        reverse=True,
    )

    return [doc_map[key] for key in sorted_keys[:k]]


class HybridRetriever(BaseRetriever):
    """Custom Retriever combining Dense and BM25 search via Reciprocal Rank Fusion."""

    dense_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    k: int = 3
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str) -> list[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        return reciprocal_rank_fusion(
            doc_lists=[dense_docs, bm25_docs],
            k=self.k,
            rrf_k=self.rrf_k,
        )
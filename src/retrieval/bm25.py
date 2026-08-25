import logging
import os
import bm25s
import Stemmer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import BM25_PERSIST_DIRECTORY

logger = logging.getLogger(__name__)

stemmer = Stemmer.Stemmer("english")


class BM25Retriever(BaseRetriever):
    """BM25 wrapper compatible with LangChain's BaseRetriever."""

    retriever: bm25s.BM25
    stemmer: Stemmer.Stemmer
    k: int = 3
    selected_files: list[str] | None = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, stopwords="en")
        fetch_k = len(self.retriever.corpus) if self.selected_files else self.k

        results, _ = self.retriever.retrieve(
            query_tokens, corpus=self.retriever.corpus, k=fetch_k
        )

        matched_docs: list[Document] = []
        for item in results[0]:
            if self.selected_files is not None:
                doc_source = item.get("metadata", {}).get("source")
                if doc_source not in self.selected_files:
                    continue

            matched_docs.append(
                Document(page_content=item["page_content"], metadata=item["metadata"])
            )

            if len(matched_docs) == self.k:
                break

        return matched_docs


def get_bm25_retriever(
    bm25_persist_dir: str | None = None,
    stemmer_instance: Stemmer.Stemmer | None = None,
    k: int = 3,
    selected_files: list[str] | None = None,
) -> BM25Retriever | None:
    """Returns a BM25 retriever if the index exists on disk."""
    if bm25_persist_dir is None:
        bm25_persist_dir = BM25_PERSIST_DIRECTORY

    if stemmer_instance is None:
        stemmer_instance = stemmer

    if not os.path.exists(bm25_persist_dir) or not os.listdir(bm25_persist_dir):
        logger.info("BM25 index not found on disk.")
        return None

    try:
        loaded_retriever = bm25s.BM25.load(bm25_persist_dir, load_corpus=True)
        return BM25Retriever(
            retriever=loaded_retriever,
            stemmer=stemmer_instance,
            k=k,
            selected_files=selected_files,
        )
    except Exception as e:
        logger.error(f"Error loading BM25 index: {e}", exc_info=True)
        raise
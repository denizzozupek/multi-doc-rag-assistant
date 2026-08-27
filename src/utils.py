import logging
import os
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
import bm25s
import shutil
import Stemmer

from src.config import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, BM25_PERSIST_DIRECTORY

logger = logging.getLogger(__name__)
stemmer = Stemmer.Stemmer("english")

def format_history(history: list[dict]) -> list[BaseMessage]:
    """Convert raw dictionary chat history into LangChain BaseMessage objects."""
    return [
        (
            HumanMessage(content=item["content"])
            if item["role"] == "user"
            else AIMessage(content=item["content"])
        )
        for item in history
    ]


def build_filter(selected_files: list[str] | None = None) -> dict | None:
    """Builds a metadata filter dictionary for vector search."""
    if not selected_files:
        return None

    if len(selected_files) == 1:
        return {"source": selected_files[0]}
    return {"source": {"$in": selected_files}}


def get_existing_file_names_with_hashes() -> dict[str, str]:
    """Returns a dictionary mapping existing file names to their corresponding PDF hashes."""
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.info(f"Persist directory '{PERSIST_DIRECTORY}' does not exist.")
        return {}

    try:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
        )
        results = vector_db.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        return {
            x["source"]: x["pdf_hash"]
            for x in metadatas
            if x and "source" in x and "pdf_hash" in x
        }
    except Exception as e:
        logger.error(f"Error retrieving file names from vector db: {e}", exc_info=True)
        return {}



def delete_file_from_vector_db(file_name: str) -> bool:
    """Deletes a file from the vector database based on its source name and its bm25 index. Returns True if deletion was successful, False otherwise."""
    try:
        # Delete from Vector Database
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
        )
        vector_db.delete(where={"source": file_name})
        logger.info(f"Successfully deleted '{file_name}' from the vector database.")

        return True
    except Exception as e:
        logger.error(
            f"Error deleting '{file_name}' from vector db: {e}", exc_info=True
        )
        return False

def delete_file_from_bm25_index(bm25_persist_directory: str, file_name: str) -> bool:
    """Deletes a file from the BM25 index based on its source name. Returns True if deletion was successful, False otherwise."""
    if bm25_persist_directory is None:
        bm25_persist_directory = BM25_PERSIST_DIRECTORY

    if not os.path.exists(bm25_persist_directory) or not os.listdir(bm25_persist_directory):
        logger.info("BM25 index not found on disk.")
        return False

    try:
        # Load the BM25 index
        loaded_retriever = bm25s.BM25.load(bm25_persist_directory, load_corpus=True)
        remaining_records = [
            item for item in loaded_retriever.corpus if item.get("metadata", {}).get("source") != file_name
        ]
        all_corpus_texts = [item["page_content"] for item in remaining_records]

        # If no records remain after deletion, remove the entire BM25 index directory
        if len(remaining_records) == 0:
            shutil.rmtree(bm25_persist_directory)
        else:
            # Create a new BM25 index with the remaining records
            tokenized_corpus = bm25s.tokenize(all_corpus_texts, stemmer=stemmer, stopwords="en")
            new_retriever = bm25s.BM25(corpus=remaining_records)
            new_retriever.index(tokenized_corpus)
            new_retriever.save(bm25_persist_directory)

    except Exception as e:
        logger.error(f"Error deleting '{file_name}' from BM25 index: {e}", exc_info=True)
        return False
    return True

def delete_file(file_name: str) -> bool:
    """Deletes a file from both the vector database and the BM25 index. Returns True if deletion was successful, False otherwise."""
    vector_db_deleted = delete_file_from_vector_db(file_name)
    bm25_index_deleted = delete_file_from_bm25_index(BM25_PERSIST_DIRECTORY, file_name)

    if vector_db_deleted and bm25_index_deleted:
        logger.info(f"Successfully deleted '{file_name}' from both vector db and BM25 index.")
        return True
    else:
        logger.error(f"Failed to delete '{file_name}' from either vector db or BM25 index.")
        return False
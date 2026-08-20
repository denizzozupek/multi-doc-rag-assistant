import os
import logging
from typing import Any
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from src.config import PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def get_existing_file_names_with_hashes() -> dict[str, str]:
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.info(
            f"Persist directory '{PERSIST_DIRECTORY}' does not exist. Returning empty dictionary."
        )
        return {}

    try:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model
        )
        results = vector_db.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        file_hash_map = {
            x["source"]: x["pdf_hash"]
            for x in metadatas
            if x and "source" in x and "pdf_hash" in x
        }

        return file_hash_map
    except Exception as e:
        logger.error(
            f"Error retrieving existing file names from vector database: {e}",
            exc_info=True,
        )
        return {}


def delete_file_from_vector_db(file_name: str) -> bool:
    """Deletes a file from the vector database based on its name."""
    try:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model
        )
        vector_db.delete(where={"source": file_name})
        logger.info(f"Successfully deleted '{file_name}' from the vector database.")
        return True
    except Exception as e:
        logger.error(
            f"Error deleting '{file_name}' from the vector database: {e}", exc_info=True
        )
        return False


def build_filter(selected_files: list[str] | None = None) -> dict | None:
    if selected_files:
        logger.info(
            f"Filtering retriever to only include selected files: {selected_files}"
        )
        if len(selected_files) == 1:
            return {"source": selected_files[0]}
        else:
            return {"source": {"$in": selected_files}}
    else:
        logger.info(
            "No specific files selected. Using the entire vector database for retrieval."
        )
        return None


def get_retriever(
    k: int = 3,
    selected_files: list[str] | None = None,
    embedding_model: Embeddings | None = None,
    persist_directory: str | None = None,
    search_type: str = "similarity",
) -> BaseRetriever | None:
    """Returns an LCEL Retriever. If selected_files is provided, the retriever will only search within those files."""

    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )

    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        logger.info(
            "Vector database not found on disk. Please load your documents to create the vector database."
        )
        return None
    else:
        logger.info("Loading existing vector database from disk...")
        try:
            vector_db = Chroma(
                persist_directory=persist_directory, embedding_function=embedding_model
            )
        except Exception as e:
            logger.error(
                f"Error occurred while loading vector database: {e}", exc_info=True
            )
            raise

    search_kwargs: dict[str, Any] = {"k": k}
    filter_dict = build_filter(selected_files)
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    # LangChain wraps the Chroma DB object inside a VectorStoreRetriever (an LCEL Runnable) to query the vector DB.
    return vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

import logging
import os
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

logger = logging.getLogger(__name__)


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
    """Deletes a file from the vector database based on its source name."""
    try:
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
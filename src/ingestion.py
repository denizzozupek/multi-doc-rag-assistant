import hashlib
import logging
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import EMBEDDING_MODEL_NAME, PDF_PATH, PERSIST_DIRECTORY, CHUNK_SIZE, OVERLAP_SIZE

logger = logging.getLogger(__name__)

# STEP 1: Hashing Function
def compute_pdf_hash(pdf_path: str) -> str:
    """Calculates the SHA-256 hash of a PDF file to uniquely identify it."""
    try:
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error computing PDF hash: {e}", exc_info=True)
        raise

# STEP 2: Check if File is Already Ingested
def is_file_already_ingested(
    hash_value: str, persist_dir: str, embedding_model: Embeddings
) -> bool:
    """Checks if a document with the specified hash is already ingested in ChromaDB."""
    if not os.path.exists(persist_dir):
        return False

    try:
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        results = vector_db.get(where={"pdf_hash": hash_value})
        return len(results.get("ids", [])) > 0
    except Exception as e:
        logger.error(f"Error checking if file is already ingested: {e}", exc_info=True)
        raise

# STEP 3: Ingestion Pipeline
def ingestion_pipeline(
    pdf_path: str = PDF_PATH, persist_directory: str | None = None, embedding_model: Embeddings | None = None
) -> VectorStore:
    """Ingests a PDF file, splits it into chunks, adds metadata, and stores it in ChromaDB.

    If the file has already been processed, it skips the ingestion API call (0 cost).
    """
    logger.info("Starting ingestion pipeline...")

    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )

    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    pdf_hash = compute_pdf_hash(pdf_path)
    logger.info(f"Processing PDF with hash: {pdf_hash}")

    
    if is_file_already_ingested(pdf_hash, persist_directory, embedding_model):
        logger.info(
            f"PDF with hash {pdf_hash[:8]}... has already been ingested. Skipping ingestion."
        )
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )

    # STEP 4: Load PDF and Split into Chunks
    try:
        loader = PyMuPDFLoader(pdf_path)
        raw_documents = loader.load()
        documents = [doc for doc in raw_documents if doc.page_content.strip()]

        logger.info(f"Successfully loaded {len(documents)} documents from PDF.")
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading PDF: {e}", exc_info=True)
        raise

    # STEP 5: Split Documents into Chunks and Add Metadata
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE
        )
        chunks = text_splitter.split_documents(documents)

        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        if not chunks:
            logger.warning("No chunks were created from the PDF. Check the PDF content.")
            raise ValueError("No chunks were created from the PDF. Check the PDF content.")


        file_name = os.path.basename(pdf_path)
        for chunk in chunks:
            chunk.metadata["pdf_hash"] = pdf_hash
            chunk.metadata["source"] = file_name

        logger.info(
            f"Successfully split documents into {len(chunks)} chunks with metadata."
        )
    except Exception as e:
        logger.error(f"Error splitting documents: {e}", exc_info=True)
        raise

    # STEP 6: Create Embeddings and Store in Vector Database
    try:
        logger.info(
            "Creating embeddings and storing them in the vector database..."
        )
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )
        vector_db.add_documents(chunks)
        logger.info(
            "Successfully created embeddings and stored them in the vector database."
        )
        return vector_db
    except Exception as e:
        logger.error(
            f"Error creating embeddings or storing in vector database: {e}",
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestion_pipeline()
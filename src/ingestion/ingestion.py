import hashlib
import logging
import os
import bm25s
import Stemmer

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    BM25_PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    PDF_PATH,
    PERSIST_DIRECTORY,
    CHUNK_SIZE,
    OVERLAP_SIZE,
)

logger = logging.getLogger(__name__)

stemmer = Stemmer.Stemmer("english")


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


# STEP 2: Validation
def is_file_already_ingested(
    hash_value: str,
    persist_dir: str,
    embedding_model: Embeddings,
    bm25_persist_dir: str,
) -> bool:
    """Checks if a document with the specified hash is already ingested in BOTH ChromaDB and BM25 index."""
    if not os.path.exists(persist_dir) or not os.path.exists(bm25_persist_dir):
        return False

    # Check ChromaDB
    try:
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        results = vector_db.get(where={"pdf_hash": hash_value})
        if not results.get("ids"):
            return False

    except Exception as e:
        logger.error(f"Error checking ChromaDB ingestion: {e}", exc_info=True)
        return False

    # Check BM25 index
    if not os.path.exists(bm25_persist_dir) or not os.listdir(bm25_persist_dir):
        return False
    try:
        bm25_retriever = bm25s.BM25.load(bm25_persist_dir, load_corpus=True)
        if bm25_retriever.corpus is None:
            return False
        return any(
            rec.get("metadata", {}).get("pdf_hash") == hash_value
            for rec in bm25_retriever.corpus
        )
    except Exception as e:
        logger.error(f"Error checking BM25 ingestion: {e}", exc_info=True)
        return False


# STEP 3: Load, Split & Metadata Enrichment
def load_and_split_pdf(
    pdf_path: str,
    pdf_hash: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = OVERLAP_SIZE,
) -> list[Document]:
    """Loads a PDF file, splits it into overlapping chunks, and attaches metadata."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        raw_documents = loader.load()
        documents = [doc for doc in raw_documents if doc.page_content.strip()]
        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading PDF: {e}", exc_info=True)
        raise

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        logger.warning("No chunks were created from the PDF. Check the PDF content.")
        raise ValueError("No chunks were created from the PDF.")

    file_name = os.path.basename(pdf_path)
    for chunk in chunks:
        chunk.metadata["pdf_hash"] = pdf_hash
        chunk.metadata["source"] = file_name

    logger.info(f"Created {len(chunks)} chunks with metadata from {file_name}")
    return chunks



def combine_indexed_records(chunks: list[Document], pdf_hash: str, bm25_persist_dir: str) -> list[dict]:
    new_records = [
                {"page_content": chunk.page_content, "metadata": chunk.metadata}
                for chunk in chunks
            ]
    old_records = []
    if os.path.exists(bm25_persist_dir) and os.listdir(bm25_persist_dir):
        try:
            old_retriever = bm25s.BM25.load(bm25_persist_dir, load_corpus=True)
            if old_retriever.corpus is not None:
                old_records = [rec for rec in old_retriever.corpus if rec.get("metadata", {}).get("pdf_hash") != pdf_hash]
        except Exception as e:
            logger.warning(
                f"Could not load existing BM25 index, creating fresh one: {e}"
            )

    return new_records + old_records

# STEP 4: BM25 Indexing (Merge & Persist)
def index_to_bm25(
    chunks: list[Document],
    bm25_persist_dir: str,
    pdf_hash: str,
) -> None:
    """Merges new chunks into the existing BM25 corpus and updates the index on disk."""
    try:
        combined_records = combine_indexed_records(chunks, pdf_hash, bm25_persist_dir)

        all_corpus_texts = [rec["page_content"] for rec in combined_records]
        all_chunk_tokens = bm25s.tokenize(
            all_corpus_texts, stemmer=stemmer, stopwords="en"
        )

        bm25_model = bm25s.BM25(corpus=combined_records)
        bm25_model.index(all_chunk_tokens)
        bm25_model.save(bm25_persist_dir)
        logger.info(f"Saved BM25 index with a total of {len(combined_records)} chunks.")
    except Exception as e:
        logger.error(f"Error indexing chunks to BM25: {e}", exc_info=True)
        raise


# STEP 5: ChromaDB Indexing
def index_to_chroma(
    chunks: list[Document],
    persist_dir: str,
    embedding_model: Embeddings,
) -> VectorStore:
    """Stores chunks and their embeddings in ChromaDB."""
    try:
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        vector_db.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to ChromaDB at {persist_dir}")
        return vector_db
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
        raise


# STEP 6: Main Pipeline (Orchestrator)
def ingestion_pipeline(
    pdf_path: str = PDF_PATH,
    persist_directory: str | None = None,
    embedding_model: Embeddings | None = None,
    bm25_persist_dir: str | None = None,
) -> VectorStore:
    """Orchestrates the ingestion pipeline for a given PDF into ChromaDB and BM25."""
    logger.info("Starting ingestion pipeline...")

    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, timeout=30, max_retries=3
        )

    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    if bm25_persist_dir is None:
        bm25_persist_dir = BM25_PERSIST_DIRECTORY

    pdf_hash = compute_pdf_hash(pdf_path)
    logger.info(f"Processing PDF with hash: {pdf_hash}")

    if is_file_already_ingested(
        pdf_hash, persist_directory, embedding_model, bm25_persist_dir
    ):
        logger.info(
            f"PDF with hash {pdf_hash[:8]}... already ingested in both stores. Skipping."
        )
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )

    # 1. Load and Chunk
    chunks = load_and_split_pdf(pdf_path=pdf_path, pdf_hash=pdf_hash)

    # 2. Update BM25 Index
    index_to_bm25(chunks=chunks, bm25_persist_dir=bm25_persist_dir, pdf_hash=pdf_hash)

    # 3. Update Vector Store
    vector_db = index_to_chroma(
        chunks=chunks,
        persist_dir=persist_directory,
        embedding_model=embedding_model,
    )

    return vector_db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestion_pipeline()

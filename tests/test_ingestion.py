import pytest
from pathlib import Path
from src.ingestion import ingestion_pipeline
from langchain_chroma import Chroma

def test_ingestion_pipeline_file_not_found():
    with pytest.raises((FileNotFoundError, ValueError)):
        ingestion_pipeline(pdf_path="non_existent_file.pdf", persist_directory="test_persist_dir")


def test_ingestion_pipeline_success(tmp_path):
    # Create a temporary PDF file for testing
    pdf_file = Path("tests/data/sample.pdf").resolve()  # Ensure this file exists for the test

    assert pdf_file.exists(), (f"Sample PDF file does not exist for testing.")

    # Run the ingestion pipeline
    vector_db = ingestion_pipeline(pdf_path=str(pdf_file), persist_directory=str(tmp_path))

    # Check if the vector database is created and has embeddings
    assert vector_db is not None
    assert isinstance(vector_db, Chroma)  # Check if the returned object is of type Chroma
import uuid
import pytest
from reportlab.pdfgen import canvas
from src.chain import conversation_history, get_chat_history
from src.ingestion import ingestion_pipeline
from src.retriever import get_retriever


@pytest.fixture
def create_pdf(tmp_path):
    """Creates a dynamic test PDF file with the specified filename and content."""
    def _create(filename: str, content: str):
        pdf_path = tmp_path / filename
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, content)
        c.save()
        return pdf_path
    return _create


def test_conversation_history_with_redis():
    session_id = str(uuid.uuid4())
    conversation_chain = conversation_history()

    response1 = conversation_chain.invoke(
        {"input": "Hello, I am Deniz"},
        {"configurable": {"session_id": session_id}},
    )
    response2 = conversation_chain.invoke(
        {"input": "What was my name?"},
        {"configurable": {"session_id": session_id}},
    )

    assert "deniz" in response2.lower()

    history = get_chat_history(session_id)
    assert len(history.messages) == 4


def test_ingestion_and_retrieval_pipeline(create_pdf, tmp_path):
    # 1. Create a test PDF file
    pdf_path = create_pdf(
        "test_doc.pdf",
        "Quantum computers solve problems faster than classical computers."
    )

    # 2. Define a unique persist directory for the vector database
    persist_directory = str(tmp_path / "chroma_db")

    # 3. Ingest the PDF into the vector database
    vector_db = ingestion_pipeline(
        pdf_path=str(pdf_path),
        persist_directory=persist_directory,
    )
    assert vector_db is not None

    # 4. Create a retriever with the same isolated directory and file list filter
    retriever = get_retriever(
        k=1,
        selected_files=[pdf_path.name],
        persist_directory=persist_directory,
    )
    assert retriever is not None

    # 5. Query the retriever and check if it returns the expected result
    query = "What is the main advantage of quantum computers compared to classical computers?"
    results = retriever.invoke(query)

    assert len(results) > 0
    assert "faster" in results[0].page_content.lower()
    assert results[0].metadata["source"] == pdf_path.name


def test_ingestion_skip_duplicate_files(create_pdf, tmp_path):
    # 1. Create a test PDF file
    pdf_path = create_pdf(
        "duplicate_doc.pdf",
        "This is a test document to check duplicate ingestion."
    )

    # 2. Define a unique persist directory for the vector database
    persist_directory = str(tmp_path / "chroma_db_duplicates")

    # 3. Ingest the PDF into the vector database for the first time
    vector_db_first_ingestion = ingestion_pipeline(
        pdf_path=str(pdf_path),
        persist_directory=persist_directory,
    )
    assert vector_db_first_ingestion is not None

    count_first_ingestion = len(vector_db_first_ingestion.get()["ids"])

    # 4. Attempt to ingest the same PDF again and check if it skips the duplicate
    vector_db_second_ingestion = ingestion_pipeline(
        pdf_path=str(pdf_path),
        persist_directory=persist_directory,
    )
    assert vector_db_second_ingestion is not None

    count_second_ingestion = len(vector_db_second_ingestion.get()["ids"])

    # 5. Check that the second ingestion did not add duplicate documents
    assert count_first_ingestion == count_second_ingestion

def test_retriever_filters_by_selected_files(create_pdf, tmp_path):
    # 1. Create two test PDF files
    pdf_path1 = create_pdf(
        "file1.pdf",
        "Ricardo Quaresma is a Portuguese footballer."
    )
    pdf_path2 = create_pdf(
        "file2.pdf",
        "Succession is a popular TV series about a media conglomerate family."
    )

    # 2. Define a unique persist directory for the vector database
    persist_directory = str(tmp_path / "chroma_db_filters")

    # 3. Ingest both PDFs into the vector database
    ingestion_pipeline(
        pdf_path=str(pdf_path1),
        persist_directory=persist_directory,
    )
    ingestion_pipeline(
        pdf_path=str(pdf_path2),
        persist_directory=persist_directory,
    )

    # 4. Create a retriever that only selects the first file
    retriever = get_retriever(
        k=2,
        selected_files=[pdf_path1.name],
        persist_directory=persist_directory,
    )
    assert retriever is not None

    # 5. Query the retriever and check if it only returns results from the selected file
    query = "Tell me about Succession TV show."
    results = retriever.invoke(query)

    assert len(results) > 0
    assert all(doc.metadata["source"] == pdf_path1.name for doc in results) 
    assert all(doc.metadata["source"] != pdf_path2.name for doc in results)
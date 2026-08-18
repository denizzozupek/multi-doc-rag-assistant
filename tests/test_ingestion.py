import pytest
from src.ingestion import compute_pdf_hash

def test_compute_pdf_hash(tmp_path):

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("This is a test PDF content.")

    pdf_hash = compute_pdf_hash(str(pdf_file))
    assert isinstance(pdf_hash, str)
    assert len(pdf_hash) == 64
    assert pdf_hash == compute_pdf_hash(str(pdf_file))


def test_compute_pdf_hash_file_not_found():
    with pytest.raises(FileNotFoundError):
        compute_pdf_hash("non_existent_file.pdf")

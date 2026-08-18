import pytest
from src.chain import format_docs_as_context
from langchain_core.documents import Document

def test_format_docs_as_context_no_docs():
    docs = None
    result = format_docs_as_context(docs)
    assert result == ""

def test_format_docs_as_context_empty_list():
    docs = []
    result = format_docs_as_context(docs)
    assert result == ""

def test_format_docs_as_context_with_docs():
    docs = [
        Document(page_content="This is the first document."),
        Document(page_content="This is the second document."),
    ]
    result = format_docs_as_context(docs)
    expected_result = "This is the first document.\n\nThis is the second document."
    assert result == expected_result
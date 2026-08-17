import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.retriever import get_retriever

def test_get_retriever_type():
    retriever = get_retriever()

    assert retriever is not None
    assert isinstance(retriever, BaseRetriever)


def test_retriever_retrieves_documents():
    retriever = get_retriever()

    docs = retriever.invoke("What is the capital of France?")

    assert isinstance(docs, list)

    if len(docs) > 0:
        assert isinstance(docs[0], Document)
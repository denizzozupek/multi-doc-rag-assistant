import pytest
from src.retriever import build_filter


def test_build_filter_no_selected_files():
    selected_files = []
    result = build_filter(selected_files)
    assert result is None

def test_build_filter_with_selected_files():
    selected_files = ["file1.pdf", "file2.pdf"]
    result = build_filter(selected_files)
    expected_result = {"source": {"$in": selected_files}}
    assert result == expected_result

def test_build_filter_with_one_selected_files():
    selected_files = ["file1.pdf"]
    result = build_filter(selected_files)
    expected_result = {"source": selected_files[0]}
    assert result == expected_result

def test_build_filter_with_none_selected_files():
    selected_files = None
    result = build_filter(selected_files)
    assert result is None
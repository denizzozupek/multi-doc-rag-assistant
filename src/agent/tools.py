import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.retrieval.pipeline import retrieve_and_rerank, format_docs
from src.retrieval.web import fetch_web_documents

logger = logging.getLogger(__name__)


@tool
async def search_docs(
    query: str,
    selected_files: Annotated[list[str] | None, InjectedState("selected_files")],
    file_name: str | None = None,
) -> str:
    """Searches the documents the user selected for this conversation
    (books, notes, papers, articles etc.) and returns the most relevant passages with
    source file and page. The returned passages are the only allowed
    evidence for an answer.

    Write `query` as a short, self-contained search phrase. For questions
    that compare several documents, call this once per part, using
    `file_name` to target one document. Omit `file_name` to search all
    selected documents.
    """
    allowed = selected_files or []

    if file_name is not None and file_name not in allowed:
        return (
            f"'{file_name}' is not a selected file. "
            f"Selected files: {', '.join(allowed)}"
        )

    scope = [file_name] if file_name else allowed

    try:
        docs = await retrieve_and_rerank(query=query, selected_files=scope)
    except Exception as e:
        logger.exception("search_docs failed")
        return f"Retriever error: {e}"

    if not docs:
        return "No relevant passages found."

    return format_docs(docs)


@tool
async def search_web(query: str) -> str:
    """Searches the web. Use ONLY when the selected documents did not
    contain the answer after searching them, and the user has enabled
    web search. Write `query` as a short, self-contained search phrase.
    Web results are not part of the user's documents; say so in the answer."""
    try:
        docs = await fetch_web_documents(query)
    except Exception as e:
        logger.exception("web_search failed")
        return f"Web search error: {e}"
    if not docs:
        return "No web results found."
    return format_docs(docs)


from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

# Graph State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rewrite_query: str
    documents: list[Document]
    selected_files: list[str] | None = None
    web_search_needed: bool | None = None
    grader_tokens: int | None = None
    grader_cost_usd: float | None = None

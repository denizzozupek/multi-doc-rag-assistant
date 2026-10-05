from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

#Graph State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rewrite_query: str
    documents: list[Document]
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    selected_files: list[str] | None
    total_tokens: Annotated[int, operator.add]
    total_cost_usd: Annotated[float, operator.add]
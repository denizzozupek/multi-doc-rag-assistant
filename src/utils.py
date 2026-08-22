
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def format_history(history: list[dict]) -> list[BaseMessage]:
    """Convert raw dictionary chat history into LangChain BaseMessage objects (HumanMessage / AIMessage)."""
    return [
        (
            HumanMessage(content=item["content"])
            if item["role"] == "user"
            else AIMessage(content=item["content"])
        )
        for item in history
    ]


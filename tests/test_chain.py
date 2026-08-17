import pytest
from src.chain import conversation_history
def test_chain_initialization():
    chain = conversation_history()
    assert chain is not None

def test_chain_invoke():
    config = {"configurable": {"session_id": "test_session"}}

    chain = conversation_history()

    result = chain.invoke({"input": "Hello, how are you?"}, config=config)

    assert result is not None

    content = result.content if hasattr(result, 'content') else str(result)
    assert len(content.strip()) > 0
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    Runnable,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.retrievers import BaseRetriever

from src.retriever import get_retriever
from src.prompt import memory_search_system_prompt, system_prompt
import logging

logger = logging.getLogger(__name__)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=3,
)


def format_docs_as_context(docs) -> str:
    try:
        if not docs:
            logger.warning("No documents retrieved. Returning empty context.")
            return "No relevant context found."
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(
            f"Error occurred while formatting documents as context: {e}", exc_info=True
        )
        raise


def history_search_chain() -> Runnable:
    """Returns a Runnable that generates a search query from the chat history and the latest user message."""
    memory_search_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", memory_search_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return memory_search_prompt | llm | StrOutputParser()


def qa_chain(retriever=None):
    """Returns a Runnable that answers a question using the provided context."""

    if retriever is None:
        try:
            logger.info("No retriever provided. Attempting to get retriever...")
            retriever = get_retriever()
        except Exception as e:
            logger.error(f"Error occurred while getting retriever: {e}", exc_info=True)
            raise

    memory_chain = history_search_chain()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # RAG CHAIN (LCEL CHAIN)
    rag_chain = (
        RunnablePassthrough.assign(
            context=memory_chain | retriever | format_docs_as_context
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


store = {}


def get_chat_history(session_id) -> BaseChatMessageHistory:
    try:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    except Exception as e:
        logger.error(f"Error occurred while retrieving chat history: {e}", exc_info=True)
        raise

def conversation_history(retriever: Optional[BaseRetriever] = None) -> RunnableWithMessageHistory:
    try:
        qa_rag_chain = qa_chain(retriever=retriever)

        conversation_history = RunnableWithMessageHistory(
            qa_rag_chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return conversation_history
    except Exception as e:
        logger.error(f"Error occurred while creating conversation history: {e}", exc_info=True)
        raise
    

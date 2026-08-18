from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    Runnable,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.retrievers import BaseRetriever

from src.retriever import get_retriever
from src.prompt import memory_search_system_prompt, system_prompt
from src.config import REDIS_URL
import logging
import os


logger = logging.getLogger(__name__)


DEFAULT_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=3,
)


def format_docs_as_context(docs: list[Document] | None) -> str:
    try:
        if not docs:
            logger.warning("No documents retrieved. Returning empty context.")
            return ""
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(
            f"Error occurred while formatting documents as context: {e}", exc_info=True
        )
        raise


def history_search_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Returns a Runnable that generates a search query from the chat history and the latest user message."""

    if llm is None:
        llm = DEFAULT_LLM

    memory_search_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", memory_search_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return memory_search_prompt | llm | StrOutputParser()


def qa_chain(
    retriever: BaseRetriever | None = None, llm: BaseChatModel | None = None
) -> Runnable:
    """Returns a Runnable that answers a question using the provided context."""
    if llm is None:
        llm = DEFAULT_LLM

    if retriever is None:
        try:
            logger.info("No retriever provided. Attempting to get retriever...")
            retriever = get_retriever()
            if retriever is None:
                logger.error(
                    "Failed to obtain a retriever. Please check your configuration."
                )
                raise ValueError("Retriever could not be obtained.")
        except Exception as e:
            logger.error(f"Error occurred while getting retriever: {e}", exc_info=True)
            raise

    memory_chain = history_search_chain(llm=llm)

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


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    try:
        redis_url = REDIS_URL
        return RedisChatMessageHistory(session_id=session_id, url=redis_url)
    except Exception as e:
        logger.error(
            f"Error occurred while initializing RedisChatMessageHistory: {e}",
            exc_info=True,
        )
        raise


def conversation_history(
    retriever: BaseRetriever | None = None, llm: BaseChatModel | None = None
) -> RunnableWithMessageHistory:
    if llm is None:
        llm = DEFAULT_LLM
    try:
        qa_rag_chain = qa_chain(retriever=retriever, llm=llm)

        conversation_history = RunnableWithMessageHistory(
            qa_rag_chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return conversation_history
    except Exception as e:
        logger.error(
            f"Error occurred while creating conversation history: {e}", exc_info=True
        )
        raise

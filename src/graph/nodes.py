import asyncio
import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import (
    DEFAULT_RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVER_SEARCH_TYPE,
)
from src.config import DEFAULT_LLM
from src.prompt import memory_search_system_prompt, system_prompt
from src.graph.state import State
from src.retrieval.reranker import reranker
from src.retrieval.retriever import get_retriever
from src.utils import format_docs_as_context

logger = logging.getLogger(__name__)


# 1. Rewrite Query Node
def create_rewrite_query_node(llm: BaseChatModel) -> Callable[[State], Any]:
    """Creates a node that rewrites the user's query based on the chat history and the latest user message."""

    memory_search_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", memory_search_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = memory_search_prompt | llm | StrOutputParser()

    async def rewrite_query_node(state: State) -> dict:
        """Reads chat history and latest message from state, rewrites the query using LLM, and returns it."""
        if "messages" not in state or not state["messages"]:
            raise ValueError("State must contain 'messages' with at least one message.")

        last_content = state["messages"][-1].content
        human_input = (
            last_content if isinstance(last_content, str) else str(last_content)
        )

        if not human_input or not human_input.strip():
            logger.warning(
                "No user input found in the last message. Returning empty rewrite query."
            )
            return {"rewrite_query": ""}

        chat_history = state["messages"][:-1]

        llm_response = await chain.ainvoke(
            {"chat_history": chat_history, "input": human_input}
        )
        logger.info("Rewritten query generated successfully: %s", llm_response)

        return {"rewrite_query": llm_response}

    logger.info("Rewrite query node created successfully.")
    return rewrite_query_node


# 2. Retrieve and Rerank Node
def create_retrieve_and_rerank_node(
    model: Any = None,
    candidate_k: int = RETRIEVAL_CANDIDATE_K,
    top_k: int = DEFAULT_RERANK_TOP_K,
    search_type: str = RETRIEVER_SEARCH_TYPE,
) -> Callable[[State], Any]:
    """Creates a node that retrieves documents dynamically based on selected files and reranks them."""

    async def retrieve_and_rerank_node(state: State) -> dict:
        """Retrieves and reranks documents based on runtime state, returning the top_k results."""
        query = state.get("rewrite_query")
        selected_files = state.get("selected_files")

        if not selected_files:
            logger.warning(
                "No selected files provided. Using all available documents for retrieval."
            )

        if not query or not query.strip():
            logger.warning("No query provided. Returning empty document list.")
            return {"documents": []}

        retriever = get_retriever(
            k=candidate_k,
            selected_files=selected_files,
            search_type=search_type,
        )
        if retriever is None:
            logger.warning(
                "Retriever could not be created. Returning empty document list."
            )
            return {"documents": []}

        raw_docs = await retriever.ainvoke(query)
        logger.info("Retrieved documents count: %s", len(raw_docs))

        if not raw_docs:
            logger.warning("No documents retrieved.")
            return {"documents": []}

        # Rerank retrieved candidate documents using Cross-Encoder model
        reranked_docs = await asyncio.to_thread(
            reranker,
            query=query,
            documents=raw_docs,
            model=model,
            top_k=top_k,
        )

        return {"documents": reranked_docs}

    logger.info("Retrieve and rerank node created successfully.")
    return retrieve_and_rerank_node


# 3. Generation Node
def create_generation_node(
    llm: BaseChatModel | None = None,
) -> Callable[[State], Any]:
    """Creates a node that answers a question using the provided context."""
    if llm is None:
        llm = DEFAULT_LLM

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm

    async def generation_node(state: State) -> dict:
        """Answers a question using the provided context and returns the answer in a dictionary."""
        documents = state.get("documents", [])
        messages = state.get("messages", [])

        if not messages:
            logger.warning("No messages found in state. Returning empty response.")
            return {
                "messages": [
                    AIMessage(content="No messages found to generate a response.")
                ]
            }

        if documents:
            context = format_docs_as_context(documents)
            response = await chain.ainvoke({"context": context, "messages": messages})
            return {"messages": [response]}

        logger.warning("No documents found in state. Returning default response.")
        return {
            "messages": [
                AIMessage(content="No relevant documents found to answer the question.")
            ]
        }

    logger.info("Generation node created successfully.")
    return generation_node
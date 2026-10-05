from typing import Callable, Any
import logging

from src.retrieval.reranker import reranker
from state import State
from generation.prompt import memory_search_system_prompt

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1. Rewrite Query Node
def create_rewrite_query_node(llm: BaseChatModel) -> Callable[[State], dict]:
    """Creates a node that rewrites the user's query based on the chat history and the latest user message."""

    memory_search_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", memory_search_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = memory_search_prompt | llm | StrOutputParser()

    def rewrite_query_node(state: State) -> dict:
        """Reads the chat history and the latest user message from the state, generates a rewritten query using the LLM, and returns it in a dictionary."""

        if "messages" not in state or not state["messages"]:
            raise ValueError("State must contain 'messages' with at least one message.")

        last_content = state["messages"][-1].content
        human_input = (
            last_content if isinstance(last_content, str) else str(last_content)
        )

        chat_history = state["messages"][:-1]

        llm_response = chain.invoke(
            {"chat_history": chat_history, "input": human_input}
        )
        logger.info("Rewritten query generated successfully: %s", llm_response)

        return {"rewrite_query": llm_response}

    logger.info("Rewrite query node created successfully.")
    return rewrite_query_node


# 2. Retrieve and Rerank Node
def create_retrieve_and_rerank_node(
    retriever: BaseRetriever, model: Any = None, top_k: int = 3
) -> Callable[[State], dict]:
    """Creates a node that retrieves documents using the provided retriever and reranks them using the specified model."""

    def retrieve_and_rerank_node(state: State) -> dict:
        """Retrieves documents using the provided retriever and reranks them using the specified model, then returns the top_k documents in a dictionary."""


        query = state.get("rewrite_query")

        if not query:
            messages = state.get("messages", [])
            last_content = messages[-1].content if messages else ""
            query = last_content if isinstance(last_content, str) else str(last_content)

        if not query or not query.strip():
            logger.warning("No query provided. Returning empty document list.")
            return {"documents": []}
           
        raw_docs = retriever.invoke(query)
        logger.info("Retrieved documents: %s", len(raw_docs))

        if not raw_docs:
            logger.warning("No documents retrieved.")
            return {"documents": []}

        # Rerank the documents if a model is provided
        if model:
            reranked_docs = reranker(
                query=query, documents=raw_docs, model=model, top_k=top_k
            )
        else:
            reranked_docs = raw_docs[:top_k]

        return {"documents": reranked_docs}

    logger.info("Retrieve and rerank node created successfully.")
    return retrieve_and_rerank_node


from typing import Callable, Any
import logging

from langchain_core.messages import AIMessage

from src.generation.chain import DEFAULT_LLM
from src.retrieval.reranker import reranker
from state import State
from generation.prompt import memory_search_system_prompt, system_prompt
from utils import format_docs_as_context

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

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

        if human_input is None or not human_input.strip():
            logger.warning("No user input found in the last message. Returning empty rewrite query.")
            return {"rewrite_query": ""}

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

    if retriever is None:
        raise ValueError("Retriever must be provided to create the retrieve and rerank node.")

    def retrieve_and_rerank_node(state: State) -> dict:
        """Retrieves documents using the provided retriever and reranks them using the specified model, then returns the top_k documents in a dictionary."""


        query = state.get("rewrite_query")

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

def create_generation_node(
    llm: BaseChatModel | None = None,
) -> Callable[[State], dict]:
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

    def generation_node(state: State) -> dict:
        """Answers a question using the provided context and returns the answer in a dictionary."""

        documents = state.get("documents", [])
        messages = state.get("messages", [])

        if not messages:
            logger.warning("No messages found in state. Returning empty response.")
            return {"messages": [AIMessage(content="No messages found to generate a response.")]}

        if documents:
            context = format_docs_as_context(documents)
            response = chain.invoke({"context": context, "messages": messages})
            return {"messages": [response]}
        else:
            logger.warning("No documents found in state. Returning default response.")
            return {"messages": [AIMessage(content="No relevant documents found to answer the question.")]}

    logger.info("Generation node created successfully.")
    return generation_node
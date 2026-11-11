import asyncio
import logging
from typing import Any, Callable

from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

from src.config import (
    DEFAULT_RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVER_SEARCH_TYPE,
)
from src.config import DEFAULT_LLM
from src.prompt import (
    RetrievalEvaluator,
    memory_search_system_prompt,
    system_prompt,
    corrective_rag_prompt,
    rewrite_for_web_search_prompt,
)
from src.graph.state import State
from src.retrieval.pipeline import retrieve_and_rerank
from src.retrieval.web import fetch_web_documents
from src.utils import format_docs_as_context
from src.agent.nodes import create_agent_node

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
        docs = await retrieve_and_rerank(
            query=state.get("rewrite_query"),
            selected_files=state.get("selected_files"),
            model=model,
            candidate_k=candidate_k,
            top_k=top_k,
            search_type=search_type,
        )
        return {"documents": docs}

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


# 4. Grade Documents Node
def create_grade_documents_node(
    llm: BaseChatModel = DEFAULT_LLM,
) -> Callable[[State], Any]:
    """Creates a node that evaluates the relevancy of retrieved documents to the user's question."""

    evaluator_chain = corrective_rag_prompt | llm.with_structured_output(
        RetrievalEvaluator
    )

    async def grade_documents_node(state: State) -> dict:
        documents = state.get("documents", [])
        query = state.get("rewrite_query") or state.get("messages", [])[-1].content

        if not documents:
            logger.warning("No documents found in state. Directed to web search.")
            return {
                "documents": [],
                "web_search_needed": True,
                "grader_tokens": 0,
                "grader_cost_usd": 0.0,
            }

        tasks = [
            evaluator_chain.ainvoke({"source_doc": doc.page_content, "query": query})
            for doc in documents
        ]
        with get_openai_callback() as cb:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            grader_tokens = cb.total_tokens
            grader_cost_usd = round(cb.total_cost, 6)

        filtered_docs = [
            doc
            for doc, result in zip(documents, results)
            if not isinstance(result, Exception)
            and getattr(result, "binary_score", "").lower() == "yes"
        ]

        web_search_needed = len(filtered_docs) == 0
        if web_search_needed:
            logger.info("No relevant documents found after grading. Web search needed.")
        else:
            logger.info(
                "Grading complete: %d of %d documents retained.",
                len(filtered_docs),
                len(documents),
            )

        return {
            "documents": filtered_docs,
            "web_search_needed": web_search_needed,
            "grader_tokens": grader_tokens,
            "grader_cost_usd": grader_cost_usd,
        }

    return grade_documents_node


# 5. Web Search Node
def create_web_search_node(llm: BaseChatModel = DEFAULT_LLM) -> Callable[[State], Any]:
    """Creates a node that rewrites the query for web search and fetches web results."""

    chain = rewrite_for_web_search_prompt | llm | StrOutputParser()

    async def web_search_node(state: State) -> dict:
        query = state.get("rewrite_query") or state["messages"][-1].content
        if not query or not query.strip():
            return {"rewrite_query": ""}

        rewritten_query = await chain.ainvoke({"user_question": query})
        logger.info("Rewritten web search query: %s", rewritten_query)

        new_docs = await fetch_web_documents(rewritten_query)
        return {"documents": [*state.get("documents", []), *new_docs]}

    return web_search_node

# 6. Fallback Agent Node for Escalate Mode
def fallback_agent_node(llm=DEFAULT_LLM):
    """Creates a fallback agent node if grade_documents_node fails to find relevant documents"""
    return create_agent_node(llm=llm)
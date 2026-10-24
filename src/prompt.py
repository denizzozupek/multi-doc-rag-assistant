from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RetrievalEvaluator(BaseModel):
    """Classify retrieved documents as relevant or not relevant to the user's question."""

    binary_score: str = Field(
        description="If document is relevant to the user's question, return 'yes', otherwise return 'no'."
    )


memory_search_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

system_prompt = """
Using the following context, answer the question below. If the answer is not contained within the context, respond with "I don't know." Do not make up an answer.
{context}
"""

corrective_rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question.\n\n"
            "If the document contains information that is relevant to the user's question, return 'yes'. If the document does not contain relevant information, return 'no'.\n\n"
            ""
        ),
        ("human", "Retrieved Document: {source_doc}\n\nUser's Question: {query}")
    ]
)


rewrite_for_web_search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that reformulates user questions for web search. "
            "Given a user question, rewrite it to be more suitable for web search. "
            "Make sure the rewritten question is clear, concise, and retains the original intent of the user's question."
        ),("human", "{user_question}")
    ]
)
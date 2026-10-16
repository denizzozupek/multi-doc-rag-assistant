memory_search_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

system_prompt = """
Using the following context, answer the question below. If the answer is not contained within the context, respond with "I don't know." Do not make up an answer.
{context}
"""
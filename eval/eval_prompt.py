def build_judge_prompt(query: str, context: str, response: str) -> str:
    return f"""You are an expert evaluator assessing an AI assistant's response in a Retrieval-Augmented Generation (RAG) system.

Evaluate the response strictly based on the provided context across two distinct criteria:

1. FAITHFULNESS (1-5):
- 5: Every single claim in the response is directly and explicitly supported by the context. If the context does not contain the answer and the assistant correctly states that it cannot answer based on the context, give a score of 5.
- 4: The core answer is fully supported, but contains minor phrasing extrapolation that does not alter facts.
- 3: Partially supported; contains at least one unsupported claim or minor hallucination alongside correct facts.
- 2: Mostly unsupported; significant claims contradict or cannot be found in the context.
- 1: Severe hallucination; entirely fabricated or directly contradicts the context.

2. RELEVANCE (1-5):
- 5: Directly and fully addresses the user's question. If the question is unanswerable from the context, a clear refusal ("Information not found in context") is considered fully relevant (score 5).
- 4: Answers the main question but includes slight irrelevant context or redundant details.
- 3: Answers only part of the question or addresses it vaguely.
- 2: Mostly off-topic; talks around the subject without answering the core question.
- 1: Completely off-topic or fails to address the question.

QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

ASSISTANT'S RESPONSE:
{response}

INSTRUCTIONS:
- First, write your reasoning by analyzing the context and claims step-by-step.
- Second, assign the integer score based strictly on your reasoning.
- Respond ONLY with a valid JSON object. Do not include markdown codeblocks or any additional text.

{{
    "faithfulness_reason": "<step-by-step verification of claims against context>",
    "faithfulness_score": <int 1-5>,
    "relevance_reason": "<analysis of how directly the response addresses the prompt>",
    "relevance_score": <int 1-5>
}}"""
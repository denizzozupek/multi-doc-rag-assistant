import datetime
import json
import logging
from dotenv import load_dotenv
from eval.eval_prompt import build_judge_prompt
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.chain import qa_chain

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("data/ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

chain = qa_chain(return_context=True)


class JudgeEvaluationPrompt(BaseModel):
    faithfulness_reason: str
    faithfulness_score: int = Field(..., ge=1, le=5)

    relevance_reason: str
    relevance_score: int = Field(..., ge=1, le=5)


judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=3,
).with_structured_output(JudgeEvaluationPrompt)

results = []

for item in ground_truth:
    query_text = item["query"]
    logger.info(f"Evaluating query: {query_text}")
    try:
        result = chain.invoke({"input": query_text, "chat_history": []})
        prompt = build_judge_prompt(
            query=query_text,
            context=result["context"],
            response=result["answer"],
        )
        judge_result = judge_llm.invoke(prompt)
        results.append(
            {
                "id": item.get("id"),
                "query": query_text,
                "context": result["context"],
                "answer": result["answer"],
                "judge_result": judge_result.model_dump(),
            }
        )
    except Exception as e:
        logger.error(f"Error evaluating query {query_text}: {e}", exc_info=True)
        results.append(
            {
                "id": item.get("id"),
                "query": query_text,
                "context": None,
                "answer": None,
                "judge_result": {
                    "faithfulness_reason": str(e),
                    "faithfulness_score": 0,
                    "relevance_reason": str(e),
                    "relevance_score": 0,
                },
            }
        )

run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

with open(f"data/judge_results_{run_id}.json", "w", encoding="utf-8") as f:
    logger.info(f"Saving judge results to data/judge_results_{run_id}.json")
    json.dump(results, f, indent=4, ensure_ascii=False)

valid_results = [
    r for r in results if r["context"] is not None and r["answer"] is not None
]

if len(valid_results) > 0:
    avg_faithfulness = sum(
        r["judge_result"]["faithfulness_score"] for r in valid_results
    ) / len(valid_results)
    avg_relevance = sum(
        r["judge_result"]["relevance_score"] for r in valid_results
    ) / len(valid_results)

    logger.info(
        f"Average Faithfulness Score: {avg_faithfulness:.2f} ({len(valid_results)}/{len(results)} valid)"
    )
    logger.info(
        f"Average Relevance Score: {avg_relevance:.2f} ({len(valid_results)}/{len(results)} valid)"
    )
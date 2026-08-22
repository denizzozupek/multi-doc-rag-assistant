import datetime
import json
import logging
from dotenv import load_dotenv
from eval.eval_prompt import build_judge_prompt
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from src.chain import qa_chain
from src.utils import format_history

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------- Initalize the QA chain and load ground truth data -------
with open("data/ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

chain = qa_chain(return_context=True)


# ------- Define the structured output model for the judge LLM -------
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


# ------- Evaluate Queries and Save Results -------
def evaluate_queries(
    judge_llm: ChatOpenAI, chain: Runnable, ground_truth: list[dict]
) -> list[dict]:
    results = []
    for item in ground_truth:
        query_text = item["query"]
        logger.info(f"Evaluating query: {query_text}")

        try:
            raw_history = item.get("conversation_history", [])
            raw_history_formatted = format_history(raw_history)
            result = chain.invoke({"input": query_text, "chat_history": raw_history_formatted})
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
                    "question_type": item.get("question_type"),
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
    return results


run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def save_judge_results(results: list[dict], run_id: str):
    """Save judge evaluation results to a JSON file."""
    valid_results = [
        r for r in results if r["context"] is not None and r["answer"] is not None
    ]

    total_questions = len(results)
    total_valid = len(valid_results)

    if total_valid > 0:
        avg_faithfulness = (
            sum(r["judge_result"]["faithfulness_score"] for r in valid_results)
            / total_valid
        )
        avg_relevance = (
            sum(r["judge_result"]["relevance_score"] for r in valid_results)
            / total_valid
        )

        logger.info(
            f"Average Faithfulness Score: {avg_faithfulness:.2f} ({total_valid}/{total_questions} valid)"
        )
        logger.info(
            f"Average Relevance Score: {avg_relevance:.2f} ({total_valid}/{total_questions} valid)"
        )
    else:
        avg_faithfulness = 0
        avg_relevance = 0

    multiturn_questions = [r for r in valid_results if r["question_type"] == "multi_turn_followup"]
    total_multiturn = len(multiturn_questions)

    if multiturn_questions:
        avg_multiturn_faithfulness = (
            sum(r["judge_result"]["faithfulness_score"] for r in multiturn_questions)
            / total_multiturn
        )
        avg_multiturn_relevance = (
            sum(r["judge_result"]["relevance_score"] for r in multiturn_questions)
            / total_multiturn
        )

        logger.info(
            f"Average Multi-turn Faithfulness Score: {avg_multiturn_faithfulness:.2f} ({total_multiturn}/{total_questions} multi-turn)"
        )
        logger.info(
            f"Average Multi-turn Relevance Score: {avg_multiturn_relevance:.2f} ({total_multiturn}/{total_questions} multi-turn)"
        )
    else:
        avg_multiturn_faithfulness = None
        avg_multiturn_relevance = None

    output_data = {
        "total_questions": total_questions,
        "total_valid": total_valid,
        "total_multiturn": total_multiturn,
        "summary": {
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_multiturn_faithfulness": avg_multiturn_faithfulness,
            "avg_multiturn_relevance": avg_multiturn_relevance,
        },
        "detailed_results": results,
    }

    with open(f"eval/evaluation_results/judge_results_{run_id}.json", "w", encoding="utf-8") as f:
        logger.info(f"Saving judge results to eval/evaluation_results/judge_results_{run_id}.json")
        json.dump(output_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    results = evaluate_queries(judge_llm, chain, ground_truth)
    save_judge_results(results, run_id)

import datetime
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from eval.generation.prompts import build_judge_prompt
from eval.generation.schemas import JudgeEvaluationPrompt
from src.graph import graph_pipeline
from src.ingestion import ingestion_pipeline
from src.utils import format_docs_as_context, format_history

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------- Evaluate Queries and Save Results -------
def evaluate_queries(
    judge_llm,
    ground_truth: list[dict],
    top_k: int = 3,
) -> list[dict]:
    results = []
    app = graph_pipeline(top_k=top_k).compile()

    for item in ground_truth:
        query_text = item["query"]
        file_name = item.get("source_doc")
        logger.info(f"Evaluating query: {query_text} (Source: {file_name})")

        try:
            raw_history = item.get("conversation_history", [])
            raw_history_formatted = format_history(raw_history)
            state_input = {
                "messages": raw_history_formatted + [HumanMessage(content=query_text)],
                "selected_files": [file_name] if file_name else None,
            }

            output = app.invoke(state_input)
            retrieved_docs = output.get("documents", [])
            context = format_docs_as_context(retrieved_docs)
            answer = output["messages"][-1].content

            # LLM Judge: Evaluate the answer for faithfulness and relevance
            prompt = build_judge_prompt(
                query=query_text,
                context=context,
                response=answer,
            )
            judge_result = judge_llm.invoke(prompt)

            results.append(
                {
                    "id": item.get("id"),
                    "query": query_text,
                    "source_doc": file_name,
                    "question_type": item.get("question_type"),
                    "top_k": top_k,
                    "context": context,
                    "answer": answer,
                    "judge_result": judge_result.model_dump(),
                }
            )
        except Exception as e:
            logger.error(f"Error evaluating query '{query_text}': {e}", exc_info=True)
            results.append(
                {
                    "id": item.get("id"),
                    "query": query_text,
                    "source_doc": file_name,
                    "question_type": item.get("question_type"),
                    "top_k": top_k,
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


def save_judge_results(results: list[dict], run_id: str):
    """Save judge evaluation results to a JSON file and log summary statistics."""
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
        avg_faithfulness = 0.0
        avg_relevance = 0.0

    multiturn_questions = [
        r for r in valid_results if r["question_type"] == "multi_turn_followup"
    ]
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

    output_path = f"eval/results/generation/judge_results_{run_id}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        logger.info(f"Saving judge results to {output_path}")
        json.dump(output_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    ingestion_pipeline(pdf_path="data/arxiv1.pdf")
    ingestion_pipeline(pdf_path="data/arxiv2.pdf")

    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        timeout=30,
        max_retries=3,
    ).with_structured_output(JudgeEvaluationPrompt)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = evaluate_queries(
        judge_llm=judge_llm,
        ground_truth=ground_truth,
        top_k=3,
    )
    save_judge_results(results, run_id)

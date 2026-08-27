import datetime
import json
import logging
from dotenv import load_dotenv
from eval.eval_prompt import build_judge_prompt
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.generation.chain import qa_chain
from src.ingestion.ingestion import ingestion_pipeline
from src.utils import format_history
from src.retrieval.retriever import get_retriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------- Define the structured output model for the judge LLM -------
class JudgeEvaluationPrompt(BaseModel):
    faithfulness_reason: str
    faithfulness_score: int = Field(..., ge=1, le=5)

    relevance_reason: str
    relevance_score: int = Field(..., ge=1, le=5)


# ------- Evaluate Queries and Save Results -------
def evaluate_queries(
    judge_llm,
    ground_truth: list[dict],
    k: int = 15,
    fetch_k: int = 20,
    top_k: int = 3,
    search_type: str = "hybrid",
    use_reranker: bool = True,
) -> list[dict]:
    results = []
    
    for item in ground_truth:
        query_text = item["query"]
        file_name = item.get("source_doc")
        logger.info(f"Evaluating query: {query_text} (Source: {file_name})")

        try:
            # 1. For every query, retrieve documents using the retriever
            retriever = get_retriever(
                k=k,
                fetch_k=fetch_k,
                selected_files=[file_name] if file_name else None,
                search_type=search_type,
            )
            
            if retriever is None:
                raise ValueError(f"Retriever could not be created for file: {file_name}")

            # Pass use_reranker flag to qa_chain
            chain = qa_chain(
                retriever=retriever,
                return_context=True,
                top_k=top_k,
                use_reranker=use_reranker,
            )

            # 2. Run the query through the chain to get the answer and context
            raw_history = item.get("conversation_history", [])
            raw_history_formatted = format_history(raw_history)
            result = chain.invoke(
                {"input": query_text, "chat_history": raw_history_formatted}
            )

            # 3. LLM Judge: Evaluate the answer for faithfulness and relevance
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
                    "source_doc": file_name,
                    "question_type": item.get("question_type"),
                    "use_reranker": use_reranker,
                    "context": result["context"],
                    "answer": result["answer"],
                    "judge_result": judge_result.model_dump(),
                }
            )
        except Exception as e:
            logger.error(f"Error evaluating query '{query_text}': {e}", exc_info=True)
            # Append a result with error details for this query
            results.append(
                {
                    "id": item.get("id"),
                    "query": query_text,
                    "source_doc": file_name,
                    "question_type": item.get("question_type"),
                    "use_reranker": use_reranker,
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

    output_path = f"eval/evaluation_results/judge_results_{run_id}.json"
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

    # Run evaluation with the best configuration from retrieval benchmarking
    results = evaluate_queries(
        judge_llm=judge_llm,
        ground_truth=ground_truth,
        k=15,
        fetch_k=20,
        top_k=5,
        search_type="hybrid",
        use_reranker=True,
    )
    save_judge_results(results, run_id)
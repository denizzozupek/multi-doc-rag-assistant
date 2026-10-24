import datetime
import json
import logging
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from time import perf_counter

from eval.generation.prompts import build_judge_prompt
from eval.generation.schemas import (
    GenerationEvaluationItem,
    GenerationEvaluationReport,
    GenerationSummary,
    JudgeEvaluationPrompt,
)
from src.config import DEFAULT_LLM, CONCURRENT_SEMAPHORE, DEFAULT_RERANK_TOP_K
from src.graph import graph_pipeline
from src.ingestion import ingestion_pipeline
from src.utils import calculate_percentile, format_docs_as_context, format_history

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(CONCURRENT_SEMAPHORE)


# ------- Evaluate Queries and Save Results -------


async def evaluate_query_with_semaphore(app, judge_llm, item, top_k):
    async with semaphore:
        query_text = item["query"]
        file_name = item.get("source_doc")
        logger.info(f"Evaluating query: {query_text} (Source: {file_name})")

        total_start = perf_counter()
        try:
            raw_history = item.get("conversation_history", [])
            raw_history_formatted = format_history(raw_history)
            selected_files = [s.strip() for s in file_name.split(",")] if file_name else None
            state_input = {
                "messages": raw_history_formatted + [HumanMessage(content=query_text)],
                "selected_files": selected_files,
            }

            gen_start = perf_counter()
            output = await app.ainvoke(state_input)
            generation_time_ms = round((perf_counter() - gen_start) * 1000, 2)

            retrieved_docs = output.get("documents", [])
            context = format_docs_as_context(retrieved_docs)
            answer = output["messages"][-1].content

            # LLM Judge: Evaluate the answer for faithfulness and relevance
            prompt = build_judge_prompt(
                query=query_text,
                context=context,
                response=answer,
            )
            judge_start = perf_counter()
            judge_result = await judge_llm.ainvoke(prompt)
            judge_time_ms = round((perf_counter() - judge_start) * 1000, 2)
            total_time_ms = round((perf_counter() - total_start) * 1000, 2)

            return {
                "id": item.get("id"),
                "query": query_text,
                "source_doc": file_name,
                "question_type": item.get("question_type"),
                "top_k": top_k,
                "context": context,
                "answer": answer,
                "generation_latency_ms": generation_time_ms,
                "judge_latency_ms": judge_time_ms,
                "total_latency_ms": total_time_ms,
                "judge_result": judge_result.model_dump(),
            }
        except Exception as e:
            logger.error(f"Error evaluating query '{query_text}': {e}", exc_info=True)
            total_time_ms = round((perf_counter() - total_start) * 1000, 2)
            return {
                "id": item.get("id"),
                "query": query_text,
                "source_doc": file_name,
                "question_type": item.get("question_type"),
                "top_k": top_k,
                "context": None,
                "answer": None,
                "generation_latency_ms": 0.0,
                "judge_latency_ms": 0.0,
                "total_latency_ms": total_time_ms,
                "judge_result": {
                    "faithfulness_reason": str(e),
                    "faithfulness_score": 0,
                    "relevance_reason": str(e),
                    "relevance_score": 0,
                },
            }


async def evaluate_queries(
    judge_llm,
    ground_truth: list[dict],
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[dict]:
    """Evaluate a list of queries using the graph pipeline and judge LLM."""

    app = graph_pipeline(llm=DEFAULT_LLM, top_k=top_k).compile()

    tasks = [
        evaluate_query_with_semaphore(app, judge_llm, item, top_k)
        for item in ground_truth
    ]
    results = await asyncio.gather(*tasks)
    return results


def save_judge_results(results: list[dict], run_id: str) -> GenerationEvaluationReport:
    """Save judge evaluation results to a JSON file and log summary statistics with latencies."""
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
        avg_gen_ms = (
            sum(r["generation_latency_ms"] for r in valid_results) / total_valid
        )
        avg_judge_ms = (
            sum(r["judge_latency_ms"] for r in valid_results) / total_valid
        )
        avg_total_ms = (
            sum(r["total_latency_ms"] for r in valid_results) / total_valid
        )
        p95_gen_ms = calculate_percentile(
            [r["generation_latency_ms"] for r in valid_results], 0.95
        )
        p95_total_ms = calculate_percentile(
            [r["total_latency_ms"] for r in valid_results], 0.95
        )

        logger.info(
            f"Average Faithfulness Score: {avg_faithfulness:.2f} ({total_valid}/{total_questions} valid)"
        )
        logger.info(
            f"Average Relevance Score: {avg_relevance:.2f} ({total_valid}/{total_questions} valid)"
        )
        logger.info(
            f"Average Generation Latency: {avg_gen_ms:.2f} ms | P95: {p95_gen_ms:.2f} ms"
        )
        logger.info(
            f"Average Total Latency: {avg_total_ms:.2f} ms | P95: {p95_total_ms:.2f} ms"
        )
    else:
        avg_faithfulness = 0.0
        avg_relevance = 0.0
        avg_gen_ms = 0.0
        avg_judge_ms = 0.0
        avg_total_ms = 0.0
        p95_gen_ms = 0.0
        p95_total_ms = 0.0

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

    summary = GenerationSummary(
        avg_faithfulness=round(avg_faithfulness, 2),
        avg_relevance=round(avg_relevance, 2),
        avg_multiturn_faithfulness=round(avg_multiturn_faithfulness, 2)
        if avg_multiturn_faithfulness is not None
        else None,
        avg_multiturn_relevance=round(avg_multiturn_relevance, 2)
        if avg_multiturn_relevance is not None
        else None,
        avg_generation_latency_ms=round(avg_gen_ms, 2),
        avg_judge_latency_ms=round(avg_judge_ms, 2),
        avg_total_latency_ms=round(avg_total_ms, 2),
        p95_generation_latency_ms=round(p95_gen_ms, 2),
        p95_total_latency_ms=round(p95_total_ms, 2),
    )

    report = GenerationEvaluationReport(
        total_questions=total_questions,
        total_valid=total_valid,
        total_multiturn=total_multiturn,
        summary=summary,
        detailed_results=[GenerationEvaluationItem(**r) for r in results],
    )

    output_path = f"eval/results/generation/judge_results_{run_id}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        logger.info(f"Saving judge results to {output_path}")
        json.dump(report.model_dump(), f, indent=4, ensure_ascii=False)

    return report


if __name__ == "__main__":
    ingestion_pipeline(pdf_path="data/arxiv1.pdf")
    ingestion_pipeline(pdf_path="data/arxiv2.pdf")
    ingestion_pipeline(pdf_path="data/arxiv3.pdf")

    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    judge_llm = DEFAULT_LLM.with_structured_output(JudgeEvaluationPrompt)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = asyncio.run(
        evaluate_queries(judge_llm, ground_truth, top_k=DEFAULT_RERANK_TOP_K)
    )
    save_judge_results(results, run_id)


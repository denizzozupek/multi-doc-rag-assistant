import datetime
import json
import logging
import asyncio
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from eval.generation.prompts import build_judge_prompt
from eval.generation.schemas import (
    GenerationEvaluationItem,
    GenerationEvaluationReport,
    GenerationSummary,
    JudgeEvaluationPrompt,
    RunMetadata,
)
from src.config import DEFAULT_LLM, CONCURRENT_SEMAPHORE, DEFAULT_RERANK_TOP_K
from src.graph import graph_pipeline
from src.ingestion import ingestion_pipeline
from src.utils import calculate_percentile, format_docs_as_context, format_history
from src.retrieval.reranker import get_cross_encoder_model

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(CONCURRENT_SEMAPHORE)


# ------- Evaluate Queries and Save Results -------


async def evaluate_query_with_semaphore(app, judge_llm, item, top_k, crag_enable=True):
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

            # 1. RAG Pipeline Execution (Rewrite, Retrieval, Grading if CRAG, Generation)
            with get_openai_callback() as rag_cb:
                gen_start = perf_counter()
                output = await app.ainvoke(state_input)
                generation_time_ms = round((perf_counter() - gen_start) * 1000, 2)

            retrieved_docs = output.get("documents", [])
            context = format_docs_as_context(retrieved_docs)
            answer = output["messages"][-1].content

            # 2. LLM Judge Execution (Independent Evaluation)
            prompt = build_judge_prompt(
                query=query_text,
                context=context,
                response=answer,
            )
            with get_openai_callback() as judge_cb:
                judge_start = perf_counter()
                judge_result = await judge_llm.ainvoke(prompt)
                judge_time_ms = round((perf_counter() - judge_start) * 1000, 2)

            total_time_ms = round((perf_counter() - total_start) * 1000, 2)

            rag_tokens = rag_cb.total_tokens
            rag_cost = round(rag_cb.total_cost, 6)
            judge_tokens = judge_cb.total_tokens
            judge_cost = round(judge_cb.total_cost, 6)
            total_tokens = rag_tokens + judge_tokens
            cost_usd = round(rag_cost + judge_cost, 6)

            result = {
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
                "rag_tokens": rag_tokens,
                "rag_cost_usd": rag_cost,
                "judge_tokens": judge_tokens,
                "judge_cost_usd": judge_cost,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "judge_result": judge_result.model_dump(),
            }

            if crag_enable:
                is_web = output.get("web_search_needed", False)
                result["crag_telemetry"] = {
                    "web_search_triggered": is_web,
                    "docs_after_grading": 0 if is_web else len(retrieved_docs),
                    "grader_tokens": output.get("grader_tokens", 0) or 0,
                    "grader_cost_usd": output.get("grader_cost_usd", 0.0) or 0.0,
                }

            return result

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
                "rag_tokens": 0,
                "rag_cost_usd": 0.0,
                "judge_tokens": 0,
                "judge_cost_usd": 0.0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "judge_result": {
                    "faithfulness_reason": str(e),
                    "faithfulness_score": 0,
                    "relevance_reason": str(e),
                    "relevance_score": 0,
                },
                "crag_telemetry": None,
            }


async def evaluate_queries(
    judge_llm,
    ground_truth: list[dict],
    top_k: int = DEFAULT_RERANK_TOP_K,
    crag_enable: bool = True,
) -> list[dict]:
    """Evaluate a list of queries using the graph pipeline and judge LLM."""

    # Warm-up Cross-Encoder model before benchmark to eliminate race condition
    get_cross_encoder_model()

    app = graph_pipeline(llm=DEFAULT_LLM, top_k=top_k, crag_enable=crag_enable).compile()

    tasks = [
        evaluate_query_with_semaphore(app, judge_llm, item, top_k, crag_enable=crag_enable)
        for item in ground_truth
    ]
    results = await asyncio.gather(*tasks)
    return results


def save_judge_results(results: list[dict], run_id: str, metadata: RunMetadata) -> GenerationEvaluationReport:
    """Save judge evaluation results to a JSON file and log summary statistics with latencies and token costs."""
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
        total_rag_cost = sum(r.get("rag_cost_usd", 0.0) for r in valid_results)
        total_judge_cost = sum(r.get("judge_cost_usd", 0.0) for r in valid_results)
        total_cost = sum(r.get("cost_usd", 0.0) for r in valid_results)
        avg_rag_tokens = sum(r.get("rag_tokens", 0) for r in valid_results) / total_valid
        avg_tokens = sum(r.get("total_tokens", 0) for r in valid_results) / total_valid

        total_grader_cost = sum(
            r.get("crag_telemetry", {}).get("grader_cost_usd", 0.0)
            for r in valid_results
            if r.get("crag_telemetry")
        )
        avg_grader_tokens = (
            sum(
                r.get("crag_telemetry", {}).get("grader_tokens", 0)
                for r in valid_results
                if r.get("crag_telemetry")
            )
            / total_valid
            if total_valid > 0
            else 0.0
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
        logger.info(
            f"Production RAG Cost: ${total_rag_cost:.4f} USD | Avg RAG Tokens/Query: {avg_rag_tokens:.1f}"
        )
        if metadata.crag_enabled:
            logger.info(
                f"CRAG Grader Cost: ${total_grader_cost:.4f} USD | Avg Grader Tokens/Query: {avg_grader_tokens:.1f}"
            )
        logger.info(
            f"Judge Cost: ${total_judge_cost:.4f} USD | Total Run Cost: ${total_cost:.4f} USD"
        )
    else:
        avg_faithfulness = 0.0
        avg_relevance = 0.0
        avg_gen_ms = 0.0
        avg_judge_ms = 0.0
        avg_total_ms = 0.0
        p95_gen_ms = 0.0
        p95_total_ms = 0.0
        total_rag_cost = 0.0
        avg_rag_tokens = 0.0
        total_grader_cost = 0.0
        avg_grader_tokens = 0.0
        total_judge_cost = 0.0
        total_cost = 0.0
        avg_tokens = 0.0

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
        total_rag_cost_usd=round(total_rag_cost, 4),
        avg_rag_tokens_per_query=round(avg_rag_tokens, 1),
        total_grader_cost_usd=round(total_grader_cost, 4),
        avg_grader_tokens_per_query=round(avg_grader_tokens, 1),
        total_judge_cost_usd=round(total_judge_cost, 4),
        total_run_cost_usd=round(total_cost, 4),
        avg_tokens_per_query=round(avg_tokens, 1),
    )

    report = GenerationEvaluationReport(
        metadata=metadata,
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

    base_judge_llm = DEFAULT_LLM
    judge_llm = base_judge_llm.with_structured_output(JudgeEvaluationPrompt)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    crag_enable = True

    results = asyncio.run(
        evaluate_queries(judge_llm, ground_truth, top_k=DEFAULT_RERANK_TOP_K, crag_enable=crag_enable)
    )

    run_prefix = "crag" if crag_enable else "baseline"

    model_temp = getattr(DEFAULT_LLM, "temperature", 0.0)
    judge_temp = getattr(base_judge_llm, "temperature", 0.0)

    metadata = RunMetadata(
        run_name=f"{run_prefix}_run_{run_id}",
        timestamp=datetime.datetime.now().isoformat(),
        crag_enabled=crag_enable,
        model=getattr(DEFAULT_LLM, "model_name", "gpt-4o-mini"),
        temperature=model_temp if model_temp is not None else 0.0,
        judge_model=getattr(base_judge_llm, "model_name", "gpt-4o-mini"),
        judge_temperature=judge_temp if judge_temp is not None else 0.0,
        top_k=DEFAULT_RERANK_TOP_K,
    )

    save_judge_results(results, run_id, metadata=metadata)

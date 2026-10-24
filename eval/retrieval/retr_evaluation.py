import json
import logging
import asyncio

from src.config import (
    CHUNK_SIZE,
    CONCURRENT_SEMAPHORE,
    DEFAULT_LLM,
    DEFAULT_RERANK_TOP_K,
    MMR_FETCH_K,
    OVERLAP_SIZE,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVER_SEARCH_TYPE,
)
from src.graph import create_rewrite_query_node
from src.ingestion import ingestion_pipeline
from src.retrieval import get_cross_encoder_model, get_retriever
from src.utils import calculate_percentile
from eval.retrieval.experiments import (
    get_experiments,
    print_experiments_summary,
    save_evaluation_results,
)
from eval.retrieval.metrics import evaluate_item
from eval.retrieval.schemas import EvaluationConfig, EvaluationReport

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

rewrite_query_node = create_rewrite_query_node(DEFAULT_LLM)

# EVALUATION PIPELINE
async def evaluate_retrieval(
    filepath: str = "data/ground_truth.json",
    k: int = RETRIEVAL_CANDIDATE_K,
    fetch_k: int = MMR_FETCH_K,
    top_k: int = DEFAULT_RERANK_TOP_K,
    search_type: str = RETRIEVER_SEARCH_TYPE,
    use_reranker: bool = True,
) -> EvaluationReport:
    """
    Run full retrieval and reranking evaluation pipeline across all valid ground truth items.
    """
    # Warm-up Cross-Encoder model before benchmark to eliminate cold-start timing penalty

    semaphore = asyncio.Semaphore(CONCURRENT_SEMAPHORE)

    if use_reranker:
        get_cross_encoder_model()

    with open(filepath, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Filter out records without expected matches
    evaluation_questions = [
        item for item in ground_truth if item.get("expected_text_match") is not None
    ]

    total_hits = 0
    mrr_total = 0
    total_baseline_hits = 0
    total_baseline_mrr = 0
    total_multi_turn_hits = 0
    total_multi_turn_mrr = 0
    multi_turn_count = 0
    detailed_results = []
        
    # Execute evaluation logic for the item
    async def evaluate_single_item(item: dict) -> dict | None:
        """
        Evaluate a single retrieval item, returning detailed metrics.
        """
        source_doc = item.get("source_doc")
        file_name = source_doc
        selected_files = [s.strip() for s in source_doc.split(",")] if source_doc else None

        # Instantiate document-filtered retriever
        try:
            retriever = get_retriever(
                k=k,
                fetch_k=fetch_k,
                selected_files=selected_files,
                search_type=search_type,
            )

            if retriever is None:
                logger.error(f"Retriever could not be created for file: {file_name}")
                return None

            return await evaluate_item(
                item=item,
                retriever=retriever,
                rewrite_node=rewrite_query_node,
                semaphore=semaphore,
                use_reranker=use_reranker,
                top_k=top_k,
            )

        except Exception as e:
            logger.error(
                f"Error while creating retriever for file {file_name}: {e}"
            )
            return None


    tasks = [evaluate_single_item(item) for item in evaluation_questions]
    results = await asyncio.gather(*tasks)  
    detailed_results = [result for result in results if result is not None]

    for result in detailed_results:

        # Add overall performance metrics
        total_hits += result["hit_score"]
        mrr_total += result["mrr_score"]

        # Track multi-turn baseline metrics for analysis between baseline and history-aware retrieval
        if result["question_type"] == "multi_turn_followup":
            multi_turn_count += 1
            total_baseline_hits += result["baseline_hit_score"]
            total_baseline_mrr += result["baseline_mrr_score"]
            total_multi_turn_hits += result["hit_score"]
            total_multi_turn_mrr += result["mrr_score"]

    total_questions = len(evaluation_questions)

    # Compute latency statistics (converted to milliseconds)
    total_latencies = [
        r["total_time"] for r in detailed_results if r.get("total_time") is not None
    ]
    retriever_latencies = [
        r["retriever_time"]
        for r in detailed_results
        if r.get("retriever_time") is not None
    ]
    reranker_latencies = [
        r["reranker_time"]
        for r in detailed_results
        if r.get("reranker_time") is not None
    ]

    avg_total_ms = (
        (sum(total_latencies) / total_questions * 1000)
        if total_questions > 0
        else 0.0
    )
    avg_retriever_ms = (
        (sum(retriever_latencies) / total_questions * 1000)
        if total_questions > 0
        else 0.0
    )
    avg_reranker_ms = (
        (sum(reranker_latencies) / total_questions * 1000)
        if total_questions > 0
        else 0.0
    )
    p95_total_ms = calculate_percentile(total_latencies, 0.95) * 1000

    config = EvaluationConfig(
        k=k,
        fetch_k=fetch_k,
        top_k=top_k,
        search_type=search_type,
        use_reranker=use_reranker,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE,
    )

    return EvaluationReport(
        config=config,
        total_evaluation_questions=total_questions,
        multi_turn_questions=multi_turn_count,
        total_hits=total_hits,
        average_hit_rate=(
            total_hits / total_questions if total_questions > 0 else 0.0
        ),
        average_mrr=mrr_total / total_questions if total_questions > 0 else 0.0,
        multi_turn_baseline_hit_rate=(
            total_baseline_hits / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        multi_turn_baseline_mrr=(
            total_baseline_mrr / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        multi_turn_history_aware_hit_rate=(
            total_multi_turn_hits / multi_turn_count
            if multi_turn_count > 0
            else 0.0
        ),
        multi_turn_history_aware_mrr=(
            total_multi_turn_mrr / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        avg_retriever_latency_ms=round(avg_retriever_ms, 2),
        avg_reranker_latency_ms=round(avg_reranker_ms, 2),
        avg_total_latency_ms=round(avg_total_ms, 2),
        p95_total_latency_ms=round(p95_total_ms, 2),
        detailed_results=detailed_results,
    )


if __name__ == "__main__":
    # Ensure vector store contains ingested documents before evaluating
    ingestion_pipeline(pdf_path="data/arxiv1.pdf")
    ingestion_pipeline(pdf_path="data/arxiv2.pdf")
    ingestion_pipeline(pdf_path="data/arxiv3.pdf")

    # Run retrieval benchmark and save results
    experiments = get_experiments()
    summary_records = []

    for exp in experiments:
        logger.info(f"Running retrieval evaluation for experiment: {exp['name']}")
        results = asyncio.run(
            evaluate_retrieval(
                filepath="data/ground_truth.json",
                k=exp["k"],
                fetch_k=exp["fetch_k"],
                search_type=exp["search_type"],
                use_reranker=exp["use_reranker"],
                top_k=exp.get("top_k", 3),
            )
        )
        save_evaluation_results(
            results, output_path=f"eval/results/retrieval/{exp['name']}_results"
        )

        summary_records.append(
            {
                "name": exp["name"],
                "hit_rate": results.average_hit_rate,
                "mrr": results.average_mrr,
                "avg_latency_ms": results.avg_total_latency_ms,
                "p95_latency_ms": results.p95_total_latency_ms,
                "retriever_latency_ms": results.avg_retriever_latency_ms,
                "reranker_latency_ms": results.avg_reranker_latency_ms,
            }
        )

        logger.info(
            f"Completed evaluation for {exp['name']}. "
            f"Average Hit Rate: {results.average_hit_rate:.4f}, Average MRR: {results.average_mrr:.4f}"
        )
        print(
            f"Completed evaluation for {exp['name']}. "
            f"Average Hit Rate: {results.average_hit_rate:.4f}, Average MRR: {results.average_mrr:.4f}"
        )

    # Print clean comparative leaderboard at the end
    print_experiments_summary(summary_records)
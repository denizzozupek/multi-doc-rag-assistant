import datetime
import json
import logging
from pathlib import Path

from src.config import RETRIEVAL_CANDIDATE_K
from eval.retrieval.schemas import EvaluationReport

logger = logging.getLogger(__name__)


def save_evaluation_results(
    results: EvaluationReport,
    output_path: str = "eval/results/retrieval/evaluation_results",
):
    """Save evaluation metrics to a timestamped JSON file and print summary stats to console."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_path}_{run_id}.json"

    # Ensure parent directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results.model_dump(), f, indent=4, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_file}.")

    # Log queries that failed retrieval (hit_score == 0)
    missed = [r for r in results.detailed_results if r["hit_score"] == 0]
    if missed:
        print(f"\nMissed ({len(missed)}) Questions :")
        for m in missed:
            print(
                f"Query: {m['search_query']}, Expected: {m['expected_text_match']}, "
                f"Retrieved: {[doc['content'][:100] for doc in m['retrieved_docs']]}"
            )

    print(
        f"\nTotal Evaluation Questions: {results.total_evaluation_questions}\n"
        f"Total Hits: {results.total_hits}\n"
        f"Average Hit Rate: {results.average_hit_rate:.4f}\n"
        f"Average MRR: {results.average_mrr:.4f}\n"
        f"Multi-turn Baseline Hit Rate: {results.multi_turn_baseline_hit_rate:.4f}\n"
        f"Multi-turn Baseline MRR: {results.multi_turn_baseline_mrr:.4f}\n"
        f"Multi-turn History-Aware Hit Rate: {results.multi_turn_history_aware_hit_rate:.4f}\n"
        f"Multi-turn History-Aware MRR: {results.multi_turn_history_aware_mrr:.4f}\n"
        f"Average Latency: {results.avg_total_latency_ms:.2f} ms "
        f"(Retriever: {results.avg_retriever_latency_ms:.2f} ms, Reranker: {results.avg_reranker_latency_ms:.2f} ms)\n"
        f"P95 Latency: {results.p95_total_latency_ms:.2f} ms\n"
    )


def print_experiments_summary(summary_records: list[dict]):
    """
    Print a clean, structured table/summary of all completed retrieval experiments.
    Prevents getting lost in long log streams.
    """
    header = (
        f"\n{'='*110}\n"
        f"{'EXPERIMENT':<28} | {'HIT RATE':<10} | {'MRR':<8} | {'AVG LAT (ms)':<14} | {'P95 LAT (ms)':<14} | {'RETRIEVER':<12} | {'RERANKER':<10}\n"
        f"{'-'*110}"
    )
    rows = []
    for s in summary_records:
        row = (
            f"{s['name']:<28} | "
            f"{s['hit_rate']:<10.4f} | "
            f"{s['mrr']:<8.4f} | "
            f"{s['avg_latency_ms']:<14.2f} | "
            f"{s['p95_latency_ms']:<14.2f} | "
            f"{s['retriever_latency_ms']:<12.2f} | "
            f"{s['reranker_latency_ms']:<10.2f}"
        )
        rows.append(row)
    footer = f"{'='*110}\n"

    print(header)
    for r in rows:
        print(r)
    print(footer)


def get_experiments() -> list[dict]:
    experiments = [
        # 1. Baseline Dense
        {
            "name": "Exp_A_Dense_k3",
            "top_k": 3,
            "k": 3,
            "fetch_k": 3,
            "search_type": "similarity",
            "use_reranker": False,
        },
        {
            "name": "Exp_A_Dense_k5",
            "top_k": 5,
            "k": 5,
            "fetch_k": 5,
            "search_type": "similarity",
            "use_reranker": False,
        },
        # 2. Baseline BM25
        {
            "name": "Exp_B_BM25_k3",
            "top_k": 3,
            "k": 3,
            "fetch_k": 3,
            "search_type": "bm25",
            "use_reranker": False,
        },
        # 3. Hybrid (Without Reranker)
        {
            "name": "Exp_C_Hybrid_k5",
            "top_k": 5,
            "k": 5,
            "fetch_k": 20,
            "search_type": "hybrid",
            "use_reranker": False,
        },
        # 4. Hybrid + Reranker (Standard: Take 20 candidates -> Rerank all 20 -> Select Top 3)
        {
            "name": "Exp_D_Hybrid_Rerank_top3",
            "top_k": 3,
            "k": RETRIEVAL_CANDIDATE_K,
            "fetch_k": RETRIEVAL_CANDIDATE_K,
            "search_type": "hybrid",
            "use_reranker": True,
        },
        # 5. Hybrid + Reranker (Standard: Take 20 candidates -> Rerank all 20 -> Select Top 5)
        {
            "name": "Exp_E_Hybrid_Rerank_top5",
            "top_k": 5,
            "k": RETRIEVAL_CANDIDATE_K,
            "fetch_k": RETRIEVAL_CANDIDATE_K,
            "search_type": "hybrid",
            "use_reranker": True,
        },
    ]

    return experiments

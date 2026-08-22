import json
import logging
import datetime

from src.utils import format_history
from src.retriever import get_retriever
from src.ingestion import ingestion_pipeline
from src.config import CHUNK_SIZE, OVERLAP_SIZE
from src.chain import history_search_chain

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the history-aware query rewriting chain once globally to avoid recreation overhead
history_chain = history_search_chain()


# HELPER FUNCTIONS
def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison by lowercasing and stripping extra whitespace."""
    return " ".join(text.lower().split())


def resolve_query(item: dict, history_chain) -> str:
    """
    Determine the query string for retrieval.
    If multi-turn, rewrite the follow-up query using conversation history.
    Otherwise, return the raw query.
    """
    if item.get("question_type") == "multi_turn_followup":
        # Extract raw history from dataset and convert to message objects
        raw_history = item.get("conversation_history", [])
        formatted_history = format_history(raw_history)
        
        # Invoke chain with 'chat_history' to match MessagesPlaceholder schema
        return history_chain.invoke(
            {"input": item["query"], "chat_history": formatted_history}
        )

    return item["query"]


# EVALUATION LOGIC (Hit Rate and MRR Metrics)
def evaluate_retrieval_single(retriever, query: str, expected_text_match: str):
    """
    Execute retrieval for a single query and compute Hit Rate and MRR metrics.
    """
    hit_score = 0
    mrr_score = 0

    try:
        retrieved_docs = retriever.invoke(query) or []
        if not retrieved_docs:
            logger.warning(f"No documents retrieved for query: '{query}'")
            return hit_score, mrr_score, []

        normalized_expected_text = normalize_text(expected_text_match)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if normalized_expected_text in normalize_text(doc.page_content):
                hit_score = 1
                mrr_score = 1 / rank
                break

        return (
            hit_score,
            mrr_score,
            [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in retrieved_docs
            ],
        )

    except Exception as e:
        logger.error(f"Error during retrieval for query '{query}': {e}", exc_info=True)
        return hit_score, mrr_score, []

# EVALUATE MULTI-TURN AND SINGLE-TURN QUERIES
def evaluate_item(item: dict, retriever, history_chain) -> dict:
    question_type = item.get("question_type")
    raw_query = item["query"]
    expected_text_match = item.get("expected_text_match")

    baseline_hit = None
    baseline_mrr = None

    # Step 1: Handle baseline evaluation for multi-turn questions
    if question_type == "multi_turn_followup":
        baseline_hit, baseline_mrr, _ = evaluate_retrieval_single(
            retriever, raw_query, expected_text_match
        )
        search_query = resolve_query(item, history_chain)
    else:
        search_query = raw_query

    # Step 2: Perform main retrieval using the resolved query
    hit_score, mrr_score, retrieved_docs = evaluate_retrieval_single(
        retriever, search_query, expected_text_match
    )

    return {
        "id": item.get("id"),
        "question_type": item.get("question_type"),
        "raw_query": item["query"],
        "search_query": search_query,
        "expected_text_match": expected_text_match,
        "retrieved_docs": retrieved_docs,
        "hit_score": hit_score,
        "mrr_score": mrr_score,
        "baseline_hit_score": baseline_hit,
        "baseline_mrr_score": baseline_mrr,
    }


# EVALUATION PIPELINE
def evaluate_retrieval(
    filepath: str = "data/ground_truth.json",
    k: int = 5,
    search_type: str = "similarity",
):
    """
    Run full retrieval evaluation pipeline across all valid ground truth items.
    """
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

    # Iterate over each evaluation item
    for item in evaluation_questions:
        file_name = item["source_doc"]

        # Instantiate document-filtered retriever
        try:
            retriever = get_retriever(
                k=k, selected_files=[file_name], search_type=search_type
            )
        except Exception as e:
            logger.error(f"Error while creating retriever for file {file_name}: {e}")
            continue
        if retriever is None:
            logger.error(f"Retriever could not be created for file: {file_name}")
            continue

        # Execute evaluation logic for the item
        result = evaluate_item(item, retriever, history_chain)
        detailed_results.append(result)

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

    return {
        "total_evaluation_questions": total_questions,
        "multi_turn_questions": multi_turn_count,
        "total_hits": total_hits,
        "average_hit_rate": total_hits / total_questions if total_questions > 0 else 0,
        "average_mrr": mrr_total / total_questions if total_questions > 0 else 0,
        "multi_turn_baseline_hit_rate": (
            total_baseline_hits / multi_turn_count if multi_turn_count > 0 else 0
        ),
        "multi_turn_baseline_mrr": (
            total_baseline_mrr / multi_turn_count if multi_turn_count > 0 else 0
        ),
        "multi_turn_history_aware_hit_rate": (
            total_multi_turn_hits / multi_turn_count if multi_turn_count > 0 else 0
        ),
        "multi_turn_history_aware_mrr": (
            total_multi_turn_mrr / multi_turn_count if multi_turn_count > 0 else 0
        ),
        "detailed_results": detailed_results,
        "chunk_size": CHUNK_SIZE,
        "overlap_size": OVERLAP_SIZE,
    }

# SAVE EVALUATION RESULTS
def save_evaluation_results(results: dict, output_path: str = "eval/evaluation_results/evaluation_results"):
    """Save evaluation metrics to a timestamped JSON file and print summary stats to console."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_path}_{run_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_file}.")

    # Log queries that failed retrieval (hit_score == 0)
    missed = [r for r in results["detailed_results"] if r["hit_score"] == 0]
    if missed:
        print(f"\nMissed ({len(missed)}) Questions :")
        for m in missed:
            print(
                f"Query: {m['search_query']}, Expected: {m['expected_text_match']}, Retrieved: {[doc['content'][:100] for doc in m['retrieved_docs']]}"
            )

    print(
        f"\nTotal Evaluation Questions: {results['total_evaluation_questions']}\n"
        f"Total Hits: {results['total_hits']}\n"
        f"Average Hit Rate: {results['average_hit_rate']:.4f}\n"
        f"Average MRR: {results['average_mrr']:.4f}\n"
        f"Multi-turn Baseline Hit Rate: {results['multi_turn_baseline_hit_rate']:.4f}\n"
        f"Multi-turn Baseline MRR: {results['multi_turn_baseline_mrr']:.4f}\n"
        f"Multi-turn History-Aware Hit Rate: {results['multi_turn_history_aware_hit_rate']:.4f}\n"
        f"Multi-turn History-Aware MRR: {results['multi_turn_history_aware_mrr']:.4f}\n"
    )

if __name__ == "__main__":
    # Ensure vector store contains ingested documents before evaluating
    ingestion_pipeline(pdf_path="data/arxiv1.pdf")
    ingestion_pipeline(pdf_path="data/arxiv2.pdf")

    # Run retrieval benchmark and save results
    results = evaluate_retrieval(
        filepath="data/ground_truth.json", k=5, search_type="similarity"
    )
    save_evaluation_results(results, output_path="eval/evaluation_results/evaluation_results")
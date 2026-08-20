import json
import logging
import datetime
from src.retriever import get_retriever
from src.ingestion import ingestion_pipeline
from src.config import CHUNK_SIZE, OVERLAP_SIZE

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def evaluate_retrieval_single(retriever, query: str, expected_text_match: str):
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


def evaluate_retrieval(
    filepath: str = "data/ground_truth.json",
    k: int = 5,
    search_type: str = "similarity"
):
    with open(filepath, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    mrr_total = 0
    total_hits = 0
    evaluation_questions = [
        item for item in ground_truth if item.get("expected_text_match") is not None
    ]
    total_evaluation_questions = len(evaluation_questions)
    detailed_results = []

    for item in evaluation_questions:
        query = item["query"]
        expected_text_match = item["expected_text_match"]
        file_name = item["source_doc"]
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

        hit_score, mrr_score, retrieved_docs = evaluate_retrieval_single(
            retriever, query, expected_text_match
        )

        total_hits += hit_score
        mrr_total += mrr_score

        detailed_results.append(
            {
                "query": query,
                "expected_text_match": expected_text_match,
                "retrieved_docs": retrieved_docs,
                "hit_score": hit_score,
                "mrr_score": mrr_score,
            })

    average_hit_rate = total_hits / total_evaluation_questions if total_evaluation_questions > 0 else 0
    average_mrr = mrr_total / total_evaluation_questions if total_evaluation_questions > 0 else 0

    results = {
        "total_evaluation_questions": total_evaluation_questions,
        "total_hits": total_hits,
        "average_hit_rate": average_hit_rate,
        "average_mrr": average_mrr,
        "detailed_results": detailed_results,
        "chunk_size": CHUNK_SIZE,
        "overlap_size": OVERLAP_SIZE,
    }

    return results

def save_evaluation_results(results: dict, output_path: str = "evaluation_results"):

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{output_path}_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_path}_{run_id}.json.")

    missed = [r for r in results["detailed_results"] if r["hit_score"] == 0]
    if missed:
        print(f"\nMissed ({len(missed)}) Questions :")
        for m in missed:
            print(f"Query: {m['query']}, Expected: {m['expected_text_match']}, Retrieved: {[doc['content'][:100] for doc in m['retrieved_docs']]}")

    print(f"\nTotal Evaluation Questions: {results['total_evaluation_questions']}\n Total Hits: {results['total_hits']}\n Average Hit Rate: {results['average_hit_rate']:.4f}\n Average MRR: {results['average_mrr']:.4f}")



if __name__ == "__main__":
    # Ensure the ingestion pipeline is run before evaluation
    ingestion_pipeline(pdf_path="data/arxiv1.pdf")
    ingestion_pipeline(pdf_path="data/arxiv2.pdf")

    results = evaluate_retrieval(
        filepath="data/ground_truth.json", k=8, search_type="similarity"
    )
    save_evaluation_results(results, output_path="evaluation_results")
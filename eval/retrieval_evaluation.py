import datetime
import json
import logging

from src.config import CHUNK_SIZE, OVERLAP_SIZE
from src.generation.chain import get_cross_encoder_model, history_search_chain
from src.ingestion.ingestion import ingestion_pipeline
from src.retrieval.reranker import reranker
from src.retrieval.retriever import get_retriever
from src.utils import format_history
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the history-aware query rewriting chain once globally to avoid recreation overhead
history_chain = history_search_chain()


class RetrievalResult(BaseModel):
    hit_score: int
    mrr_score: float
    retrieved_docs: list[dict]


class EvaluationConfig(BaseModel):
    k: int
    fetch_k: int
    top_k: int
    search_type: str
    use_reranker: bool
    chunk_size: int
    overlap_size: int


class EvaluationReport(BaseModel):
    config: EvaluationConfig
    total_evaluation_questions: int
    multi_turn_questions: int
    total_hits: int
    average_hit_rate: float
    average_mrr: float
    multi_turn_baseline_hit_rate: float
    multi_turn_baseline_mrr: float
    multi_turn_history_aware_hit_rate: float
    multi_turn_history_aware_mrr: float
    detailed_results: list[dict]


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
def evaluate_retrieval_single(
    retriever, query: str, expected_text_match: str, use_reranker: bool = True, top_k: int = 3
) -> RetrievalResult:
    """
    Execute retrieval and reranking for a single query,
    then compute Hit Rate and MRR metrics.
    """
    hit_score = 0
    mrr_score = 0

    try:
        retrieved_docs = retriever.invoke(query) or []
        if use_reranker:
            reranked_docs = reranker(
                query=query,
                documents=retrieved_docs,
                model=get_cross_encoder_model(),
                top_k=top_k,
            )
        else:
            reranked_docs = retrieved_docs

        if not reranked_docs:
            logger.warning(f"No documents retrieved or reranked for query: '{query}'")
            return RetrievalResult(
                hit_score=hit_score, mrr_score=mrr_score, retrieved_docs=[]
            )

        normalized_expected_text = normalize_text(expected_text_match)
        for rank, doc in enumerate(reranked_docs, start=1):
            if normalized_expected_text in normalize_text(doc.page_content):
                hit_score = 1
                mrr_score = 1 / rank
                break

        return RetrievalResult(
            hit_score=hit_score,
            mrr_score=mrr_score,
            retrieved_docs=[
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in reranked_docs
            ],
        )

    except Exception as e:
        logger.error(
            f"Error during retrieval/reranking for query '{query}': {e}", exc_info=True
        )
        return RetrievalResult(hit_score=0, mrr_score=0, retrieved_docs=[])


# EVALUATE MULTI-TURN AND SINGLE-TURN QUERIES
def evaluate_item(
    item: dict, retriever, history_chain, use_reranker: bool = True, top_k: int = 3
) -> dict:
    question_type = item.get("question_type")
    raw_query = item["query"]
    expected_text_match = item.get("expected_text_match")

    baseline_hit = None
    baseline_mrr = None

    # Step 1: Handle baseline evaluation for multi-turn questions
    if question_type == "multi_turn_followup":
        baseline_res = evaluate_retrieval_single(
            retriever, raw_query, expected_text_match, use_reranker=use_reranker, top_k=top_k
        )
        baseline_hit = baseline_res.hit_score
        baseline_mrr = baseline_res.mrr_score
        search_query = resolve_query(item, history_chain)
    else:
        search_query = raw_query

    # Step 2: Perform main retrieval using the resolved query
    main_res = evaluate_retrieval_single(
        retriever, search_query, expected_text_match, use_reranker=use_reranker, top_k=top_k
    )

    return {
        "id": item.get("id"),
        "question_type": item.get("question_type"),
        "raw_query": item["query"],
        "search_query": search_query,
        "expected_text_match": expected_text_match,
        "retrieved_docs": main_res.retrieved_docs,
        "hit_score": main_res.hit_score,
        "mrr_score": main_res.mrr_score,
        "baseline_hit_score": baseline_hit,
        "baseline_mrr_score": baseline_mrr,
        "use_reranker": use_reranker,
        "top_k": top_k,
    }


# EVALUATION PIPELINE
def evaluate_retrieval(
    filepath: str = "data/ground_truth.json",
    k: int = 15,
    fetch_k: int = 20,
    top_k: int = 3,
    search_type: str = "hybrid",
    use_reranker: bool = True,
):
    """
    Run full retrieval and reranking evaluation pipeline across all valid ground truth items.
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
                k=k,
                fetch_k=fetch_k,
                selected_files=[file_name],
                search_type=search_type,
            )
        except Exception as e:
            logger.error(f"Error while creating retriever for file {file_name}: {e}")
            continue
        if retriever is None:
            logger.error(f"Retriever could not be created for file: {file_name}")
            continue

        # Execute evaluation logic for the item
        result = evaluate_item(
            item, retriever, history_chain, use_reranker=use_reranker, top_k=top_k
        )
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
        average_hit_rate=total_hits / total_questions if total_questions > 0 else 0.0,
        average_mrr=mrr_total / total_questions if total_questions > 0 else 0.0,
        multi_turn_baseline_hit_rate=(
            total_baseline_hits / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        multi_turn_baseline_mrr=(
            total_baseline_mrr / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        multi_turn_history_aware_hit_rate=(
            total_multi_turn_hits / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        multi_turn_history_aware_mrr=(
            total_multi_turn_mrr / multi_turn_count if multi_turn_count > 0 else 0.0
        ),
        detailed_results=detailed_results,
    )


# SAVE EVALUATION RESULTS
def save_evaluation_results(
    results: EvaluationReport,
    output_path: str = "eval/evaluation_results/evaluation_results",
):
    """Save evaluation metrics to a timestamped JSON file and print summary stats to console."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_path}_{run_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results.model_dump(), f, indent=4, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_file}.")

    # Log queries that failed retrieval (hit_score == 0)
    missed = [r for r in results.detailed_results if r["hit_score"] == 0]
    if missed:
        print(f"\nMissed ({len(missed)}) Questions :")
        for m in missed:
            print(
                f"Query: {m['search_query']}, Expected: {m['expected_text_match']}, Retrieved: {[doc['content'][:100] for doc in m['retrieved_docs']]}"
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
    )


if __name__ == "__main__":
    # Ensure vector store contains ingested documents before evaluating
    ingestion_pipeline(pdf_path="data/arxiv1.pdf")
    ingestion_pipeline(pdf_path="data/arxiv2.pdf")

    # Run retrieval benchmark and save results
    experiments = [
    # 1. Baseline Dense 
    {"name": "Exp_A_Dense_k3","top_k": 3, "k": 3, "fetch_k": 3, "search_type": "similarity", "use_reranker": False},
    {"name": "Exp_A_Dense_k5", "top_k": 5, "k": 5, "fetch_k": 5, "search_type": "similarity", "use_reranker": False},
    
    # 2. Baseline BM25
    {"name": "Exp_B_BM25_k3", "top_k": 3, "k": 3, "fetch_k": 3, "search_type": "bm25", "use_reranker": False},
    
    # 3. Hybrid (Without Reranker)
    {"name": "Exp_C_Hybrid_k5", "top_k": 5, "k": 5, "fetch_k": 20, "search_type": "hybrid", "use_reranker": False},
    
    # 4. Hybrid + Reranker (Maximum Hit: Take 20 candidates -> Give 15 to Reranker -> Take Top 3)
    {"name": "Exp_D_Hybrid_Rerank_top3","top_k": 3, "k": 15, "fetch_k": 20, "search_type": "hybrid", "use_reranker": True}, # top_k=3
    
    # 5. Hybrid + Reranker (Maximum Hit: Take 20 candidates -> Give 15 to Reranker -> Take Top 5)
    {"name": "Exp_E_Hybrid_Rerank_top5", "top_k": 5, "k": 15, "fetch_k": 20, "search_type": "hybrid", "use_reranker": True}, # top_k=5
]
    for exp in experiments:
        logger.info(f"Running retrieval evaluation for experiment: {exp['name']}")
        results = evaluate_retrieval(
            filepath="data/ground_truth.json",
            k=exp["k"],
            fetch_k=exp["fetch_k"],
            search_type=exp["search_type"],
            use_reranker=exp["use_reranker"],
            top_k=exp.get("top_k", 3),
        )
        save_evaluation_results(
            results, output_path=f"eval/evaluation_results/{exp['name']}_results"
        )

        logger.info(
            f"Completed evaluation for {exp['name']}. Average Hit Rate: {results.average_hit_rate:.4f}, Average MRR: {results.average_mrr:.4f}"
        )
        print(
            f"Completed evaluation for {exp['name']}. Average Hit Rate: {results.average_hit_rate:.4f}, Average MRR: {results.average_mrr:.4f}"
        )
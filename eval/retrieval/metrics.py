import logging
import asyncio
from typing import Callable
from time import perf_counter

from langchain_core.messages import HumanMessage

from src.config import CONCURRENT_SEMAPHORE, DEFAULT_RERANK_TOP_K
from src.retrieval import get_cross_encoder_model, reranker
from src.utils import format_history
from eval.retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(CONCURRENT_SEMAPHORE)

# HELPER FUNCTIONS
def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison by lowercasing and stripping extra whitespace."""
    return " ".join(text.lower().split())


async def resolve_query(item: dict, rewrite_node: Callable) -> str:
    """
    Determine the query string for retrieval.
    If multi-turn, rewrite the follow-up query using rewrite_query_node.
    Otherwise, return the raw query.
    """
    if item.get("question_type") == "multi_turn_followup":
        raw_history = item.get("conversation_history", [])
        formatted_history = format_history(raw_history)

        query = item["query"]
        state = {"messages": formatted_history + [HumanMessage(content=query)]}
        result = await asyncio.to_thread(rewrite_node, state)

        return result.get("rewrite_query", query)

    return item["query"]


# EVALUATION LOGIC (Hit Rate and MRR Metrics)
async def evaluate_retrieval_single_with_semaphore(
    retriever,
    query: str,
    expected_text_match: str,
    use_reranker: bool = True,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> RetrievalResult:
    """
    Execute retrieval and reranking for a single query,
    then compute Hit Rate, MRR, and Latency metrics.
    """
    async with semaphore:
        start_time = perf_counter()
        
        hit_score = 0
        mrr_score = 0
        retriever_time = 0.0
        reranker_time = 0.0


        try:
            retriever_start = perf_counter()
            retrieved_docs = await retriever.ainvoke(query) or [] #We call the Langchain BaseRetriever's async invoke method to get the retrieved documents for the query
            retriever_time = perf_counter() - retriever_start

            if use_reranker and retrieved_docs:
                rerank_start = perf_counter()
                reranked_docs = await asyncio.to_thread(
                    reranker,
                    query=query,
                    documents=retrieved_docs,
                    model=get_cross_encoder_model(),
                    top_k=top_k,
                )
                reranker_time = perf_counter() - rerank_start
            else:
                reranked_docs = retrieved_docs

            total_time = perf_counter() - start_time

            if not reranked_docs:
                logger.warning(f"No documents retrieved or reranked for query: '{query}'")
                return RetrievalResult(
                    hit_score=hit_score,
                    mrr_score=mrr_score,
                    retrieved_docs=[],
                    retriever_time=retriever_time,
                    reranker_time=reranker_time,
                    total_time=total_time,
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
                retriever_time=retriever_time,
                reranker_time=reranker_time,
                total_time=total_time,
            )

        except Exception as e:
            logger.error(
                f"Error during retrieval/reranking for query '{query}': {e}", exc_info=True
            )
            return RetrievalResult(
                hit_score=0,
                mrr_score=0,
                retrieved_docs=[],
                retriever_time=retriever_time,
                reranker_time=reranker_time,
                total_time=perf_counter() - start_time,
            )


# EVALUATE MULTI-TURN AND SINGLE-TURN QUERIES
async def evaluate_item(
    item: dict,
    retriever,
    rewrite_node: Callable,
    use_reranker: bool = True,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> dict:
    question_type = item.get("question_type")
    raw_query = item["query"]
    expected_text_match = item.get("expected_text_match")

    baseline_hit = None
    baseline_mrr = None
    rewrite_time = 0.0

    # Step 1: Handle baseline evaluation for multi-turn questions
    if question_type == "multi_turn_followup":
        baseline_res = await evaluate_retrieval_single_with_semaphore(
            retriever,
            raw_query,
            expected_text_match,
            use_reranker=use_reranker,
            top_k=top_k,
        )
        baseline_hit = baseline_res.hit_score
        baseline_mrr = baseline_res.mrr_score

        # Measure query rewrite latency
        rewrite_start = perf_counter()
        search_query = await resolve_query(item, rewrite_node)
        rewrite_time = perf_counter() - rewrite_start
    else:
        search_query = raw_query

    # Step 2: Perform main retrieval using the resolved query
    main_res = await evaluate_retrieval_single_with_semaphore(
        retriever,
        search_query,
        expected_text_match,
        use_reranker=use_reranker,
        top_k=top_k,
    )

    retriever_time = main_res.retriever_time or 0.0
    reranker_time = main_res.reranker_time or 0.0
    total_time = (main_res.total_time or 0.0) + rewrite_time

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
        "rewrite_time": rewrite_time,
        "retriever_time": retriever_time,
        "reranker_time": reranker_time,
        "total_time": total_time,
    }

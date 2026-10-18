from pydantic import BaseModel


class RetrievalResult(BaseModel):
    hit_score: int
    mrr_score: float
    retrieved_docs: list[dict]
    retriever_time: float | None = None  # Optional field to store retrieval time
    reranker_time: float | None = None  # Optional field to store reranker time
    total_time: float | None = None  # Optional field to store total time


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
    avg_retriever_latency_ms: float = 0.0
    avg_reranker_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    p95_total_latency_ms: float = 0.0
    detailed_results: list[dict]

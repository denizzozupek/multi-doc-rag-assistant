from pydantic import BaseModel


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

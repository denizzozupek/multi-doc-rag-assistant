from pydantic import BaseModel, Field


class JudgeEvaluationPrompt(BaseModel):
    faithfulness_reason: str
    faithfulness_score: int = Field(..., ge=1, le=5)

    relevance_reason: str
    relevance_score: int = Field(..., ge=1, le=5)


class GenerationEvaluationItem(BaseModel):
    id: str | None = None
    query: str
    source_doc: str | None = None
    question_type: str | None = None
    top_k: int
    context: str | None = None
    answer: str | None = None
    generation_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    judge_result: JudgeEvaluationPrompt


class GenerationSummary(BaseModel):
    avg_faithfulness: float
    avg_relevance: float
    avg_multiturn_faithfulness: float | None = None
    avg_multiturn_relevance: float | None = None
    avg_generation_latency_ms: float
    avg_judge_latency_ms: float
    avg_total_latency_ms: float
    p95_generation_latency_ms: float
    p95_total_latency_ms: float


class GenerationEvaluationReport(BaseModel):
    total_questions: int
    total_valid: int
    total_multiturn: int
    summary: GenerationSummary
    detailed_results: list[GenerationEvaluationItem]

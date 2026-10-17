from pydantic import BaseModel, Field


class JudgeEvaluationPrompt(BaseModel):
    faithfulness_reason: str
    faithfulness_score: int = Field(..., ge=1, le=5)

    relevance_reason: str
    relevance_score: int = Field(..., ge=1, le=5)

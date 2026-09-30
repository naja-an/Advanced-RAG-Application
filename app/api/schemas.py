"""HTTP request/response schemas (the public API contract)."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain import DocumentStatus


class _FromDomain(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SessionResponse(_FromDomain):
    session_id: UUID
    created_at: datetime
    expires_at: datetime | None = None


class DocumentResponse(_FromDomain):
    document_id: UUID
    name: str
    source: str
    status: DocumentStatus
    error: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)


class SourceResponse(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str


class EvaluationResponse(BaseModel):
    answer_relevance: float | None = None
    faithfulness: float | None = None
    latency_ms: float
    error: str | None = None


class EvaluationTaskResponse(BaseModel):
    status: Literal["pending", "completed", "failed"]
    evaluation: EvaluationResponse | None = None


class ChatResponse(BaseModel):
    question: str
    query_used_for_retrieval: str
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    evaluation: EvaluationResponse
    evaluation_id: UUID


class ChatTurnResponse(_FromDomain):
    turn_id: int
    question: str
    query_used_for_retrieval: str
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str

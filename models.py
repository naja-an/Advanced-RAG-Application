"""Pydantic models for the RAG chatbot."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field


class DocumentStatus(str, Enum):
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"


class SessionResponse(BaseModel):
	session_id: UUID
	created_at: datetime
	expires_at: datetime | None = None


class DocumentResponse(BaseModel):
	document_id: UUID
	name: str
	source: str
	status: DocumentStatus
	error: str | None = None


class DocumentIngestionRequest(BaseModel):
	urls: list[AnyHttpUrl] = Field(default_factory=list, max_length=20)


class UploadedFileMetadata(BaseModel):
	"""Validated metadata for a multipart-uploaded document."""

	filename: str = Field(min_length=1, max_length=255)
	content_type: str | None = None
	size_bytes: int = Field(ge=0)


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


class ChatTurnResponse(BaseModel):
	turn_id: int
	question: str
	query_used_for_retrieval: str
	answer: str
	sources: list[SourceResponse] = Field(default_factory=list)
	created_at: datetime


class MessageResponse(BaseModel):
	role: str
	content: str
	created_at: datetime


class ErrorResponse(BaseModel):
	detail: str

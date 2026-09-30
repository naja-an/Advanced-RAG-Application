"""Plain domain types shared across layers (no framework dependencies except `Document`)."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from langchain_core.documents import Document


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SessionRecord:
    session_id: UUID
    created_at: datetime
    expires_at: datetime | None = None


@dataclass(frozen=True)
class DocumentRecord:
    document_id: UUID
    name: str
    source: str
    status: DocumentStatus
    error: str | None = None


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content: bytes


@dataclass
class IngestedSource:
    """Result of parsing one uploaded file or URL."""

    document_id: UUID
    name: str
    source: str
    chunks: list[Document] = field(default_factory=list)
    error: str | None = None

    def to_record(self) -> DocumentRecord:
        status = DocumentStatus.FAILED if self.error else DocumentStatus.COMPLETED
        return DocumentRecord(self.document_id, self.name, self.source, status, self.error)


@dataclass(frozen=True)
class ChatTurn:
    turn_id: int | None
    question: str
    query_used_for_retrieval: str
    answer: str
    sources: list[dict[str, Any]]
    created_at: datetime


@dataclass(frozen=True)
class RAGResult:
    answer: str
    documents: list[Document]
    query_used_for_retrieval: str


@dataclass(frozen=True)
class ChatOutcome:
    result: RAGResult
    evaluation_id: UUID
    latency_ms: float


@dataclass(frozen=True)
class EvaluationScores:
    answer_relevance: float | None = None
    faithfulness: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: UUID
    session_id: UUID
    status: EvaluationStatus
    latency_ms: float
    answer_relevance: float | None = None
    faithfulness: float | None = None
    error: str | None = None

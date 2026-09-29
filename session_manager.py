"""Session management for the RAG chatbot."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from document_processor import create_session_storage, remove_session_storage
from models import DocumentResponse, SessionResponse


@dataclass
class SessionState:
	session_id: UUID
	created_at: datetime
	expires_at: datetime | None = None
	documents: dict[UUID, DocumentResponse] = field(default_factory=dict)
	loaded_documents: list[object] = field(default_factory=list)
	storage_dir: Path = field(default_factory=create_session_storage)
	pipeline: object | None = None


class SessionNotFoundError(KeyError):
	"""Raised when a requested session does not exist."""


class SessionManager:
	"""Thread-safe in-memory session registry for the initial API implementation."""

	def __init__(self) -> None:
		self._sessions: dict[UUID, SessionState] = {}
		self._lock = RLock()

	def create(self) -> SessionResponse:
		now = datetime.now(timezone.utc)
		state = SessionState(session_id=uuid4(), created_at=now)
		with self._lock:
			self._sessions[state.session_id] = state
		return SessionResponse(
			session_id=state.session_id,
			created_at=state.created_at,
			expires_at=state.expires_at,
		)

	def get(self, session_id: UUID) -> SessionState:
		with self._lock:
			state = self._sessions.get(session_id)
		if state is None:
			raise SessionNotFoundError(session_id)
		return state

	def list_documents(self, session_id: UUID) -> list[DocumentResponse]:
		return list(self.get(session_id).documents.values())

	def add_documents(
		self,
		session_id: UUID,
		documents: list[DocumentResponse],
		loaded_documents: list[object],
	) -> None:
		state = self.get(session_id)
		with self._lock:
			for document in documents:
				state.documents[document.document_id] = document
			state.loaded_documents.extend(loaded_documents)

	def set_pipeline(self, session_id: UUID, pipeline: object) -> None:
		with self._lock:
			self.get(session_id).pipeline = pipeline

	def delete(self, session_id: UUID) -> None:
		with self._lock:
			state = self._sessions.pop(session_id, None)
			if state is None:
				raise SessionNotFoundError(session_id)
		remove_session_storage(state.storage_dir)

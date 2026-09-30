import asyncio
import logging
import sqlite3
from time import perf_counter
from uuid import UUID

from app.domain import ChatOutcome, ChatTurn, RAGResult
from app.evaluation.service import EvaluationService
from app.exceptions import ChatTimeoutError, NoDocumentsError
from app.observability import Tracing
from app.persistence.turns import TurnRepository
from app.rag.index import IndexRegistry
from app.rag.pipeline import RAGPipeline
from app.services.sessions import SessionService
from app.utils.locks import KeyedLocks

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        sessions: SessionService,
        indexes: IndexRegistry,
        pipeline: RAGPipeline,
        turns: TurnRepository,
        evaluations: EvaluationService,
        tracing: Tracing,
        *,
        history_max_turns: int,
        timeout_s: float,
    ):
        self._sessions = sessions
        self._indexes = indexes
        self._pipeline = pipeline
        self._turns = turns
        self._evaluations = evaluations
        self._tracing = tracing
        self._history_max_turns = history_max_turns
        self._timeout_s = timeout_s
        self._locks = KeyedLocks()  # one chat turn at a time per session

    def forget(self, session_id: UUID) -> None:
        self._locks.discard(session_id)

    async def history(self, session_id: UUID) -> list[ChatTurn]:
        await self._sessions.require(session_id)
        return await self._turns.list_all(session_id)

    async def chat(self, session_id: UUID, question: str) -> ChatOutcome:
        started_at = perf_counter()
        await self._sessions.require(session_id)
        index = await self._indexes.get(session_id)
        if index is None or index.chunk_count == 0:
            raise NoDocumentsError()

        config = {
            "callbacks": self._tracing.callbacks(),
            "metadata": {"langfuse_session_id": str(session_id)},
        }
        async with self._locks.get(session_id):
            history = await self._turns.recent(session_id, self._history_max_turns)
            try:
                result = await asyncio.wait_for(
                    self._pipeline.answer(question, history, index, config=config),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise ChatTimeoutError() from exc
            await self._save_turn(session_id, question, result)

        latency_ms = round((perf_counter() - started_at) * 1000, 1)
        evaluation_id = await self._evaluations.submit(
            session_id, question, result.answer, [d.page_content for d in result.documents], latency_ms
        )
        logger.info(
            "chat_completed session_id=%s source_count=%d duration_ms=%d",
            session_id,
            len(result.documents),
            int(latency_ms),
        )
        return ChatOutcome(result, evaluation_id, latency_ms)

    async def _save_turn(self, session_id: UUID, question: str, result: RAGResult) -> None:
        """Persisting history is best-effort: the user already has a good answer."""
        sources = [{"metadata": d.metadata, "content": d.page_content} for d in result.documents]
        try:
            await self._turns.add(
                session_id, question, result.query_used_for_retrieval, result.answer, sources
            )
        except sqlite3.Error:
            logger.exception("turn_persist_failed session_id=%s", session_id)

"""Background answer evaluation with bounded concurrency and orderly shutdown."""

import asyncio
import logging
import sqlite3
from uuid import UUID, uuid4

from app.domain import EvaluationRecord
from app.evaluation.metrics import AnswerEvaluator
from app.exceptions import EvaluationNotFoundError
from app.persistence.evaluations import EvaluationRepository

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(
        self,
        repository: EvaluationRepository,
        evaluator: AnswerEvaluator,
        *,
        concurrency: int = 2,
        shutdown_grace_s: float = 10.0,
    ):
        self._repo = repository
        self._evaluator = evaluator
        self._semaphore = asyncio.Semaphore(concurrency)
        self._grace_s = shutdown_grace_s
        self._tasks: set[asyncio.Task] = set()

    async def submit(
        self, session_id: UUID, question: str, answer: str, contexts: list[str], latency_ms: float
    ) -> UUID:
        """Record a pending evaluation and score it in the background. Never raises for storage
        problems: a failed bookkeeping write must not cost the user their answer."""
        evaluation_id = uuid4()
        try:
            await self._repo.create_pending(evaluation_id, session_id, latency_ms)
        except sqlite3.Error:
            logger.exception("evaluation_record_failed session_id=%s", session_id)
            return evaluation_id
        task = asyncio.create_task(self._run(evaluation_id, question, answer, contexts))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return evaluation_id

    async def _run(self, evaluation_id: UUID, question: str, answer: str, contexts: list[str]) -> None:
        async with self._semaphore:
            try:
                scores = await asyncio.to_thread(self._evaluator.evaluate, question, answer, contexts)
                await self._repo.complete(evaluation_id, scores)
            except asyncio.CancelledError:
                raise
            except sqlite3.Error:
                logger.exception("evaluation_store_failed evaluation_id=%s", evaluation_id)
            except Exception as exc:  # evaluator failures are data, not crashes
                logger.exception("evaluation_failed evaluation_id=%s", evaluation_id)
                try:
                    await self._repo.fail(evaluation_id, f"Evaluation unavailable: {type(exc).__name__}")
                except sqlite3.Error:
                    logger.exception("evaluation_store_failed evaluation_id=%s", evaluation_id)

    async def get(self, session_id: UUID, evaluation_id: UUID) -> EvaluationRecord:
        record = await self._repo.get(session_id, evaluation_id)
        if record is None:
            raise EvaluationNotFoundError(evaluation_id)
        return record

    async def fail_interrupted(self) -> None:
        count = await self._repo.fail_pending("Evaluation interrupted by a server restart")
        if count:
            logger.info("evaluations_marked_interrupted count=%d", count)

    async def shutdown(self) -> None:
        """Give in-flight evaluations a grace period, then cancel what is left."""
        if not self._tasks:
            return
        _, pending = await asyncio.wait(set(self._tasks), timeout=self._grace_s)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

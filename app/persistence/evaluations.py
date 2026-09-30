import sqlite3
from datetime import datetime, timezone
from uuid import UUID

from app.domain import EvaluationRecord, EvaluationScores, EvaluationStatus
from app.persistence.database import Database


class EvaluationRepository:
    def __init__(self, database: Database):
        self._db = database

    async def create_pending(self, evaluation_id: UUID, session_id: UUID, latency_ms: float) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO evaluations (evaluation_id, session_id, status, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(evaluation_id), str(session_id), EvaluationStatus.PENDING.value, latency_ms, now),
            )

        await self._db.run(work)

    async def complete(self, evaluation_id: UUID, scores: EvaluationScores) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                """UPDATE evaluations SET status = ?, answer_relevance = ?, faithfulness = ?, error = ?
                   WHERE evaluation_id = ?""",
                (
                    EvaluationStatus.COMPLETED.value,
                    scores.answer_relevance,
                    scores.faithfulness,
                    scores.error,
                    str(evaluation_id),
                ),
            )

        await self._db.run(work)

    async def fail(self, evaluation_id: UUID, error: str) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE evaluations SET status = ?, error = ? WHERE evaluation_id = ?",
                (EvaluationStatus.FAILED.value, error, str(evaluation_id)),
            )

        await self._db.run(work)

    async def get(self, session_id: UUID, evaluation_id: UUID) -> EvaluationRecord | None:
        def work(conn: sqlite3.Connection) -> EvaluationRecord | None:
            row = conn.execute(
                """SELECT status, latency_ms, answer_relevance, faithfulness, error FROM evaluations
                   WHERE evaluation_id = ? AND session_id = ?""",
                (str(evaluation_id), str(session_id)),
            ).fetchone()
            if row is None:
                return None
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                session_id=session_id,
                status=EvaluationStatus(row["status"]),
                latency_ms=row["latency_ms"],
                answer_relevance=row["answer_relevance"],
                faithfulness=row["faithfulness"],
                error=row["error"],
            )

        return await self._db.run(work)

    async def fail_pending(self, reason: str) -> int:
        """Mark evaluations left pending by a previous process as failed."""

        def work(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "UPDATE evaluations SET status = ?, error = ? WHERE status = ?",
                (EvaluationStatus.FAILED.value, reason, EvaluationStatus.PENDING.value),
            )
            return cursor.rowcount

        return await self._db.run(work)

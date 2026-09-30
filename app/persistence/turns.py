import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.domain import ChatTurn
from app.persistence.database import Database

_COLUMNS = "turn_id, question, query_used_for_retrieval, answer, sources_json, created_at"


def _to_turn(row: sqlite3.Row) -> ChatTurn:
    return ChatTurn(
        turn_id=row["turn_id"],
        question=row["question"],
        query_used_for_retrieval=row["query_used_for_retrieval"],
        answer=row["answer"],
        sources=json.loads(row["sources_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class TurnRepository:
    def __init__(self, database: Database):
        self._db = database

    async def add(
        self,
        session_id: UUID,
        question: str,
        query_used_for_retrieval: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO turns (session_id, question, query_used_for_retrieval, answer,
                                      sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(session_id),
                    question,
                    query_used_for_retrieval,
                    answer,
                    json.dumps(sources, ensure_ascii=False, default=str),
                    now,
                ),
            )

        await self._db.run(work)

    async def recent(self, session_id: UUID, limit: int) -> list[ChatTurn]:
        """The last `limit` turns, oldest first."""

        def work(conn: sqlite3.Connection) -> list[ChatTurn]:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM turns WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
                (str(session_id), limit),
            ).fetchall()
            return [_to_turn(r) for r in reversed(rows)]

        return await self._db.run(work)

    async def list_all(self, session_id: UUID) -> list[ChatTurn]:
        def work(conn: sqlite3.Connection) -> list[ChatTurn]:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM turns WHERE session_id = ? ORDER BY turn_id",
                (str(session_id),),
            ).fetchall()
            return [_to_turn(r) for r in rows]

        return await self._db.run(work)

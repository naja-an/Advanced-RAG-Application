import sqlite3
from datetime import datetime
from uuid import UUID

from app.domain import SessionRecord
from app.persistence.database import Database


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SessionRepository:
    def __init__(self, database: Database):
        self._db = database

    async def create(self, record: SessionRecord) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, expires_at) VALUES (?, ?, ?)",
                (str(record.session_id), _iso(record.created_at), _iso(record.expires_at)),
            )

        await self._db.run(work)

    async def get(self, session_id: UUID) -> SessionRecord | None:
        def work(conn: sqlite3.Connection) -> SessionRecord | None:
            row = conn.execute(
                "SELECT created_at, expires_at FROM sessions WHERE session_id = ?",
                (str(session_id),),
            ).fetchone()
            if row is None:
                return None
            return SessionRecord(session_id, _parse(row["created_at"]), _parse(row["expires_at"]))

        return await self._db.run(work)

    async def delete(self, session_id: UUID) -> bool:
        def work(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (str(session_id),))
            return cursor.rowcount > 0

        return await self._db.run(work)

    async def purge_expired(self, now: datetime) -> list[UUID]:
        def work(conn: sqlite3.Connection) -> list[UUID]:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (_iso(now),),
            ).fetchall()
            ids = [UUID(row["session_id"]) for row in rows]
            for session_id in ids:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (str(session_id),))
            return ids

        return await self._db.run(work)

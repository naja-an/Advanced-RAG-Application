import json
import sqlite3
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.documents import Document

from app.domain import DocumentRecord, DocumentStatus, IngestedSource
from app.persistence.database import Database


def _dump_chunks(chunks: list[Document]) -> str:
    payload = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    return json.dumps(payload, ensure_ascii=False, default=str)


def _load_chunks(raw: str) -> list[Document]:
    return [Document(page_content=i["content"], metadata=i["metadata"]) for i in json.loads(raw)]


class DocumentRepository:
    """Stores document records together with their parsed chunks, so search indexes can be
    rebuilt after a restart without re-parsing the original files."""

    def __init__(self, database: Database):
        self._db = database

    async def add(self, session_id: UUID, sources: list[IngestedSource]) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def work(conn: sqlite3.Connection) -> None:
            for source in sources:
                record = source.to_record()
                conn.execute(
                    """INSERT INTO documents (document_id, session_id, name, source, status, error,
                                              chunks_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(record.document_id),
                        str(session_id),
                        record.name,
                        record.source,
                        record.status.value,
                        record.error,
                        _dump_chunks(source.chunks),
                        now,
                    ),
                )

        await self._db.run(work)

    async def list_for_session(self, session_id: UUID) -> list[DocumentRecord]:
        def work(conn: sqlite3.Connection) -> list[DocumentRecord]:
            rows = conn.execute(
                """SELECT document_id, name, source, status, error FROM documents
                   WHERE session_id = ? ORDER BY rowid""",
                (str(session_id),),
            ).fetchall()
            return [
                DocumentRecord(
                    UUID(r["document_id"]), r["name"], r["source"], DocumentStatus(r["status"]), r["error"]
                )
                for r in rows
            ]

        return await self._db.run(work)

    async def load_chunks(self, session_id: UUID) -> list[Document]:
        def work(conn: sqlite3.Connection) -> list[Document]:
            rows = conn.execute(
                "SELECT chunks_json FROM documents WHERE session_id = ? ORDER BY rowid",
                (str(session_id),),
            ).fetchall()
            return [chunk for r in rows for chunk in _load_chunks(r["chunks_json"])]

        return await self._db.run(work)

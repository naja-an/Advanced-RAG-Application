"""SQLite persistence for chat turns and their retrieved source snapshots."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID


class ConversationStore:
	def __init__(self, database_path: Path):
		self.database_path = database_path
		self.database_path.parent.mkdir(parents=True, exist_ok=True)
		with self._connection() as connection:
			connection.executescript(
				"""
				CREATE TABLE IF NOT EXISTS conversation_sessions (
					session_id TEXT PRIMARY KEY,
					created_at TEXT NOT NULL
				);
				CREATE TABLE IF NOT EXISTS conversation_turns (
					turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
					session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
					question TEXT NOT NULL,
					query_used_for_retrieval TEXT NOT NULL,
					answer TEXT NOT NULL,
					sources_json TEXT NOT NULL,
					created_at TEXT NOT NULL
				);
				CREATE INDEX IF NOT EXISTS conversation_turns_session_order
					ON conversation_turns(session_id, turn_id);
				"""
			)

	def _connect(self) -> sqlite3.Connection:
		connection = sqlite3.connect(self.database_path, timeout=10)
		connection.row_factory = sqlite3.Row
		connection.execute("PRAGMA foreign_keys = ON")
		return connection

	@contextmanager
	def _connection(self) -> Iterator[sqlite3.Connection]:
		connection = self._connect()
		try:
			yield connection
			connection.commit()
		except Exception:
			connection.rollback()
			raise
		finally:
			connection.close()

	def create_session(self, session_id: UUID, created_at: datetime) -> None:
		with self._connection() as connection:
			connection.execute(
				"INSERT INTO conversation_sessions (session_id, created_at) VALUES (?, ?)",
				(str(session_id), created_at.isoformat()),
			)

	def add_turn(
		self,
		session_id: UUID,
		question: str,
		query_used_for_retrieval: str,
		answer: str,
		sources: list[dict[str, Any]],
	) -> None:
		with self._connection() as connection:
			connection.execute(
				"""
				INSERT INTO conversation_turns (
					session_id, question, query_used_for_retrieval, answer, sources_json, created_at
				) VALUES (?, ?, ?, ?, ?, ?)
				""",
				(
					str(session_id),
					question,
					query_used_for_retrieval,
					answer,
					json.dumps(sources, ensure_ascii=False, default=str),
					datetime.now(timezone.utc).isoformat(),
				),
			)

	def prompt_messages(
		self,
		session_id: UUID,
		max_turns: int,
		max_chars_per_message: int,
	) -> list[tuple[str, str]]:
		with self._connection() as connection:
			rows = connection.execute(
				"""
				SELECT question, answer FROM conversation_turns
				WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?
				""",
				(str(session_id), max_turns),
			).fetchall()
		return [
			(message[:max_chars_per_message], role)
			for row in reversed(rows)
			for message, role in ((row["question"], "human"), (row["answer"], "ai"))
		]

	def list_turns(self, session_id: UUID) -> list[dict[str, Any]] | None:
		with self._connection() as connection:
			exists = connection.execute(
				"SELECT 1 FROM conversation_sessions WHERE session_id = ?",
				(str(session_id),),
			).fetchone()
			if exists is None:
				return None
			rows = connection.execute(
				"""
				SELECT turn_id, question, query_used_for_retrieval, answer, sources_json, created_at
				FROM conversation_turns WHERE session_id = ? ORDER BY turn_id
				""",
				(str(session_id),),
			).fetchall()
		return [
			{
				"turn_id": row["turn_id"],
				"question": row["question"],
				"query_used_for_retrieval": row["query_used_for_retrieval"],
				"answer": row["answer"],
				"sources": json.loads(row["sources_json"]),
				"created_at": row["created_at"],
			}
			for row in rows
		]

	def delete_session(self, session_id: UUID) -> None:
		with self._connection() as connection:
			connection.execute(
				"DELETE FROM conversation_sessions WHERE session_id = ?",
				(str(session_id),),
			)

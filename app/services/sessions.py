import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.domain import SessionRecord
from app.exceptions import SessionNotFoundError
from app.ingestion.storage import UploadStorage
from app.persistence.sessions import SessionRepository

logger = logging.getLogger(__name__)


class SessionService:
    def __init__(self, repository: SessionRepository, storage: UploadStorage, ttl: timedelta | None = None):
        self._repo = repository
        self._storage = storage
        self._ttl = ttl
        self._on_delete: list[Callable[[UUID], None]] = []

    def on_delete(self, callback: Callable[[UUID], None]) -> None:
        """Register cleanup (e.g. evict caches) to run whenever a session goes away."""
        self._on_delete.append(callback)

    async def create(self) -> SessionRecord:
        now = datetime.now(timezone.utc)
        record = SessionRecord(uuid4(), now, now + self._ttl if self._ttl else None)
        await self._repo.create(record)
        logger.info("session_created session_id=%s", record.session_id)
        return record

    async def require(self, session_id: UUID) -> SessionRecord:
        record = await self._repo.get(session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        if record.expires_at is not None and record.expires_at <= datetime.now(timezone.utc):
            await self._discard(session_id)
            raise SessionNotFoundError(session_id)
        return record

    async def delete(self, session_id: UUID) -> None:
        if not await self._repo.delete(session_id):
            raise SessionNotFoundError(session_id)
        await self._cleanup(session_id)
        logger.info("session_deleted session_id=%s", session_id)

    async def purge_expired(self) -> None:
        for session_id in await self._repo.purge_expired(datetime.now(timezone.utc)):
            await self._cleanup(session_id)
            logger.info("session_expired session_id=%s", session_id)

    async def _discard(self, session_id: UUID) -> None:
        await self._repo.delete(session_id)
        await self._cleanup(session_id)

    async def _cleanup(self, session_id: UUID) -> None:
        await asyncio.to_thread(self._storage.remove, session_id)
        for callback in self._on_delete:
            callback(session_id)

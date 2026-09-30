import asyncio
import logging
from uuid import UUID

from app.domain import DocumentRecord, UploadedFile
from app.ingestion.ingestor import DocumentIngestor
from app.persistence.documents import DocumentRepository
from app.rag.index import IndexRegistry
from app.services.sessions import SessionService

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        sessions: SessionService,
        ingestor: DocumentIngestor,
        indexes: IndexRegistry,
        repository: DocumentRepository,
    ):
        self._sessions = sessions
        self._ingestor = ingestor
        self._indexes = indexes
        self._repo = repository

    async def list_documents(self, session_id: UUID) -> list[DocumentRecord]:
        await self._sessions.require(session_id)
        return await self._repo.list_for_session(session_id)

    async def add_documents(
        self, session_id: UUID, files: list[UploadedFile], urls: list[str]
    ) -> list[DocumentRecord]:
        await self._sessions.require(session_id)
        sources = await asyncio.to_thread(self._ingestor.ingest, session_id, files, urls)
        new_chunks = [chunk for source in sources for chunk in source.chunks]

        try:
            if new_chunks:
                index = await self._indexes.get(session_id, create=True)
                await index.add(new_chunks)  # only the new chunks are embedded
            await self._repo.add(session_id, sources)
        except BaseException:
            # The cached index may now disagree with the database; drop it so the next use
            # rebuilds it from what was actually persisted.
            self._indexes.evict(session_id)
            raise

        failed = sum(1 for source in sources if source.error)
        logger.info(
            "document_ingestion_completed session_id=%s sources=%d failed=%d chunks=%d",
            session_id,
            len(sources),
            failed,
            len(new_chunks),
        )
        return [source.to_record() for source in sources]

"""Turns a batch of uploaded files and URLs into parsed chunks (blocking; run in a thread)."""

import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid4

from app.domain import IngestedSource, UploadedFile
from app.exceptions import DocumentInputError, DocumentLoadError
from app.ingestion.loader import DocumentLoader
from app.ingestion.storage import UploadStorage
from app.ingestion.validators import validate_upload, validate_url

logger = logging.getLogger(__name__)


class DocumentIngestor:
    def __init__(
        self,
        storage: UploadStorage,
        loader: DocumentLoader,
        *,
        max_upload_bytes: int,
        max_sources: int,
    ):
        self._storage = storage
        self._loader = loader
        self._max_upload_bytes = max_upload_bytes
        self._max_sources = max_sources

    def ingest(self, session_id: UUID, files: list[UploadedFile], urls: list[str]) -> list[IngestedSource]:
        """Each source is isolated: one failed document never discards the others."""
        if not files and not urls:
            raise DocumentInputError("Provide at least one file or URL")
        if len(files) + len(urls) > self._max_sources:
            raise DocumentInputError(f"A request can contain at most {self._max_sources} sources")

        results = [self._ingest_file(session_id, upload) for upload in files]
        results += [self._ingest_url(url) for url in urls]
        return results

    def _ingest_file(self, session_id: UUID, upload: UploadedFile) -> IngestedSource:
        document_id = uuid4()
        display_name = Path(upload.filename).name or "upload"
        stored_path: Path | None = None
        try:
            name = validate_upload(upload.filename, len(upload.content), self._max_upload_bytes)
            stored_path = self._storage.save(session_id, name, upload.content)
            chunks = self._load(str(stored_path))
            return IngestedSource(document_id, name, f"upload/{document_id}/{name}", chunks)
        except (DocumentInputError, DocumentLoadError) as error:
            return self._failed(document_id, display_name, f"upload/{document_id}", error)
        finally:
            if stored_path is not None:  # chunks are persisted; the raw file is no longer needed
                stored_path.unlink(missing_ok=True)

    def _ingest_url(self, url: str) -> IngestedSource:
        document_id = uuid4()
        try:
            validated = validate_url(url)
            parsed = urlparse(validated)
            name = Path(parsed.path).name or parsed.netloc
            return IngestedSource(document_id, name, validated, self._load(validated))
        except (DocumentInputError, DocumentLoadError) as error:
            return self._failed(document_id, url, url, error)

    def _load(self, source: str):
        chunks = self._loader.load(source)
        if not chunks:
            raise DocumentLoadError("No content could be extracted from the document")
        return chunks

    @staticmethod
    def _failed(document_id: UUID, name: str, source: str, error: Exception) -> IngestedSource:
        logger.warning(
            "document_source_failed document_id=%s error_type=%s", document_id, type(error).__name__
        )
        return IngestedSource(document_id, name, source, [], str(error))

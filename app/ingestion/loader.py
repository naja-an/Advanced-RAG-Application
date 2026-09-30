"""Document parsing and chunking (Docling)."""

import logging
from typing import Protocol

from langchain_core.documents import Document

from app.exceptions import DocumentLoadError
from app.observability import observe

logger = logging.getLogger(__name__)


class DocumentLoader(Protocol):
    def load(self, source: str) -> list[Document]:
        """Parse a file path or URL into chunks. Raises DocumentLoadError on failure."""
        ...


class DoclingDocumentLoader:
    def __init__(self) -> None:
        self._chunker = None  # created once, on first use

    def _get_chunker(self):
        if self._chunker is None:
            from docling.chunking import HybridChunker

            self._chunker = HybridChunker()
        return self._chunker

    @observe(name="document.load")
    def load(self, source: str) -> list[Document]:
        try:
            from langchain_docling.loader import DoclingLoader, ExportType

            loader = DoclingLoader(
                file_path=source, export_type=ExportType.DOC_CHUNKS, chunker=self._get_chunker()
            )
            return loader.load()
        except Exception as exc:  # third-party boundary: Docling raises many unrelated types
            logger.warning("document_load_failed source_type=%s", type(exc).__name__, exc_info=True)
            raise DocumentLoadError(f"Error occurred while loading document: {exc}") from exc

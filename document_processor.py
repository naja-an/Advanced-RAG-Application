import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import urlparse
from uuid import UUID, uuid4

from langfuse import observe
from langchain_docling.loader import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from langchain_classic.vectorstores import FAISS

logger = logging.getLogger(__name__)


class DocumentLoaderException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class DocumentInputException(ValueError):
    """Raised when an upload or URL cannot be accepted for ingestion."""


@dataclass
class IngestedSource:
    """Result of processing one uploaded file or URL."""

    document_id: UUID
    name: str
    source: str
    documents: list
    stored_path: Path | None = None
    error: str | None = None


ALLOWED_FILE_EXTENSIONS = {
    ".csv",
    ".docx",
    ".html",
    ".md",
    ".pdf",
    ".pptx",
    ".txt",
    ".xlsx",
}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SOURCES_PER_REQUEST = 20


def create_session_storage() -> Path:
    """Create an isolated temporary directory for one API session."""
    return Path(mkdtemp(prefix="rag-session-"))


def remove_session_storage(path: Path) -> None:
    """Remove all files retained for a session."""
    shutil.rmtree(path, ignore_errors=True)


def save_uploaded_file(content: bytes, filename: str, directory: Path) -> Path:
    """Validate and save upload bytes using a generated, safe filename."""
    original_name = Path(filename).name
    if not original_name or original_name in {".", ".."}:
        raise DocumentInputException("Uploaded files must have a filename")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentInputException(
            f"{original_name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )

    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_FILE_EXTENSIONS:
        raise DocumentInputException(
            f"Unsupported file type: {extension or 'no extension'}"
        )

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    path = directory / f"{uuid4()}-{safe_name}"
    path.write_bytes(content)
    return path


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DocumentInputException(f"Invalid URL: {url}")
    return url


def ingest_sources(
    file_sources: list[tuple[str, bytes]],
    urls: list[str],
    storage_dir: Path,
) -> list[IngestedSource]:
    """Save and load a mixed batch of files and URLs.

    Each source is isolated so one failed document does not discard successful
    loads from the same request.
    """
    if not file_sources and not urls:
        raise DocumentInputException("Provide at least one file or URL")
    if len(file_sources) + len(urls) > MAX_SOURCES_PER_REQUEST:
        raise DocumentInputException(
            f"A request can contain at most {MAX_SOURCES_PER_REQUEST} sources"
        )

    results: list[IngestedSource] = []
    for filename, content in file_sources:
        document_id = uuid4()
        try:
            stored_path = save_uploaded_file(content, filename, storage_dir)
            documents = create_docs(str(stored_path))
            results.append(IngestedSource(
                document_id=document_id,
                name=Path(filename).name,
                source=f"upload/{document_id}/{Path(filename).name}",
                documents=documents,
                stored_path=stored_path,
            ))
        except (DocumentInputException, DocumentLoaderException) as error:
            logger.warning(
                "document_source_failed document_id=%s error_type=%s",
                document_id,
                type(error).__name__,
            )
            results.append(IngestedSource(
                document_id=document_id,
                name=Path(filename).name or "upload",
                source=f"upload/{document_id}",
                documents=[],
                error=str(error),
            ))

    for url in urls:
        document_id = uuid4()
        try:
            validated_url = _validate_url(url)
            parsed = urlparse(validated_url)
            name = Path(parsed.path).name or parsed.netloc
            documents = create_docs(validated_url)
            results.append(IngestedSource(
                document_id=document_id,
                name=name,
                source=validated_url,
                documents=documents,
            ))
        except (DocumentInputException, DocumentLoaderException) as error:
            logger.warning(
                "document_source_failed document_id=%s error_type=%s",
                document_id,
                type(error).__name__,
            )
            results.append(IngestedSource(
                document_id=document_id,
                name=url,
                source=url,
                documents=[],
                error=str(error),
            ))

    return results


def normalize_for_lexical_search(text: str) -> str:
    """Create the normalized representation used by lexical retrieval."""
    text = text.lower()
    text = re.sub(r"[.,!?;:\"'()\[\]{}<>/\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@observe(name="document.load")
def create_docs(path: str):
    """
    Create documents from a file or URL using DoclingLoader.
    Args:
        path (str): The path to the file or URL.
    Returns:
        List[Document]: A list of loaded documents.
    """
    try:
        loader = DoclingLoader(file_path=path,
                               export_type=ExportType.DOC_CHUNKS,
                               chunker=HybridChunker(),)
        doc_splits = loader.load()
    except Exception as e:
        raise DocumentLoaderException(f"Error occurred while loading document: {e}")

    return doc_splits


@observe(name="vectorstore.build")
def create_vectorstore(docs, embedding):
    """
    Create a FAISS vectorstore from the provided documents.
    Args:
        docs (List[Document]): A list of loaded documents.
        embedding: The embedding model to use.
    Returns:
        FAISS: A FAISS vectorstore.
    """
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding,
    )
    return vectorstore

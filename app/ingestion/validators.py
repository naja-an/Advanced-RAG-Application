from pathlib import Path
from urllib.parse import urlparse

from app.exceptions import DocumentInputError

ALLOWED_FILE_EXTENSIONS = {".csv", ".docx", ".html", ".md", ".pdf", ".pptx", ".txt", ".xlsx"}


def validate_upload(filename: str, size: int, max_bytes: int) -> str:
    """Return the clean display name, or raise DocumentInputError."""
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise DocumentInputError("Uploaded files must have a filename")
    if size > max_bytes:
        raise DocumentInputError(f"{name} exceeds the {max_bytes // (1024 * 1024)} MB limit")
    extension = Path(name).suffix.lower()
    if extension not in ALLOWED_FILE_EXTENSIONS:
        raise DocumentInputError(f"Unsupported file type: {extension or 'no extension'}")
    return name


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DocumentInputError(f"Invalid URL: {url}")
    return url

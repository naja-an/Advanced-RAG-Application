from fastapi import UploadFile

from app.domain import UploadedFile


async def read_uploads(files: list[UploadFile], max_bytes: int) -> list[UploadedFile]:
    """Read uploads without ever buffering more than `max_bytes + 1` per file. An oversized file
    arrives as `max_bytes + 1` bytes and is rejected by validation as a failed document."""
    return [
        UploadedFile(upload.filename, await upload.read(max_bytes + 1)) for upload in files if upload.filename
    ]

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.dependencies import get_document_service, get_settings
from app.api.schemas import DocumentResponse, ErrorResponse
from app.api.uploads import read_uploads
from app.config import Settings
from app.services.documents import DocumentService

router = APIRouter(prefix="/sessions/{session_id}/documents", tags=["documents"])


@router.get("", response_model=list[DocumentResponse], responses={404: {"model": ErrorResponse}})
async def list_documents(
    session_id: UUID, service: DocumentService = Depends(get_document_service)
) -> list[DocumentResponse]:
    return [DocumentResponse.model_validate(d) for d in await service.list_documents(session_id)]


@router.post(
    "",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def add_documents(
    session_id: UUID,
    files: Annotated[list[UploadFile] | None, File()] = None,
    urls: Annotated[list[str] | None, Form()] = None,
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> list[DocumentResponse]:
    uploads = await read_uploads(files or [], settings.max_upload_bytes)
    cleaned_urls = [url.strip() for url in (urls or []) if url.strip()]
    records = await service.add_documents(session_id, uploads, cleaned_urls)
    return [DocumentResponse.model_validate(r) for r in records]

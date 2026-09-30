from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_session_service
from app.api.schemas import ErrorResponse, SessionResponse
from app.services.sessions import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(service: SessionService = Depends(get_session_service)) -> SessionResponse:
    return SessionResponse.model_validate(await service.create())


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def delete_session(session_id: UUID, service: SessionService = Depends(get_session_service)) -> None:
    await service.delete(session_id)

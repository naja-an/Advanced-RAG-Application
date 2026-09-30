from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTurnResponse,
    ErrorResponse,
    EvaluationResponse,
    SourceResponse,
)
from app.services.chat import ChatService

router = APIRouter(prefix="/sessions/{session_id}", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def chat(
    session_id: UUID, request: ChatRequest, service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    outcome = await service.chat(session_id, request.question)
    result = outcome.result
    return ChatResponse(
        question=request.question,
        query_used_for_retrieval=result.query_used_for_retrieval,
        answer=result.answer,
        evaluation=EvaluationResponse(latency_ms=outcome.latency_ms),
        evaluation_id=outcome.evaluation_id,
        sources=[SourceResponse(metadata=d.metadata, content=d.page_content) for d in result.documents],
    )


@router.get("/history", response_model=list[ChatTurnResponse], responses={404: {"model": ErrorResponse}})
async def get_history(
    session_id: UUID, service: ChatService = Depends(get_chat_service)
) -> list[ChatTurnResponse]:
    return [ChatTurnResponse.model_validate(t) for t in await service.history(session_id)]

from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_evaluation_service, get_session_service
from app.api.schemas import ErrorResponse, EvaluationResponse, EvaluationTaskResponse
from app.evaluation.service import EvaluationService
from app.services.sessions import SessionService

router = APIRouter(prefix="/sessions/{session_id}/evaluations", tags=["evaluations"])


@router.get(
    "/{evaluation_id}",
    response_model=EvaluationTaskResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_evaluation(
    session_id: UUID,
    evaluation_id: UUID,
    sessions: SessionService = Depends(get_session_service),
    evaluations: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationTaskResponse:
    await sessions.require(session_id)
    record = await evaluations.get(session_id, evaluation_id)
    return EvaluationTaskResponse(
        status=record.status.value,
        evaluation=EvaluationResponse(
            answer_relevance=record.answer_relevance,
            faithfulness=record.faithfulness,
            latency_ms=record.latency_ms,
            error=record.error,
        ),
    )

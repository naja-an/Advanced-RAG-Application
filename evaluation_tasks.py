"""Background evaluation task orchestration."""

import logging
from uuid import UUID

from evaluation import evaluate_answer
from models import EvaluationResponse, EvaluationTaskResponse
from session_manager import SessionManager, SessionNotFoundError

logger = logging.getLogger(__name__)


def run_answer_evaluation(
	session_manager: SessionManager,
	session_id: UUID,
	evaluation_id: UUID,
	question: str,
	answer: str,
	retrieval_context: list[str],
	latency_ms: float,
) -> None:
	"""Evaluate a completed answer and save its result to the owning session."""
	try:
		evaluation = evaluate_answer(question, answer, retrieval_context)
		result = EvaluationTaskResponse(
			status="completed",
			evaluation=EvaluationResponse(**{
				**evaluation,
				"latency_ms": latency_ms,
			}),
		)
	except Exception as error:
		logger.exception("answer_evaluation_unavailable session_id=%s", session_id)
		result = EvaluationTaskResponse(
			status="failed",
			evaluation=EvaluationResponse(
				latency_ms=latency_ms,
				error=f"Evaluation unavailable: {type(error).__name__}",
			),
		)

	try:
		session_manager.set_evaluation(session_id, evaluation_id, result)
	except SessionNotFoundError:
		logger.info("evaluation_discarded_session_deleted session_id=%s", session_id)
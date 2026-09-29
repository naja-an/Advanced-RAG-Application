"""DeepEval metrics for per-turn RAG answer evaluation."""

import logging
import os
from typing import Any

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


def evaluate_answer(
	question: str,
	answer: str,
	retrieval_context: list[str],
) -> dict[str, Any]:
	"""Score answer relevance and faithfulness against the retrieved context."""
	judge = GeminiModel(
		model=os.getenv("RAG_EVALUATION_MODEL", os.getenv("RAG_MODEL", "gemini-3.5-flash-lite")),
		api_key=os.getenv("GOOGLE_API_KEY"),
	)
	test_case = LLMTestCase(
		input=question,
		actual_output=answer,
		retrieval_context=retrieval_context,
	)
	results: dict[str, Any] = {
		"answer_relevance": None,
		"faithfulness": None,
		"error": None,
	}
	metric_failures = []

	for name, metric_type in (
		("answer_relevance", AnswerRelevancyMetric),
		("faithfulness", FaithfulnessMetric),
	):
		try:
			metric = metric_type(model=judge, include_reason=False)
			metric.measure(test_case, _show_indicator=False)
			results[name] = metric.score
		except Exception as error:
			logger.exception("answer_evaluation_failed metric=%s", name)
			metric_failures.append(type(error).__name__)

	if metric_failures:
		results["error"] = "Evaluation unavailable: " + ", ".join(metric_failures)
	return results

"""DeepEval-based answer scoring (blocking; called from a worker thread)."""

import logging
import threading
from typing import Protocol

from app.domain import EvaluationScores

logger = logging.getLogger(__name__)


class AnswerEvaluator(Protocol):
    def evaluate(self, question: str, answer: str, contexts: list[str]) -> EvaluationScores: ...


class DeepEvalEvaluator:
    """Builds the judge model once (lazily) instead of once per answer."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._judge = None
        self._lock = threading.Lock()

    def _get_judge(self):
        with self._lock:
            if self._judge is None:
                from deepeval.models import GeminiModel

                self._judge = GeminiModel(model=self._model_name)
            return self._judge

    def evaluate(self, question: str, answer: str, contexts: list[str]) -> EvaluationScores:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        judge = self._get_judge()
        test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
        scores: dict[str, float | None] = {"answer_relevance": None, "faithfulness": None}
        failures: list[str] = []
        for name, metric_type in (
            ("answer_relevance", AnswerRelevancyMetric),
            ("faithfulness", FaithfulnessMetric),
        ):
            try:
                metric = metric_type(model=judge, include_reason=False)
                metric.measure(test_case, _show_indicator=False)
                scores[name] = metric.score
            except Exception as exc:  # DeepEval/provider errors vary; one metric must not sink the other
                logger.warning("answer_evaluation_failed metric=%s", name, exc_info=True)
                failures.append(type(exc).__name__)
        error = ("Evaluation unavailable: " + ", ".join(failures)) if failures else None
        return EvaluationScores(scores["answer_relevance"], scores["faithfulness"], error)

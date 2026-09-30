"""LLM access with retries and timeouts, raising typed errors."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.exceptions import LLMError, LLMUnavailableError
from app.rag.resilience import is_provider_error, is_transient

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    """The slice of a LangChain chat model that we use."""

    async def ainvoke(self, input: Sequence[BaseMessage], config: RunnableConfig | None = None) -> Any: ...


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content).strip()


class ResilientLLM:
    """Retries transient failures with jittered backoff, bounds each attempt with a timeout and
    converts provider failures into `LLMUnavailableError` / `LLMError`. Anything else (a bug)
    propagates unchanged."""

    def __init__(
        self,
        model: ChatModel,
        *,
        max_attempts: int = 3,
        timeout_s: float = 30.0,
        base_delay_s: float = 0.5,
        max_delay_s: float = 4.0,
    ):
        self._model = model
        self._max_attempts = max_attempts
        self._timeout_s = timeout_s
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s

    @staticmethod
    def _log_retry(state: RetryCallState) -> None:
        error = state.outcome.exception() if state.outcome else None
        logger.warning("llm_retry attempt=%d error=%s", state.attempt_number, type(error).__name__)

    async def complete(self, messages: Sequence[BaseMessage], config: RunnableConfig | None = None) -> str:
        reply: Any = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=wait_random_exponential(multiplier=self._base_delay_s, max=self._max_delay_s),
                retry=retry_if_exception(is_transient),
                before_sleep=self._log_retry,
                reraise=True,
            ):
                with attempt:
                    reply = await asyncio.wait_for(
                        self._model.ainvoke(list(messages), config=config), timeout=self._timeout_s
                    )
        except Exception as exc:
            if is_transient(exc):
                raise LLMUnavailableError() from exc
            if is_provider_error(exc):
                raise LLMError(f"The language model request failed: {type(exc).__name__}") from exc
            raise
        return message_text(reply)

"""Tracing (Langfuse). Created once by the container; a no-op when credentials are absent."""

import logging
from collections.abc import Callable
from typing import Any

from langfuse import observe

from app.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["Tracing", "create_tracing", "observe"]


class Tracing:
    def __init__(self, handler: Any | None = None, flusher: Callable[[], None] | None = None):
        self._handler = handler
        self._flusher = flusher

    def callbacks(self) -> list[Any]:
        return [self._handler] if self._handler else []

    def flush(self) -> None:
        """Send buffered events; called on shutdown so traces are not lost."""
        if self._flusher:
            try:
                self._flusher()
            except Exception:  # never let telemetry break shutdown
                logger.warning("langfuse_flush_failed", exc_info=True)


def create_tracing(settings: Settings) -> Tracing:
    if not settings.tracing_enabled:
        return Tracing()
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    return Tracing(CallbackHandler(), lambda: get_client().flush())

"""Domain exceptions. Each carries the HTTP status it maps to; the mapping is applied once
by the handler in `app.api.error_handlers`, never inside individual endpoints."""


class AppError(Exception):
    status_code = 500
    default_detail = "Internal error"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class SessionNotFoundError(AppError):
    status_code = 404

    def __init__(self, session_id):
        super().__init__(f"Session {session_id} was not found")


class EvaluationNotFoundError(AppError):
    status_code = 404

    def __init__(self, evaluation_id):
        super().__init__(f"Evaluation {evaluation_id} was not found")


class NoDocumentsError(AppError):
    status_code = 400
    default_detail = "Upload at least one document before starting a chat"


class DocumentInputError(AppError):
    """An upload or URL cannot be accepted (bad type, too large, invalid URL, ...)."""

    status_code = 400


class DocumentLoadError(AppError):
    """One source could not be parsed. Recorded per document, not raised to the client."""

    status_code = 422


class IndexingError(AppError):
    status_code = 502
    default_detail = "Unable to index the documents right now"


class RetrievalError(AppError):
    status_code = 502
    default_detail = "Retrieval is temporarily unavailable"


class LLMError(AppError):
    status_code = 502
    default_detail = "The language model request failed"


class LLMUnavailableError(LLMError):
    status_code = 503
    default_detail = "The language model is temporarily unavailable, please retry"


class ChatTimeoutError(AppError):
    status_code = 504
    default_detail = "The answer took too long to generate"


class ConfigurationError(Exception):
    """Raised at startup when required configuration is missing."""

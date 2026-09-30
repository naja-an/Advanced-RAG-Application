"""Classifies provider errors so callers can retry transient ones and let real bugs surface."""

import asyncio

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = {
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    "InternalServerError",
    "TooManyRequests",
    "ServerError",
    "ReadTimeout",
    "ConnectTimeout",
}
_PROVIDER_MODULES = {"google", "grpc", "httpx", "httpcore", "requests", "urllib3", "aiohttp"}


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def is_transient(exc: BaseException) -> bool:
    """Rate limits, 5xx responses, timeouts and network failures: worth retrying."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if _status_code(exc) in _TRANSIENT_STATUS:
        return True
    return type(exc).__name__ in _TRANSIENT_NAMES


def is_provider_error(exc: BaseException) -> bool:
    """Any failure that originates in an external API or network layer (as opposed to a bug
    in this codebase, which must propagate with its traceback)."""
    if is_transient(exc) or _status_code(exc) is not None:
        return True
    return type(exc).__module__.split(".")[0] in _PROVIDER_MODULES

"""One place that turns domain exceptions into HTTP responses."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import AppError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error(
                "request_failed path=%s error=%s", request.url.path, type(exc).__name__, exc_info=exc
            )
        else:
            logger.info("request_rejected path=%s error=%s", request.url.path, type(exc).__name__)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

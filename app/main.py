"""Application factory.  Run with:  uvicorn app.main:create_app --factory"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.routes import chat, documents, evaluations, meta, sessions
from app.config import Settings, get_settings
from app.container import ContainerOverrides, create_container
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, overrides: ContainerOverrides | None = None) -> FastAPI:
    """Build the app. Nothing heavy happens here: models, the database and tracing are created in
    the lifespan, so importing this module has no side effects."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        configure_logging(resolved.log_level)
        container = await create_container(resolved, overrides)
        await container.startup()
        app.state.container = container
        logger.info("application_started")
        try:
            yield
        finally:
            await container.shutdown()  # drains evaluations and flushes Langfuse
            logger.info("application_stopped")

    app = FastAPI(
        title="RAG Chatbot API",
        version="0.2.0",
        description="Session-based API for document-grounded conversations.",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    for module in (meta, sessions, documents, chat, evaluations):
        app.include_router(module.router)
    return app

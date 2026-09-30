"""FastAPI dependency providers. Everything is read from the container that the lifespan put on
`app.state`, so tests can swap any piece via `create_app(overrides=...)` or
`app.dependency_overrides`."""

from fastapi import Depends, Request

from app.config import Settings
from app.container import Container
from app.evaluation.service import EvaluationService
from app.services.chat import ChatService
from app.services.documents import DocumentService
from app.services.sessions import SessionService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_settings(container: Container = Depends(get_container)) -> Settings:
    return container.settings


def get_session_service(container: Container = Depends(get_container)) -> SessionService:
    return container.sessions


def get_document_service(container: Container = Depends(get_container)) -> DocumentService:
    return container.documents


def get_chat_service(container: Container = Depends(get_container)) -> ChatService:
    return container.chat


def get_evaluation_service(container: Container = Depends(get_container)) -> EvaluationService:
    return container.evaluations

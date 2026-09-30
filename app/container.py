"""Composition root: builds every long-lived object once and wires dependencies explicitly.

Nothing here runs at import time. `create_container` is called from the FastAPI lifespan, and
tests pass fakes through `ContainerOverrides`."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.evaluation.metrics import AnswerEvaluator, DeepEvalEvaluator
from app.evaluation.service import EvaluationService
from app.exceptions import ConfigurationError
from app.ingestion.ingestor import DocumentIngestor
from app.ingestion.loader import DoclingDocumentLoader, DocumentLoader
from app.ingestion.storage import UploadStorage
from app.observability import Tracing, create_tracing
from app.persistence.database import Database
from app.persistence.documents import DocumentRepository
from app.persistence.evaluations import EvaluationRepository
from app.persistence.sessions import SessionRepository
from app.persistence.turns import TurnRepository
from app.rag.index import IndexRegistry, SessionIndex
from app.rag.llm import ChatModel, ResilientLLM
from app.rag.pipeline import RAGPipeline
from app.services.chat import ChatService
from app.services.documents import DocumentService
from app.services.sessions import SessionService

logger = logging.getLogger(__name__)


@dataclass
class ContainerOverrides:
    """Replace real integrations (Gemini, Docling, the reranker, DeepEval, Langfuse) in tests."""

    llm: ChatModel | None = None
    embeddings: Embeddings | None = None
    reranker: BaseCrossEncoder | None = None
    loader: DocumentLoader | None = None
    evaluator: AnswerEvaluator | None = None
    tracing: Tracing | None = None


@dataclass
class Container:
    settings: Settings
    tracing: Tracing
    sessions: SessionService
    documents: DocumentService
    chat: ChatService
    evaluations: EvaluationService

    async def startup(self) -> None:
        await self.sessions.purge_expired()
        await self.evaluations.fail_interrupted()

    async def shutdown(self) -> None:
        await self.evaluations.shutdown()
        await asyncio.to_thread(self.tracing.flush)


def _require_api_key(settings: Settings) -> None:
    if settings.google_api_key is None:
        raise ConfigurationError("GOOGLE_API_KEY is not set (add it to your environment or .env file)")


def _create_llm(settings: Settings) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=0,
        max_output_tokens=settings.max_output_tokens,
        max_retries=0,  # retries are handled once, in ResilientLLM
    )


def _create_embeddings(settings: Settings) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


def _create_reranker(settings: Settings) -> BaseCrossEncoder:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=settings.reranker_model)


async def create_container(settings: Settings, overrides: ContainerOverrides | None = None) -> Container:
    o = overrides or ContainerOverrides()
    if o.llm is None or o.embeddings is None or o.evaluator is None:
        _require_api_key(settings)

    # Heavy or blocking construction happens once, off the event loop.
    reranker = o.reranker or await asyncio.to_thread(_create_reranker, settings)
    llm = o.llm or _create_llm(settings)
    embeddings = o.embeddings or _create_embeddings(settings)
    tracing = o.tracing or create_tracing(settings)

    database = await asyncio.to_thread(Database, settings.database_path)
    storage = UploadStorage(settings.upload_dir)
    documents_repo = DocumentRepository(database)
    turns_repo = TurnRepository(database)

    resilient_llm = ResilientLLM(
        llm,
        max_attempts=settings.llm_max_attempts,
        timeout_s=settings.llm_timeout_s,
        base_delay_s=settings.llm_retry_base_delay_s,
        max_delay_s=settings.llm_retry_max_delay_s,
    )
    pipeline = RAGPipeline(
        resilient_llm,
        history_max_chars=settings.history_max_chars,
        rewrite_history_turns=settings.rewrite_history_turns,
        rewrite_max_chars=settings.rewrite_max_chars,
    )
    indexes = IndexRegistry(
        factory=lambda: SessionIndex(
            embeddings,
            reranker,
            vector_k=settings.vector_k,
            lexical_k=settings.lexical_k,
            final_k=settings.final_k,
        ),
        chunk_source=documents_repo,
    )

    ttl = timedelta(hours=settings.session_ttl_hours) if settings.session_ttl_hours else None
    sessions = SessionService(SessionRepository(database), storage, ttl)
    ingestor = DocumentIngestor(
        storage,
        o.loader or DoclingDocumentLoader(),
        max_upload_bytes=settings.max_upload_bytes,
        max_sources=settings.max_sources_per_request,
    )
    evaluations = EvaluationService(
        EvaluationRepository(database),
        o.evaluator or DeepEvalEvaluator(settings.judge_model),
        concurrency=settings.evaluation_concurrency,
        shutdown_grace_s=settings.shutdown_grace_s,
    )
    chat = ChatService(
        sessions,
        indexes,
        pipeline,
        turns_repo,
        evaluations,
        tracing,
        history_max_turns=settings.history_max_turns,
        timeout_s=settings.chat_timeout_s,
    )
    documents = DocumentService(sessions, ingestor, indexes, documents_repo)

    sessions.on_delete(indexes.evict)
    sessions.on_delete(chat.forget)
    return Container(settings, tracing, sessions, documents, chat, evaluations)

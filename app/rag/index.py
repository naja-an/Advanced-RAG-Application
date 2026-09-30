"""Per-session search indexes.

`SessionIndex` grows incrementally: only new chunks are embedded, and the slow embedding call
runs outside the lock, so uploads never block chat for long. `IndexRegistry` owns the cache
and lazily rebuilds an index from persisted chunks after a restart."""

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.exceptions import IndexingError, RetrievalError
from app.observability import observe
from app.rag.resilience import is_provider_error
from app.rag.retrieval import build_hybrid_retriever
from app.utils.locks import KeyedLocks

logger = logging.getLogger(__name__)


class ChunkSource(Protocol):
    async def load_chunks(self, session_id: UUID) -> list[Document]: ...


class SessionIndex:
    def __init__(
        self,
        embeddings: Embeddings,
        reranker: BaseCrossEncoder,
        *,
        vector_k: int,
        lexical_k: int,
        final_k: int,
    ):
        self._embeddings = embeddings
        self._reranker = reranker
        self._k = {"vector_k": vector_k, "lexical_k": lexical_k, "final_k": final_k}
        self._chunks: list[Document] = []
        self._store: FAISS | None = None
        self._retriever: BaseRetriever | None = None
        self._lock = asyncio.Lock()  # guards the FAISS store, chunk list and retriever

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    async def add(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        texts = [chunk.page_content for chunk in chunks]
        try:
            vectors = await self._embeddings.aembed_documents(texts)  # slow; outside the lock
        except Exception as exc:
            if is_provider_error(exc):
                raise IndexingError() from exc
            raise
        async with self._lock:
            await asyncio.to_thread(self._apply, chunks, list(zip(texts, vectors, strict=True)))

    def _apply(self, chunks: list[Document], pairs: list[tuple[str, list[float]]]) -> None:
        metadatas = [chunk.metadata for chunk in chunks]
        if self._store is None:
            self._store = FAISS.from_embeddings(pairs, self._embeddings, metadatas=metadatas)
        else:
            self._store.add_embeddings(pairs, metadatas=metadatas)
        self._chunks = [*self._chunks, *chunks]
        self._retriever = build_hybrid_retriever(self._store, self._chunks, self._reranker, **self._k)

    @observe(name="rag.retrieve")
    async def search(self, query: str) -> list[Document]:
        async with self._lock:
            retriever = self._retriever
            if retriever is None:
                return []
            try:
                documents = await retriever.ainvoke(query)
            except Exception as exc:
                if is_provider_error(exc):
                    raise RetrievalError() from exc
                raise
        logger.debug("retrieval_completed document_count=%d", len(documents))
        return documents


class IndexRegistry:
    def __init__(self, factory: Callable[[], SessionIndex], chunk_source: ChunkSource):
        self._factory = factory
        self._chunks = chunk_source
        self._indexes: dict[UUID, SessionIndex] = {}
        self._locks = KeyedLocks()

    async def get(self, session_id: UUID, *, create: bool = False) -> SessionIndex | None:
        """Return the session's index, rebuilding it from persisted chunks if it is not cached.
        Returns None when the session has no chunks, unless `create` is set."""
        if (cached := self._indexes.get(session_id)) is not None:
            return cached
        async with self._locks.get(session_id):
            if (cached := self._indexes.get(session_id)) is not None:
                return cached
            persisted = await self._chunks.load_chunks(session_id)
            if not persisted and not create:
                return None
            index = self._factory()
            if persisted:
                await index.add(persisted)
                logger.info("index_rebuilt session_id=%s chunks=%d", session_id, len(persisted))
            self._indexes[session_id] = index
            return index

    def evict(self, session_id: UUID) -> None:
        """Drop the cached index (it is rebuilt from the database on next use)."""
        self._indexes.pop(session_id, None)
        self._locks.discard(session_id)

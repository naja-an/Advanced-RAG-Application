"""Stateless RAG flow: rewrite -> retrieve -> generate.

The pipeline holds no per-session state. History and the retriever are passed in per call, so a
single instance serves every session and never needs rebuilding."""

import logging
from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.domain import ChatTurn, RAGResult
from app.exceptions import LLMError
from app.observability import observe
from app.rag.llm import ResilientLLM
from app.rag.prompts import ANSWER_PROMPT, NOT_FOUND_MESSAGE, REWRITE_QUERY_PROMPT

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    async def search(self, query: str) -> list[Document]: ...


class RAGPipeline:
    def __init__(
        self,
        llm: ResilientLLM,
        *,
        history_max_chars: int = 2000,
        rewrite_history_turns: int = 2,
        rewrite_max_chars: int = 400,
    ):
        self._llm = llm
        self._history_max_chars = history_max_chars
        self._rewrite_history_turns = rewrite_history_turns
        self._rewrite_max_chars = rewrite_max_chars

    @observe(name="rag.answer", capture_input=False)
    async def answer(
        self,
        question: str,
        history: Sequence[ChatTurn],
        retriever: Retriever,
        config: RunnableConfig | None = None,
    ) -> RAGResult:
        query = await self._rewrite_query(question, history, config)
        documents = await retriever.search(query)
        if not documents:
            return RAGResult(NOT_FOUND_MESSAGE, [], query)

        context = "\n\n".join(f"[Source {i}]\n{d.page_content}" for i, d in enumerate(documents, start=1))
        messages = [
            SystemMessage(ANSWER_PROMPT),
            *self._history_messages(history),
            HumanMessage(f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer = await self._llm.complete(messages, config)
        return RAGResult(answer, documents, query)

    def _history_messages(self, history: Sequence[ChatTurn]) -> list[BaseMessage]:
        limit = self._history_max_chars
        messages: list[BaseMessage] = []
        for turn in history:
            messages.append(HumanMessage(turn.question[:limit]))
            messages.append(AIMessage(turn.answer[:limit]))
        return messages

    async def _rewrite_query(
        self, question: str, history: Sequence[ChatTurn], config: RunnableConfig | None
    ) -> str:
        if not history:  # first turn: nothing to resolve, skip the LLM call
            return question
        limit = self._rewrite_max_chars
        transcript = "\n".join(
            f"User: {turn.question[:limit]}\nAssistant: {turn.answer[:limit]}"
            for turn in history[-self._rewrite_history_turns :]
        )
        try:
            rewritten = await self._llm.complete(
                [
                    SystemMessage(REWRITE_QUERY_PROMPT),
                    HumanMessage(f"Conversation:\n{transcript}\n\nLatest question: {question}"),
                ],
                config,
            )
        except LLMError:
            # Rewriting is an optimisation: fall back to the raw question. Bugs still propagate.
            logger.warning("query_rewrite_failed, using the raw question", exc_info=True)
            return question
        return rewritten or question

"""Deterministic RAG flow (no agent loop).

Per turn:
    1. condense   - rewrite follow-ups into a standalone query (skipped on turn 1)
    2. retrieve   - BM25 + vector -> cross-encoder rerank
    3. generate   - ONE LLM call over [system, trimmed chat history, fresh context + question]

Conversation history stores ONLY the user question and final answer text.
Retrieved chunks are injected into the current turn's prompt and never saved,
so there are no tool messages to trim and the context stays small.
"""

import asyncio
import logging
from functools import lru_cache

from langchain_classic.retrievers import BM25Retriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from document_processor import normalize_for_lexical_search

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Sorry, I couldn't find that in the document."

ANSWER_PROMPT = f"""You answer questions using ONLY the numbered context passages provided in the user's message.
Cite supporting passages as [Source N].
If the passages do not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
Earlier conversation turns are for continuity only; never use them as a source of facts."""

CONDENSE_PROMPT = """Rewrite the user's latest question as a single standalone search query,
resolving pronouns and references using the conversation. Do not answer it.
If it is already standalone, return it unchanged. Output only the query."""


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = {
    "ResourceExhausted", "ServiceUnavailable", "DeadlineExceeded", "InternalServerError",
    "TooManyRequests", "ServerError", "ReadTimeout", "ConnectTimeout",
}


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying (rate limits, 5xx, timeouts, network).

    Matches on status code / class name so it works across Gemini SDK versions
    (google.api_core.exceptions.* and google.genai.errors.*).
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _TRANSIENT_STATUS:
        return True
    return type(exc).__name__ in _TRANSIENT_NAMES


# --------------------------------------------------------------------------- #
# Conversation memory (lives on the session, survives pipeline rebuilds)
# --------------------------------------------------------------------------- #
class ConversationMemory:
    """Sliding window of the last `max_turns` (question, answer) pairs as plain text."""

    def __init__(self, max_turns: int = 6, max_chars_per_message: int = 2000):
        self.max_turns = max_turns
        self.max_chars = max_chars_per_message
        self._messages: list[BaseMessage] = []
        self.lock = asyncio.Lock()  # serialises turns within one session

    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def add_turn(self, question: str, answer: str) -> None:
        self._messages += [
            HumanMessage(question[: self.max_chars]),
            AIMessage(answer[: self.max_chars]),
        ]
        self._messages = self._messages[-2 * self.max_turns:]

    def clear(self) -> None:
        self._messages.clear()


# --------------------------------------------------------------------------- #
# Reranker singleton (~1 GB; must not be loaded per session/upload)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_reranker_model(name: str = "BAAI/bge-reranker-base") -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(model_name=name)


class RAGPipeline:
    def __init__(
        self,
        llm,
        vectorstore,
        docs: list[Document],
        vector_k: int = 8,
        lexical_k: int = 8,
        final_k: int = 5,
        max_attempts: int = 3,
        llm_timeout_s: float = 30.0,
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self.docs = docs
        self.max_attempts = max_attempts
        self.llm_timeout_s = llm_timeout_s
        self.memory = ConversationMemory()

        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_k})
        bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=normalize_for_lexical_search)
        bm25_retriever.k = lexical_k
        ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever])
        reranker = CrossEncoderReranker(model=get_reranker_model(), top_n=final_k)
        self.retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble)

    # ---- LLM call with retries + per-attempt timeout ----------------------- #
    async def _llm_call(self, messages: list[BaseMessage], config: RunnableConfig | None):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_random_exponential(multiplier=0.5, max=4),
            retry=retry_if_exception(is_transient),
            before_sleep=lambda rs: logger.warning(
                "llm_retry attempt=%d error=%s", rs.attempt_number, type(rs.outcome.exception()).__name__
            ),
            reraise=True,
        ):
            with attempt:
                return await asyncio.wait_for(
                    self.llm.ainvoke(messages, config=config), timeout=self.llm_timeout_s
                )

    @staticmethod
    def _text(message) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content).strip()

    # ---- step 1: condense --------------------------------------------------- #
    async def _standalone_query(self, question: str, history: list[BaseMessage], config) -> str:
        if not history:  # first turn: no LLM call needed
            return question
        transcript = "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {self._text(m)[:400]}" for m in history[-4:]
        )
        try:
            reply = await self._llm_call(
                [SystemMessage(CONDENSE_PROMPT),
                 HumanMessage(f"Conversation:\n{transcript}\n\nLatest question: {question}")],
                config,
            )
            return self._text(reply) or question
        except Exception as e:  # condensing is an optimisation; never fail the turn on it
            logger.warning("condense_failed, using raw question: %s", type(e).__name__)
            return question

    # ---- step 2: retrieve --------------------------------------------------- #
    @observe(name="rag.retrieve_documents")
    async def retrieve_documents(self, query: str) -> list[Document]:
        documents = await self.retriever.ainvoke(query)
        logger.debug("retrieval_completed document_count=%d", len(documents))
        return documents

    # ---- step 3: generate --------------------------------------------------- #
    @observe(name="rag.generate_answer")
    async def generate_answer(
        self, question: str, config: RunnableConfig | None = None
    ) -> dict:
        try:
            async with self.memory.lock:
                history = self.memory.messages()
                query = await self._standalone_query(question, history, config)
                documents = await self.retrieve_documents(query)

                if not documents:
                    answer = NOT_FOUND_MESSAGE
                else:
                    context = "\n\n".join(
                        f"[Source {i}]\n{d.page_content}" for i, d in enumerate(documents, start=1)
                    )
                    reply = await self._llm_call(
                        [SystemMessage(ANSWER_PROMPT), *history,
                         HumanMessage(f"Context:\n{context}\n\nQuestion: {query}")],
                        config,
                    )
                    answer = self._text(reply)

                self.memory.add_turn(question, answer)  # text only - chunks are never stored
            return {"answer": answer, "documents": documents, "query_used_for_retrieval": query}
        except Exception as e:
            logger.error("answer_generation_failed: %s", e)
            raise RAGException(f"Error occurred while response generation: {e}") from e


class RAGException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message

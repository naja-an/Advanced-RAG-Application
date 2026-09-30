"""Test doubles for every external integration."""

import asyncio
from types import SimpleNamespace

from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from app.domain import EvaluationScores
from app.exceptions import DocumentLoadError


class ServiceUnavailable(Exception):
    """Looks like a transient provider error (matched by class name)."""


class BadRequest(Exception):
    """Looks like a permanent provider error (has an HTTP-style code)."""

    code = 400


class FakeChatModel:
    """Answers rewrite calls and answer calls; can be scripted to fail."""

    def __init__(self):
        self.calls: list[list] = []
        self.failures: list[BaseException] = []  # raised one per call, in order
        self.delay_s = 0.0

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.failures:
            raise self.failures.pop(0)
        system = messages[0].content
        if system.startswith("Rewrite"):
            last = messages[-1].content.split("Latest question:")[-1].strip()
            return SimpleNamespace(content=f"standalone: {last}")
        return SimpleNamespace(content="The answer is forty-two [Source 1]")

    @property
    def answer_calls(self):
        return [c for c in self.calls if not c[0].content.startswith("Rewrite")]

    @property
    def rewrite_calls(self):
        return [c for c in self.calls if c[0].content.startswith("Rewrite")]


class CountingEmbeddings(Embeddings):
    """Records how many texts were embedded so tests can prove incremental indexing."""

    def __init__(self):
        self._inner = DeterministicFakeEmbedding(size=16)
        self.embedded_documents = 0
        self.failure: BaseException | None = None

    def embed_documents(self, texts):
        if self.failure:
            raise self.failure
        self.embedded_documents += len(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


class OverlapReranker(BaseCrossEncoder):
    def score(self, text_pairs):
        scores = []
        for query, text in text_pairs:
            q, t = set(query.lower().split()), set(text.lower().split())
            scores.append(float(len(q & t)))
        return scores


class FakeLoader:
    """Treats a file as text; one chunk per blank-line-separated paragraph."""

    def load(self, source: str) -> list[Document]:
        if source.startswith("http"):
            raise DocumentLoadError("cannot fetch in tests")
        with open(source, encoding="utf-8") as handle:
            paragraphs = [p.strip() for p in handle.read().split("\n\n") if p.strip()]
        return [
            Document(page_content=p, metadata={"source": source, "chunk": i})
            for i, p in enumerate(paragraphs)
        ]


class FakeEvaluator:
    def __init__(self):
        self.calls = 0
        self.error: Exception | None = None

    def evaluate(self, question, answer, contexts):
        self.calls += 1
        if self.error:
            raise self.error
        return EvaluationScores(answer_relevance=0.9, faithfulness=0.8)


class FakeTracing:
    def __init__(self):
        self.flushed = 0

    def callbacks(self):
        return []

    def flush(self):
        self.flushed += 1

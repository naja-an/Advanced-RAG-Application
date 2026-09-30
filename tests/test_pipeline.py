from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from app.domain import ChatTurn
from app.exceptions import LLMError, LLMUnavailableError
from app.rag.llm import ResilientLLM
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import NOT_FOUND_MESSAGE
from tests.fakes import BadRequest, FakeChatModel, ServiceUnavailable


class StubRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries: list[str] = []

    async def search(self, query):
        self.queries.append(query)
        return self.documents


def turn(i: int) -> ChatTurn:
    return ChatTurn(i, f"q{i}", f"q{i}", f"a{i}", [], datetime.now(timezone.utc))


def make_pipeline(model: FakeChatModel, **kwargs) -> RAGPipeline:
    llm = ResilientLLM(model, max_attempts=2, timeout_s=1, base_delay_s=0.001, max_delay_s=0.002)
    return RAGPipeline(llm, **kwargs)


async def test_empty_retrieval_answers_not_found_without_calling_the_llm():
    model = FakeChatModel()
    result = await make_pipeline(model).answer("anything", [], StubRetriever([]))
    assert result.answer == NOT_FOUND_MESSAGE and result.documents == []
    assert model.calls == []


async def test_rewrite_uses_only_the_most_recent_turns_and_truncates():
    model = FakeChatModel()
    pipeline = make_pipeline(model, rewrite_history_turns=2, rewrite_max_chars=5)
    history = [
        ChatTurn(i, f"question-{i}-long", "", f"answer-{i}-long", [], datetime.now(timezone.utc))
        for i in range(5)
    ]
    retriever = StubRetriever([Document(page_content="ctx", metadata={})])
    await pipeline.answer("follow up", history, retriever)
    transcript = model.rewrite_calls[0][-1].content
    assert "question-0" not in transcript and "question-2" not in transcript
    assert "quest" in transcript and "question-3-long" not in transcript  # truncated to 5 chars
    assert retriever.queries == ["standalone: follow up"]


async def test_history_messages_are_truncated_for_the_answer_call():
    model = FakeChatModel()
    pipeline = make_pipeline(model, history_max_chars=4)
    retriever = StubRetriever([Document(page_content="ctx", metadata={})])
    await pipeline.answer("next", [turn(1)], retriever)
    prompt = model.answer_calls[0]
    assert prompt[1].content == "q1" and prompt[2].content == "a1"


async def test_llm_error_types():
    model = FakeChatModel()
    llm = ResilientLLM(model, max_attempts=2, timeout_s=1, base_delay_s=0.001, max_delay_s=0.002)

    model.failures = [ServiceUnavailable(), ServiceUnavailable()]
    with pytest.raises(LLMUnavailableError):
        await llm.complete([])
    assert len(model.calls) == 2

    model.calls.clear()
    model.failures = [BadRequest()]
    with pytest.raises(LLMError) as info:
        await llm.complete([])
    assert not isinstance(info.value, LLMUnavailableError) and len(model.calls) == 1


async def test_per_attempt_timeout_is_retried_then_reported_unavailable():
    model = FakeChatModel()
    model.delay_s = 1
    llm = ResilientLLM(model, max_attempts=2, timeout_s=0.02, base_delay_s=0.001, max_delay_s=0.002)
    with pytest.raises(LLMUnavailableError):
        await llm.complete([])
    assert len(model.calls) == 2

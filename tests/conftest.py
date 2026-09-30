import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import ContainerOverrides
from app.main import create_app
from tests.fakes import (
    CountingEmbeddings,
    FakeChatModel,
    FakeEvaluator,
    FakeLoader,
    FakeTracing,
    OverlapReranker,
)

DOC_TEXT = (
    "The capital of France is Paris.\n\n"
    "Paris has a population of about two million people.\n\n"
    "The Eiffel Tower was completed in 1889."
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_path=tmp_path / "app.sqlite3",
        upload_dir=tmp_path / "uploads",
        llm_max_attempts=3,
        llm_timeout_s=2,
        llm_retry_base_delay_s=0.001,
        llm_retry_max_delay_s=0.005,
        chat_timeout_s=5,
        shutdown_grace_s=2,
    )


@pytest.fixture
def fakes():
    return {
        "llm": FakeChatModel(),
        "embeddings": CountingEmbeddings(),
        "reranker": OverlapReranker(),
        "loader": FakeLoader(),
        "evaluator": FakeEvaluator(),
        "tracing": FakeTracing(),
    }


@pytest.fixture
def make_client(settings, fakes):
    """Factory so a test can start several app instances on the same database (restart)."""
    clients: list[TestClient] = []

    def _make() -> TestClient:
        app = create_app(settings, ContainerOverrides(**fakes))
        client = TestClient(app)
        client.__enter__()  # runs the lifespan
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def upload(client: TestClient, session_id: str, text: str = DOC_TEXT, name: str = "doc.txt"):
    return client.post(
        f"/sessions/{session_id}/documents", files=[("files", (name, text.encode(), "text/plain"))]
    )


def new_session(client: TestClient) -> str:
    response = client.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def wait_for_evaluation(
    client: TestClient, session_id: str, evaluation_id: str, timeout: float = 5.0
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/sessions/{session_id}/evaluations/{evaluation_id}").json()
        if body["status"] != "pending":
            return body
        time.sleep(0.02)
    raise AssertionError("evaluation did not finish")

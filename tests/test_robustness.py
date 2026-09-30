import asyncio
import sqlite3

from langchain_core.documents import Document

from app.persistence.database import Database
from app.persistence.turns import TurnRepository
from app.rag.index import IndexRegistry, SessionIndex
from tests.conftest import new_session, upload, wait_for_evaluation
from tests.fakes import CountingEmbeddings, OverlapReranker


def test_everything_survives_a_restart(make_client, fakes):
    first = make_client()
    session_id = new_session(first)
    upload(first, session_id)
    first.post(f"/sessions/{session_id}/chat", json={"question": "capital of France"})
    first.__exit__(None, None, None)

    second = make_client()  # new process, same database
    assert second.get(f"/sessions/{session_id}/documents").json()[0]["status"] == "completed"
    assert len(second.get(f"/sessions/{session_id}/history").json()) == 1

    embedded_before = fakes["embeddings"].embedded_documents
    response = second.post(f"/sessions/{session_id}/chat", json={"question": "and the tower?"})
    assert response.status_code == 200  # index rebuilt from persisted chunks
    assert fakes["embeddings"].embedded_documents > embedded_before
    # history from before the restart is part of the prompt
    assert len(fakes["llm"].answer_calls[-1]) == 4


def test_second_upload_embeds_only_new_chunks(client, fakes):
    session_id = new_session(client)
    upload(client, session_id)  # 3 chunks
    assert fakes["embeddings"].embedded_documents == 3
    upload(client, session_id, text="Berlin is the capital of Germany.", name="b.txt")
    assert fakes["embeddings"].embedded_documents == 4  # +1, not 3 + 4

    answer = client.post(f"/sessions/{session_id}/chat", json={"question": "Berlin Germany capital"})
    contents = " ".join(s["content"] for s in answer.json()["sources"])
    assert "Berlin" in contents and "Paris" in contents  # both uploads are searchable


def test_history_survives_additional_uploads(client):
    session_id = new_session(client)
    upload(client, session_id)
    client.post(f"/sessions/{session_id}/chat", json={"question": "first"})
    upload(client, session_id, text="More text here.", name="more.txt")
    assert len(client.get(f"/sessions/{session_id}/history").json()) == 1


def test_failed_indexing_persists_nothing_and_leaves_session_usable(make_client, fakes):
    client = make_client()
    session_id = new_session(client)
    upload(client, session_id)

    class Boom(Exception):
        code = 503

    fakes["embeddings"].failure = Boom()
    failed = upload(client, session_id, text="Extra paragraph.", name="extra.txt")
    assert failed.status_code == 502
    fakes["embeddings"].failure = None

    assert len(client.get(f"/sessions/{session_id}/documents").json()) == 1  # extra.txt not recorded
    assert client.post(f"/sessions/{session_id}/chat", json={"question": "capital"}).status_code == 200


async def test_concurrent_uploads_are_all_indexed(tmp_path):
    embeddings = CountingEmbeddings()
    index = SessionIndex(embeddings, OverlapReranker(), vector_k=4, lexical_k=4, final_k=3)
    batches = [
        [Document(page_content=f"batch {b} item {i}", metadata={}) for i in range(5)] for b in range(6)
    ]
    await asyncio.gather(*(index.add(b) for b in batches))
    assert index.chunk_count == 30
    assert embeddings.embedded_documents == 30


async def test_registry_serialises_concurrent_first_access(tmp_path):
    loads = 0

    class Source:
        async def load_chunks(self, session_id):
            nonlocal loads
            loads += 1
            await asyncio.sleep(0.02)
            return [Document(page_content="persisted chunk", metadata={})]

    registry = IndexRegistry(
        lambda: SessionIndex(CountingEmbeddings(), OverlapReranker(), vector_k=2, lexical_k=2, final_k=1),
        Source(),
    )
    from uuid import uuid4

    sid = uuid4()
    indexes = await asyncio.gather(*(registry.get(sid) for _ in range(5)))
    assert loads == 1 and all(i is indexes[0] for i in indexes)


def test_history_write_failure_does_not_lose_the_answer(make_client, monkeypatch):
    client = make_client()
    session_id = new_session(client)
    upload(client, session_id)

    async def broken_add(self, *args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(TurnRepository, "add", broken_add)
    response = client.post(f"/sessions/{session_id}/chat", json={"question": "capital"})
    assert response.status_code == 200
    assert response.json()["answer"]


async def test_database_calls_do_not_block_the_event_loop(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    task = asyncio.create_task(ticker())

    def slow_query(conn):
        conn.execute("SELECT count(*) FROM sessions")
        import time

        time.sleep(0.1)  # simulates a slow disk

    await db.run(slow_query)
    task.cancel()
    assert ticks > 20  # the loop kept running while the query was in flight


def test_evaluation_failure_is_recorded_not_lost(client, fakes):
    fakes["evaluator"].error = RuntimeError("judge down")
    session_id = new_session(client)
    upload(client, session_id)
    body = client.post(f"/sessions/{session_id}/chat", json={"question": "capital"}).json()
    result = wait_for_evaluation(client, session_id, body["evaluation_id"])
    assert result["status"] == "failed"
    assert "RuntimeError" in result["evaluation"]["error"]


def test_interrupted_evaluations_are_failed_on_restart(make_client, fakes):
    first = make_client()
    session_id = new_session(first)
    upload(first, session_id)

    import threading

    release = threading.Event()
    original = fakes["evaluator"].evaluate

    def blocked(*args):
        release.wait(timeout=3)
        return original(*args)

    fakes["evaluator"].evaluate = blocked
    body = first.post(f"/sessions/{session_id}/chat", json={"question": "capital"}).json()
    first.app.state.container.settings.shutdown_grace_s = 0  # simulate a hard stop
    first.app.state.container.evaluations._grace_s = 0
    first.__exit__(None, None, None)
    release.set()

    second = make_client()
    status = second.get(f"/sessions/{session_id}/evaluations/{body['evaluation_id']}").json()
    assert status["status"] == "failed"
    assert "restart" in status["evaluation"]["error"]


def test_shutdown_flushes_tracing(make_client, fakes):
    client = make_client()
    client.__exit__(None, None, None)
    assert fakes["tracing"].flushed == 1


def test_expired_sessions_are_not_found(make_client, settings):
    settings.session_ttl_hours = 0.000001  # ~3.6 ms
    client = make_client()
    session_id = new_session(client)
    import time

    time.sleep(0.05)
    assert client.get(f"/sessions/{session_id}/documents").status_code == 404


def test_settings_keep_legacy_env_names(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("RAG_MODEL", "my-model")
    monkeypatch.setenv("RAG_DATABASE_PATH", "/tmp/custom.sqlite3")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    s = Settings()
    assert s.chat_model == "my-model"
    assert str(s.database_path) == "/tmp/custom.sqlite3"
    assert s.tracing_enabled and s.judge_model == "my-model"

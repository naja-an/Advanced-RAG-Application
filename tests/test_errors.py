import pytest

from tests.conftest import new_session, upload
from tests.fakes import BadRequest, ServiceUnavailable


def test_unknown_session_is_404_everywhere(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/sessions/{missing}/documents").status_code == 404
    assert client.get(f"/sessions/{missing}/history").status_code == 404
    assert client.post(f"/sessions/{missing}/chat", json={"question": "x"}).status_code == 404
    assert client.delete(f"/sessions/{missing}").status_code == 404
    assert upload(client, missing).status_code == 404


def test_chat_without_documents_is_400(client):
    session_id = new_session(client)
    response = client.post(f"/sessions/{session_id}/chat", json={"question": "hello"})
    assert response.status_code == 400
    assert "Upload at least one document" in response.json()["detail"]


def test_bad_upload_is_recorded_per_document_and_empty_request_is_400(client):
    session_id = new_session(client)
    response = upload(client, session_id, name="malware.exe")
    assert response.status_code == 202
    assert response.json()[0]["status"] == "failed"
    assert "Unsupported file type" in response.json()[0]["error"]
    assert client.post(f"/sessions/{session_id}/documents").status_code == 400


def test_oversized_upload_fails_that_document_only(client, settings):
    session_id = new_session(client)
    big = "x" * (settings.max_upload_bytes + 10)
    files = [
        ("files", ("big.txt", big.encode(), "text/plain")),
        ("files", ("ok.txt", b"hello world", "text/plain")),
    ]
    body = client.post(f"/sessions/{session_id}/documents", files=files).json()
    assert [d["status"] for d in body] == ["failed", "completed"]
    assert "exceeds" in body[0]["error"]


def test_failed_url_is_reported_not_raised(client):
    session_id = new_session(client)
    body = client.post(f"/sessions/{session_id}/documents", data={"urls": "https://example.com/x"}).json()
    assert body[0]["status"] == "failed"


def test_transient_llm_errors_are_retried_then_succeed(client, fakes):
    session_id = new_session(client)
    upload(client, session_id)
    fakes["llm"].failures = [ServiceUnavailable(), ServiceUnavailable()]
    response = client.post(f"/sessions/{session_id}/chat", json={"question": "capital"})
    assert response.status_code == 200
    assert len(fakes["llm"].calls) == 3


def test_exhausted_retries_map_to_503(client, fakes):
    session_id = new_session(client)
    upload(client, session_id)
    fakes["llm"].failures = [ServiceUnavailable()] * 3
    response = client.post(f"/sessions/{session_id}/chat", json={"question": "capital"})
    assert response.status_code == 503
    assert len(fakes["llm"].calls) == 3


def test_permanent_provider_error_is_502_and_not_retried(client, fakes):
    session_id = new_session(client)
    upload(client, session_id)
    fakes["llm"].failures = [BadRequest()]
    assert client.post(f"/sessions/{session_id}/chat", json={"question": "capital"}).status_code == 502
    assert len(fakes["llm"].calls) == 1


def test_programming_bugs_are_not_disguised_as_provider_errors(client, fakes):
    session_id = new_session(client)
    upload(client, session_id)
    fakes["llm"].failures = [KeyError("bug")]
    # Not converted into LLMError/502: the original exception surfaces (a 500 with traceback in
    # production; the test client re-raises it) and is never retried.
    with pytest.raises(KeyError):
        client.post(f"/sessions/{session_id}/chat", json={"question": "capital"})
    assert len(fakes["llm"].calls) == 1


def test_failed_rewrite_falls_back_to_raw_question(client, fakes):
    session_id = new_session(client)
    upload(client, session_id)
    client.post(f"/sessions/{session_id}/chat", json={"question": "first"})
    fakes["llm"].failures = [BadRequest()]  # the rewrite call fails permanently
    response = client.post(f"/sessions/{session_id}/chat", json={"question": "second"})
    assert response.status_code == 200
    assert response.json()["query_used_for_retrieval"] == "second"


def test_overall_chat_deadline_maps_to_504(make_client, fakes, settings):
    settings.chat_timeout_s = 0.05
    settings.llm_timeout_s = 5
    client = make_client()
    session_id = new_session(client)
    upload(client, session_id)
    fakes["llm"].delay_s = 1
    assert client.post(f"/sessions/{session_id}/chat", json={"question": "capital"}).status_code == 504

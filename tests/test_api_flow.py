from tests.conftest import new_session, upload, wait_for_evaluation


def test_full_conversation_flow(client, fakes):
    session_id = new_session(client)

    uploaded = upload(client, session_id)
    assert uploaded.status_code == 202
    assert uploaded.json()[0]["status"] == "completed"
    assert client.get(f"/sessions/{session_id}/documents").json()[0]["name"] == "doc.txt"

    first = client.post(f"/sessions/{session_id}/chat", json={"question": "capital of France"})
    assert first.status_code == 200
    body = first.json()
    assert body["answer"].startswith("The answer is forty-two")
    assert body["query_used_for_retrieval"] == "capital of France"  # first turn: no rewrite call
    assert body["sources"] and "Paris" in body["sources"][0]["content"]
    assert len(fakes["llm"].rewrite_calls) == 0

    second = client.post(f"/sessions/{session_id}/chat", json={"question": "and its population?"})
    assert second.json()["query_used_for_retrieval"] == "standalone: and its population?"
    assert len(fakes["llm"].rewrite_calls) == 1

    # the final prompt carries the history as plain messages, without old retrieved chunks
    final_prompt = fakes["llm"].answer_calls[-1]
    assert [m.type for m in final_prompt] == ["system", "human", "ai", "human"]
    assert "Eiffel" not in final_prompt[1].content and "Eiffel" not in final_prompt[2].content

    history = client.get(f"/sessions/{session_id}/history").json()
    assert [t["question"] for t in history] == ["capital of France", "and its population?"]

    scores = wait_for_evaluation(client, session_id, body["evaluation_id"])
    assert scores["status"] == "completed"
    assert scores["evaluation"]["faithfulness"] == 0.8

    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert client.get(f"/sessions/{session_id}/history").status_code == 404


def test_history_window_is_bounded(client, fakes, settings):
    session_id = new_session(client)
    upload(client, session_id)
    for i in range(settings.history_max_turns + 3):
        client.post(f"/sessions/{session_id}/chat", json={"question": f"question {i}"})
    final_prompt = fakes["llm"].answer_calls[-1]
    assert len(final_prompt) == 1 + 2 * settings.history_max_turns + 1


def test_delete_removes_uploaded_files_and_index(client, settings):
    session_id = new_session(client)
    upload(client, session_id)
    client.delete(f"/sessions/{session_id}")
    assert not (settings.upload_dir / session_id).exists()
    assert client.post(f"/sessions/{session_id}/chat", json={"question": "hi"}).status_code == 404


def test_frontend_and_health_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ui").status_code == 200

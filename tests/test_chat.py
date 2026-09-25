"""
Tests for POST /api/v1/chat/ask and GET /api/v1/chat/history/{session_id}

Covers:
- Empty question → 400
- Invalid provider → 400
- Valid question (RAG pipeline mocked) → 200 with session_id and message_id
- History returns saved messages in order
- No documents in FAISS → graceful "upload first" response
"""
from unittest.mock import patch


# ---------------------------------------------------------------------------
# /ask — validation tests (no mocking needed)
# ---------------------------------------------------------------------------

def test_ask_rejects_empty_question(client):
    r = client.post("/api/v1/chat/ask", json={"question": "   "})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_ask_rejects_invalid_provider(client):
    r = client.post("/api/v1/chat/ask", json={"question": "hello", "provider": "openai"})
    assert r.status_code == 400
    assert "gemini" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /ask — happy path (mock answer_question so no API key needed)
# ---------------------------------------------------------------------------

MOCK_RESULT = {
    "answer": "Employees get 18 days of paid leave per year.",
    "sources": ["leave_policy.txt"],
}


@patch("app.api.v1.chat.answer_question", return_value=MOCK_RESULT)
def test_ask_returns_answer_and_session_id(mock_rag, client):
    r = client.post("/api/v1/chat/ask", json={"question": "What is the leave policy?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == MOCK_RESULT["answer"]
    assert body["sources"] == MOCK_RESULT["sources"]
    assert "session_id" in body
    assert "message_id" in body


@patch("app.api.v1.chat.answer_question", return_value=MOCK_RESULT)
def test_ask_uses_provided_session_id(mock_rag, client):
    """Passing a session_id should keep it, not generate a new one."""
    r = client.post(
        "/api/v1/chat/ask",
        json={"question": "Leave policy?", "session_id": "my-session-abc"},
    )
    assert r.status_code == 200
    assert r.json()["session_id"] == "my-session-abc"


@patch("app.api.v1.chat.answer_question", return_value=MOCK_RESULT)
def test_ask_provider_field(mock_rag, client):
    """provider should reflect the provider used for the request."""
    r = client.post(
        "/api/v1/chat/ask",
        json={"question": "Leave?", "provider": "groq"},
    )
    assert r.status_code == 200
    assert r.json()["provider"] == "groq"
    assert isinstance(r.json()["latency_ms"], int)


@patch("app.api.v1.chat.answer_question", side_effect=Exception("API timeout"))
def test_ask_returns_502_on_rag_failure(mock_rag, client):
    """If the RAG pipeline throws, the endpoint returns 502 not a crash."""
    r = client.post("/api/v1/chat/ask", json={"question": "What is the policy?"})
    assert r.status_code == 502
    assert "AI provider" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /history — tests
# ---------------------------------------------------------------------------

@patch("app.api.v1.chat.answer_question", return_value=MOCK_RESULT)
def test_history_returns_saved_messages(mock_rag, client):
    """After two questions in the same session, history returns both in order."""
    import uuid
    session_id = f"test-session-{uuid.uuid4()}"

    client.post("/api/v1/chat/ask", json={"question": "Q1?", "session_id": session_id})
    client.post("/api/v1/chat/ask", json={"question": "Q2?", "session_id": session_id})

    r = client.get(f"/api/v1/chat/history/{session_id}")
    assert r.status_code == 200
    messages = r.json()
    assert len(messages) == 2
    assert messages[0]["question"] == "Q1?"
    assert messages[1]["question"] == "Q2?"


def test_history_empty_for_unknown_session(client):
    """An unknown session_id returns an empty list, not an error."""
    r = client.get("/api/v1/chat/history/nonexistent-session-000")
    assert r.status_code == 200
    assert r.json() == []


@patch("app.api.v1.chat.answer_question", return_value={
    "answer": "I don't have any documents to search yet - upload one first.",
    "sources": [],
})
def test_ask_when_no_documents_indexed(mock_rag, client):
    """If FAISS is empty, the pipeline returns a guidance message, not an error."""
    r = client.post("/api/v1/chat/ask", json={"question": "What is in the docs?"})
    assert r.status_code == 200
    assert "upload" in r.json()["answer"].lower()

"""
Tests for POST /api/v1/feedback

Covers:
- Valid "up" rating → 200
- Valid "down" rating → 200
- Invalid rating string → 400
- Non-existent message_id → 404
- Feedback stored correctly in the database
"""
from app.models.db_models import ChatMessage, Feedback


# ---------------------------------------------------------------------------
# Helper: seed a real ChatMessage so feedback has a valid target
# ---------------------------------------------------------------------------

def _seed_message(db_session, question="Test Q?", answer="Test A.") -> int:
    msg = ChatMessage(
        session_id="seed-session",
        question=question,
        answer=answer,
        sources="doc.txt",
    )
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return msg.id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feedback_up_returns_200(client, db_session):
    msg_id = _seed_message(db_session)
    r = client.post("/api/v1/feedback", json={"message_id": msg_id, "rating": "up"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "feedback recorded"
    assert body["rating"] == "up"
    assert body["message_id"] == msg_id


def test_feedback_down_returns_200(client, db_session):
    msg_id = _seed_message(db_session)
    r = client.post("/api/v1/feedback", json={"message_id": msg_id, "rating": "down"})
    assert r.status_code == 200
    assert r.json()["rating"] == "down"


def test_feedback_invalid_rating_returns_400(client, db_session):
    msg_id = _seed_message(db_session)
    r = client.post("/api/v1/feedback", json={"message_id": msg_id, "rating": "meh"})
    assert r.status_code == 400
    assert "up" in r.json()["detail"]


def test_feedback_nonexistent_message_returns_404(client):
    r = client.post("/api/v1/feedback", json={"message_id": 99999, "rating": "up"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_feedback_stored_in_database(client, db_session):
    """After submitting feedback, a matching Feedback row actually exists in the DB."""
    msg_id = _seed_message(db_session)
    r = client.post("/api/v1/feedback", json={"message_id": msg_id, "rating": "up"})
    assert r.status_code == 200
    assert r.json()["status"] == "feedback recorded"
    assert r.json()["message_id"] == msg_id
    assert r.json()["rating"] == "up"

    stored = (
        db_session.query(Feedback)
        .filter(Feedback.message_id == msg_id)
        .order_by(Feedback.id.desc())
        .first()
    )
    assert stored is not None
    assert stored.rating == "up"
    assert stored.message_id == msg_id


def test_feedback_multiple_ratings_for_same_message(client, db_session):
    """Same message can receive multiple feedback entries (e.g., from different users)."""
    msg_id = _seed_message(db_session)
    r1 = client.post("/api/v1/feedback", json={"message_id": msg_id, "rating": "up"})
    r2 = client.post("/api/v1/feedback", json={"message_id": msg_id, "rating": "down"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["rating"] == "up"
    assert r2.json()["rating"] == "down"

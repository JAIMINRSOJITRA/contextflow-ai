"""
Tests for POST /api/v1/evaluate

These tests mock the Gemini evaluator so no API key is needed.
They check:
  - Input validation (empty question, empty answer, empty contexts)
  - Happy path: all 4 metric scores returned with correct shape
  - 502 when the underlying Gemini call fails
"""
from unittest.mock import patch

MOCK_EVAL_RESULT = {
    "faithfulness": 0.95,
    "answer_relevancy": 0.90,
    "context_precision": 0.85,
    "context_recall": 0.88,
    "scores_detail": {
        "faithfulness_reason": "Answer is well grounded.",
        "answer_relevancy_reason": "Directly answers the question.",
        "context_precision_reason": "All chunks are relevant.",
        "context_recall_reason": "Chunks fully support the answer.",
    },
}

VALID_PAYLOAD = {
    "question": "What is the leave policy?",
    "answer": "Employees get 18 days of paid leave per year.",
    "contexts": ["The company provides 18 days paid leave per year to all employees."],
}


# ---------------------------------------------------------------------------
# Validation tests — fire before Gemini is called
# ---------------------------------------------------------------------------

def test_evaluate_rejects_empty_question(client):
    payload = {**VALID_PAYLOAD, "question": "   "}
    r = client.post("/api/v1/evaluate", json=payload)
    assert r.status_code == 400
    assert "question" in r.json()["detail"]


def test_evaluate_rejects_empty_answer(client):
    payload = {**VALID_PAYLOAD, "answer": ""}
    r = client.post("/api/v1/evaluate", json=payload)
    assert r.status_code == 400
    assert "answer" in r.json()["detail"]


def test_evaluate_rejects_empty_contexts(client):
    payload = {**VALID_PAYLOAD, "contexts": ["   "]}
    r = client.post("/api/v1/evaluate", json=payload)
    assert r.status_code == 400
    assert "context" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@patch("app.api.v1.evaluate.evaluate_sample", return_value=MOCK_EVAL_RESULT)
def test_evaluate_returns_all_four_metrics(mock_eval, client):
    r = client.post("/api/v1/evaluate", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "faithfulness" in body
    assert "answer_relevancy" in body
    assert "context_precision" in body
    assert "context_recall" in body
    assert "scores_detail" in body


@patch("app.api.v1.evaluate.evaluate_sample", return_value=MOCK_EVAL_RESULT)
def test_evaluate_scores_are_floats_between_0_and_1(mock_eval, client):
    r = client.post("/api/v1/evaluate", json=VALID_PAYLOAD)
    body = r.json()
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert 0.0 <= body[metric] <= 1.0, f"{metric} out of range"


@patch("app.api.v1.evaluate.evaluate_sample", return_value=MOCK_EVAL_RESULT)
def test_evaluate_scores_detail_has_reasons(mock_eval, client):
    r = client.post("/api/v1/evaluate", json=VALID_PAYLOAD)
    detail = r.json()["scores_detail"]
    assert "faithfulness_reason" in detail
    assert "answer_relevancy_reason" in detail
    assert "context_precision_reason" in detail
    assert "context_recall_reason" in detail


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

@patch("app.api.v1.evaluate.evaluate_sample", side_effect=Exception("LLM judge timeout"))
def test_evaluate_returns_502_on_llm_judge_failure(mock_eval, client):
    r = client.post("/api/v1/evaluate", json=VALID_PAYLOAD)
    assert r.status_code == 502
    assert "Evaluation failed" in r.json()["detail"]

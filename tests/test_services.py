"""
Tests for service-layer functions that can run without API keys.
These are unit tests — no HTTP layer, no real database, no external calls.

Covers:
- chunk_text: size, overlap, empty input, short input
- vector_store: add + search mechanics using fake vectors
- health check endpoint
"""
import numpy as np
from app.services.chunker import chunk_text


# ---------------------------------------------------------------------------
# chunker tests
# ---------------------------------------------------------------------------

def test_chunk_text_single_chunk_for_short_text():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_chunk_text_multiple_chunks():
    # With size=500, overlap=50 → stride=450
    # 1000 chars: chunk0=0-500, chunk1=450-950, chunk2=900-1000 → 3 chunks
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 3


def test_chunk_text_overlap_is_present():
    """Last 50 chars of chunk 0 should appear at the start of chunk 1."""
    text = "A" * 450 + "B" * 100
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    # The overlap region: last 50 chars of chunk[0] == first 50 chars of chunk[1]
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_empty_string():
    chunks = chunk_text("", chunk_size=500, chunk_overlap=50)
    assert chunks == []


def test_chunk_text_exact_size():
    text = "X" * 500
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
    assert len(chunks) == 1
    assert len(chunks[0]) == 500


# ---------------------------------------------------------------------------
# vector_store tests (fake vectors, no Gemini key needed)
# ---------------------------------------------------------------------------

def test_vector_store_add_and_search(tmp_path, monkeypatch):
    """
    Use monkeypatch to redirect FAISS index files to a temp directory
    so tests never touch the real data/vector_index on disk.
    """
    import app.services.vector_store as vs

    monkeypatch.setattr(vs, "VECTOR_INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(vs, "METADATA_PATH", str(tmp_path / "metadata.pkl"))

    dim = 64
    fake_embeddings = [np.random.rand(dim).tolist() for _ in range(3)]
    chunks = ["chunk alpha", "chunk beta", "chunk gamma"]

    vs.add_chunks(chunks, fake_embeddings, "test_doc.txt")
    results = vs.search(fake_embeddings[0], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "chunk alpha"
    assert results[0]["source"] == "test_doc.txt"


def test_vector_store_search_returns_empty_when_no_index(tmp_path, monkeypatch):
    import app.services.vector_store as vs

    monkeypatch.setattr(vs, "VECTOR_INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(vs, "METADATA_PATH", str(tmp_path / "metadata.pkl"))
    
    # Clear cache before test
    vs._invalidate_cache()
    
    results = vs.search([0.1] * 3072, top_k=4)
    assert results == []


def test_vector_store_top_k_limits_results(tmp_path, monkeypatch):
    import app.services.vector_store as vs

    monkeypatch.setattr(vs, "VECTOR_INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(vs, "METADATA_PATH", str(tmp_path / "metadata.pkl"))

    dim = 64
    embeddings = [np.random.rand(dim).tolist() for _ in range(5)]
    chunks = [f"chunk {i}" for i in range(5)]
    vs.add_chunks(chunks, embeddings, "doc.txt")

    results = vs.search(embeddings[0], top_k=2)
    assert len(results) == 2


def test_vector_store_removes_only_requested_document(tmp_path, monkeypatch):
    import app.services.vector_store as vs

    monkeypatch.setattr(vs, "VECTOR_INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(vs, "METADATA_PATH", str(tmp_path / "metadata.pkl"))
    
    # Clear cache before test
    vs._invalidate_cache()
    
    vs.add_chunks(["alpha"], [[1.0, 0.0]], "alpha.txt", document_id="alpha")
    vs.add_chunks(["beta"], [[0.0, 1.0]], "beta.txt", document_id="beta")

    assert vs.remove_document_chunks("alpha") == 1
    results = vs.search([0.0, 1.0], top_k=1)
    assert results == [{"text": "beta", "source": "beta.txt", "document_id": "beta"}]


def test_rag_returns_upload_guidance_without_embedding_when_index_is_empty(monkeypatch):
    import app.services.rag_pipeline as pipeline

    monkeypatch.setattr(pipeline, "has_chunks", lambda: False)
    monkeypatch.setattr(
        pipeline,
        "embed_text",
        lambda _: (_ for _ in ()).throw(AssertionError("embedding should not be called")),
    )

    result = pipeline.answer_question("What is the policy?")
    assert "upload" in result["answer"].lower()


def test_sentence_transformers_embedding_fallback(monkeypatch):
    import app.services.embeddings as embedding_service

    class FakeModel:
        def encode(self, values, normalize_embeddings):
            assert normalize_embeddings is True
            if isinstance(values, str):
                return np.array([0.25, 0.75])
            return np.array([[0.25, 0.75] for _ in values])

    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setattr(embedding_service, "_local_embedding_model", lambda: FakeModel())

    assert embedding_service.embed_text("hello") == [0.25, 0.75]
    assert embedding_service.embed_chunks(["first", "second"]) == [
        [0.25, 0.75],
        [0.25, 0.75],
    ]


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "ContextFlow AI"}

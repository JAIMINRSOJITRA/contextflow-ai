"""
ContextFlow AI — Streamlit Frontend
Replaces the HTML/JS/CSS Web UI with a Streamlit app for free HuggingFace Spaces deployment.
All backend services (RAG pipeline, FAISS, LLM) are called directly — no FastAPI layer needed.
"""

import os
import sys
import uuid
import tempfile
import json
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="ContextFlow AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Dark CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Dark glassmorphism theme ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.3);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    margin-left: 20%;
    color: white;
    font-size: 14px;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.ai-bubble {
    background: rgba(30, 30, 60, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    margin-right: 10%;
    color: #e2e8f0;
    font-size: 14px;
    backdrop-filter: blur(10px);
}
.source-tag {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11px;
    color: #a5b4fc;
    margin: 2px;
}
.metric-card {
    background: rgba(30, 30, 60, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-score {
    font-size: 32px;
    font-weight: bold;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.doc-card {
    background: rgba(30, 30, 60, 0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Bootstrap: ensure project root is on sys.path ───────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Lazy-load backend services ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ContextFlow AI backend…")
def load_backend():
    """Initialize DB and pre-load embedding model once per session."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    from app.db.database import initialize_database
    
    # Ensure database is initialized (creates tables if they don't exist)
    try:
        initialize_database()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        raise
    
    # pre-warm embedding model
    try:
        from app.services.embeddings import embed_text
        embed_text("warmup")
    except Exception as e:
        # Non-critical - embeddings will load on first use
        pass

    return True


load_backend()

# ── Import services after backend is ready ───────────────────────────────────
from app.services.document_processor import extract_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_chunks
from app.services.vector_store import add_chunks, remove_document_chunks, has_chunks
from app.services.rag_pipeline import answer_question
from app.db.database import get_db
from app.models.db_models import Document, ChatMessage, Feedback
from app.core.config import DEFAULT_LLM_PROVIDER


# ── Session state defaults ───────────────────────────────────────────────────
if "session_id"      not in st.session_state: st.session_state.session_id      = str(uuid.uuid4())
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "last_message_id" not in st.session_state: st.session_state.last_message_id = None
if "provider"        not in st.session_state: st.session_state.provider        = DEFAULT_LLM_PROVIDER


# ── Helper utilities ─────────────────────────────────────────────────────────
def get_documents():
    """Get all documents from database with error handling."""
    db = next(get_db())
    try:
        return db.query(Document).order_by(Document.uploaded_at.desc()).all()
    except Exception as e:
        # If table doesn't exist or query fails, return empty list
        # This handles first-run scenarios on Streamlit Cloud
        return []
    finally:
        db.close()


def save_message(question: str, answer: str, sources: list[str]) -> int | None:
    db = next(get_db())
    try:
        msg = ChatMessage(
            session_id=st.session_state.session_id,
            question=question,
            answer=answer,
            sources=json.dumps(sources),
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg.id
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()


def save_feedback(message_id: int, rating: str):
    db = next(get_db())
    try:
        fb = Feedback(message_id=message_id, rating=rating)
        db.add(fb)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def get_all_sessions():
    """Get all chat sessions with error handling."""
    db = next(get_db())
    try:
        from sqlalchemy import func
        rows = (
            db.query(
                ChatMessage.session_id,
                func.max(ChatMessage.created_at).label("latest"),
                func.count(ChatMessage.id).label("count"),
            )
            .group_by(ChatMessage.session_id)
            .order_by(func.max(ChatMessage.created_at).desc())
            .all()
        )
        results = []
        for row in rows:
            last = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == row.session_id)
                .order_by(ChatMessage.created_at.desc())
                .first()
            )
            results.append({
                "session_id": row.session_id,
                "last_question": last.question if last else "Session",
                "count": row.count,
                "updated_at": row.latest,
            })
        return results
    except Exception:
        # Return empty list if database query fails
        return []
    finally:
        db.close()


def get_session_messages(session_id: str):
    db = next(get_db())
    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .all()
        )
    finally:
        db.close()


def delete_session(session_id: str):
    db = next(get_db())
    try:
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 ContextFlow AI")
    st.caption("Enterprise RAG Document Intelligence")
    st.divider()

    # Provider selector
    st.session_state.provider = st.selectbox(
        "🤖 LLM Provider",
        ["groq", "gemini"],
        index=0 if st.session_state.provider == "groq" else 1,
    )

    st.divider()

    # ── Document Upload ──────────────────────────────────────────────────────
    st.markdown("### 📁 Documents")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_file and st.button("⬆️ Upload & Index", use_container_width=True):
        with st.spinner("Processing document…"):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                text = extract_text(tmp_path)
                if not text.strip():
                    st.error("No readable text found in this document.")
                else:
                    chunks     = chunk_text(text)
                    embeddings = embed_chunks(chunks)
                    source_id  = uuid.uuid4().hex
                    add_chunks(chunks, embeddings, uploaded_file.name, document_id=source_id)

                    db = next(get_db())
                    try:
                        doc = Document(
                            filename=uploaded_file.name,
                            storage_filename=f"{source_id}_{uploaded_file.name}",
                            source_id=source_id,
                        )
                        db.add(doc)
                        db.commit()
                    finally:
                        db.close()

                    st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")
                    st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")
            finally:
                os.unlink(tmp_path)

    # ── Indexed Documents List ───────────────────────────────────────────────
    docs = get_documents()
    if docs:
        st.markdown(f"**{len(docs)} document(s) indexed:**")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f'<div class="doc-card">📄 {doc.filename}</div>', unsafe_allow_html=True)
            if col2.button("🗑️", key=f"del_{doc.id}", help="Delete document"):
                remove_document_chunks(doc.source_id or doc.filename)
                db = next(get_db())
                try:
                    db.delete(db.get(Document, doc.id))
                    db.commit()
                finally:
                    db.close()
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    st.divider()

    # ── Chat Sessions ────────────────────────────────────────────────────────
    st.markdown("### 💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id   = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.last_message_id = None
        st.rerun()

    sessions = get_all_sessions()
    for sess in sessions[:10]:
        preview = sess["last_question"][:28] + "…" if len(sess["last_question"]) > 28 else sess["last_question"]
        is_current = sess["session_id"] == st.session_state.session_id
        label = f"{'▶ ' if is_current else ''}{preview}"
        col1, col2 = st.columns([4, 1])
        if col1.button(label, key=f"sess_{sess['session_id']}", use_container_width=True):
            msgs = get_session_messages(sess["session_id"])
            st.session_state.session_id   = sess["session_id"]
            st.session_state.chat_history = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": m.question if i % 2 == 0 else m.answer,
                 "sources": json.loads(m.sources) if m.sources else [],
                 "id": m.id}
                for m in msgs
                for i in [0, 1]
            ]
            st.rerun()
        if col2.button("✕", key=f"delsess_{sess['session_id']}"):
            delete_session(sess["session_id"])
            if sess["session_id"] == st.session_state.session_id:
                st.session_state.session_id   = str(uuid.uuid4())
                st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — TABS
# ════════════════════════════════════════════════════════════════════════════
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluate"])

# ── TAB 1: CHAT ──────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("## 💬 Ask Your Documents")

    if not has_chunks():
        st.info("👆 Upload a document in the sidebar to get started!")

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-bubble">🧠 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                src_html = " ".join(f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"])
                st.markdown(src_html, unsafe_allow_html=True)

            # Feedback buttons
            msg_id = msg.get("id")
            if msg_id:
                col1, col2, col3 = st.columns([1, 1, 8])
                if col1.button("👍", key=f"up_{msg_id}"):
                    save_feedback(msg_id, "up")
                    st.toast("Thanks for your feedback!")
                if col2.button("👎", key=f"dn_{msg_id}"):
                    save_feedback(msg_id, "down")
                    st.toast("Thanks for your feedback!")

    # Quick prompt chips
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        chips = [
            "Summarize the key points of this document",
            "What are the main findings?",
            "List all important dates mentioned",
            "What action items are recommended?",
        ]
        cols = st.columns(2)
        for i, chip in enumerate(chips):
            if cols[i % 2].button(chip, key=f"chip_{i}", use_container_width=True):
                st.session_state._quick_prompt = chip
                st.rerun()

    # Chat input
    question = st.chat_input("Ask a question about your documents…")
    if not question and hasattr(st.session_state, "_quick_prompt"):
        question = st.session_state._quick_prompt
        del st.session_state._quick_prompt

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🧠 Thinking…"):
            try:
                result = answer_question(question, provider=st.session_state.provider)
                answer  = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer  = f"⚠️ Error: {str(e)}"
                sources = []

        msg_id = save_message(question, answer, sources)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "id": msg_id,
        })
        st.session_state.last_message_id = msg_id
        st.rerun()


# ── TAB 2: EVALUATION ────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("## 📊 RAG Quality Evaluation")
    st.caption("Evaluate answer quality using the Ragas framework (Groq LLM Judge)")

    with st.form("eval_form"):
        eval_question = st.text_area("Question", placeholder="What is the refund policy?")
        eval_answer   = st.text_area("Answer",   placeholder="The answer generated by ContextFlow AI…")
        eval_context  = st.text_area("Context (retrieved passages, one per line)",
                                      placeholder="Paste retrieved document chunks here…")
        eval_reference = st.text_input("Reference Answer (optional)", placeholder="Leave blank to use generated answer")
        submitted = st.form_submit_button("🚀 Run Ragas Evaluation", use_container_width=True)

    if submitted:
        if not eval_question.strip() or not eval_answer.strip() or not eval_context.strip():
            st.error("Question, Answer, and Context are all required.")
        else:
            with st.spinner("Running Ragas evaluation… this may take 30–60 seconds"):
                try:
                    from app.evaluation.evaluate_rag import evaluate_sample
                    contexts = [c.strip() for c in eval_context.split("\n") if c.strip()]
                    scores = evaluate_sample(
                        question=eval_question,
                        answer=eval_answer,
                        contexts=contexts,
                        reference=eval_reference.strip() or None,
                    )

                    st.success("✅ Evaluation complete!")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = [
                        ("Faithfulness",      scores.get("faithfulness", 0),      col1),
                        ("Answer Relevancy",  scores.get("answer_relevancy", 0),  col2),
                        ("Context Precision", scores.get("context_precision", 0), col3),
                        ("Context Recall",    scores.get("context_recall", 0),    col4),
                    ]
                    for name, score, col in metrics:
                        pct = int(score * 100)
                        color = "#10b981" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
                        col.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:12px; color:#94a3b8;">{name}</div>
                            <div class="metric-score" style="-webkit-text-fill-color:{color};">{pct}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("📋 Detailed Reasoning"):
                        for key, val in scores.get("scores_detail", {}).items():
                            st.markdown(f"**{key.replace('_', ' ').title()}**: {val}")

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

"""All application settings — loaded from the .env file at import time.

Every configurable value lives here. No other file reads os.getenv()
or hardcodes numeric constants. Changing a value means editing .env,
not hunting through source code.

If a required setting is missing or invalid, this module raises
immediately at startup — fail fast, clear error messages.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# override=False (the default): real environment variables — e.g. secrets
# injected by HuggingFace Spaces, Docker, or any deployment platform — must
# win over a committed/local .env file, not get silently overwritten by it.
load_dotenv()


def _int_setting(name: str, default: int, minimum: int = 0) -> int:
    """Read an integer from the environment and fail early with a clear message."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}.")
    return value


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").strip().lower()
if DEFAULT_LLM_PROVIDER not in {"gemini", "groq"}:
    raise ValueError("DEFAULT_LLM_PROVIDER must be 'gemini' or 'groq'.")

# ---------------------------------------------------------------------------
# Database and file storage
# ---------------------------------------------------------------------------
DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///./app/db/contextflow.db")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "data/uploads")

# VECTOR_INDEX_PATH can be set to either a directory or the .faiss file itself.
_raw_vector_path = Path(os.getenv("VECTOR_INDEX_PATH", "data/vector_index/index.faiss"))
VECTOR_INDEX_PATH = str(
    _raw_vector_path / "index.faiss" if not _raw_vector_path.suffix else _raw_vector_path
)
METADATA_PATH = str(Path(VECTOR_INDEX_PATH).with_name("metadata.pkl"))

# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------
# Defaults to the free local model so the app works with zero API keys out
# of the box (matches the "free HuggingFace Spaces deployment" goal stated
# in app.py). Set EMBEDDING_PROVIDER=gemini in .env to use Gemini embeddings
# instead, which requires GEMINI_API_KEY.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").strip().lower()
if EMBEDDING_PROVIDER not in {"gemini", "sentence-transformers"}:
    raise ValueError("EMBEDDING_PROVIDER must be 'gemini' or 'sentence-transformers'.")

# Only used when EMBEDDING_PROVIDER=sentence-transformers
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------
CHUNK_SIZE    = _int_setting("CHUNK_SIZE",    500, minimum=1)
CHUNK_OVERLAP = _int_setting("CHUNK_OVERLAP",  50, minimum=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

TOP_K_RESULTS = _int_setting("TOP_K_RESULTS", 4, minimum=1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")

# ---------------------------------------------------------------------------
# Advanced RAG tuning
# ---------------------------------------------------------------------------
# Hybrid Search: How many extra candidates to retrieve before RRF fusion (multiplier)
HYBRID_SEARCH_MULTIPLIER = _int_setting("HYBRID_SEARCH_MULTIPLIER", 3, minimum=1)

# RRF (Reciprocal Rank Fusion) constant - higher values give more weight to top-ranked items
# Standard value is 60 based on research (Cormack et al., 2009)
RRF_K = _int_setting("RRF_K", 60, minimum=1)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

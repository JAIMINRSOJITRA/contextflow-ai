"""POST /api/v1/documents/upload — ingest a document into the knowledge base.
GET  /api/v1/documents         — list all uploaded documents.
DELETE /api/v1/documents/{id}  — remove a document and its indexed chunks.

Upload pipeline (in order):
  1. Validate file type and save to disk.
  2. Extract text (plain read for .txt/.docx, PyMuPDF + Gemini OCR for .pdf).
  3. Split into overlapping chunks.
  4. Embed each chunk via the configured embedding provider.
  5. Store vectors in the FAISS index.
  6. Write a record to the documents table.

If any step fails, we clean up what was already written before returning
an error — no half-indexed documents left behind.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_FOLDER
from app.core.logging_config import get_logger
from app.db.database import get_db
from app.models.db_models import Document
from app.schemas.document_schema import (
    DocumentDeleteResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from app.services.chunker import chunk_text
from app.services.document_processor import extract_text
from app.services.embeddings import embed_chunks
from app.services.vector_store import add_chunks, remove_document_chunks

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE_MB = 50  # Maximum file size in megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def _safe_filename(filename: str | None) -> str:
    """Strip directory components from the filename to prevent path traversal."""
    normalized = (filename or "").replace("\\", "/")
    return Path(normalized).name.strip()


def _remove_file(path: Path) -> bool:
    """Delete a file from disk, logging a warning if it can't be removed."""
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("Could not delete uploaded file '%s': %s", path, exc)
        return False


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Accept a file upload, run it through the full ingestion pipeline, and return a summary."""
    filename  = _safe_filename(file.filename)
    extension = Path(filename).suffix.lower()

    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required.")
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Check file size before processing
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB, but file is {file_size / (1024 * 1024):.1f}MB.",
        )

    # Use a UUID prefix so two uploads of the same filename don't overwrite each other.
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    source_id        = uuid.uuid4().hex
    storage_filename = f"{source_id}_{filename}"
    file_path        = Path(UPLOAD_FOLDER) / storage_filename

    try:
        with file_path.open("wb") as f:
            f.write(file.file.read())
    except OSError as exc:
        logger.error("Failed to save uploaded file '%s': %s", filename, exc)
        raise HTTPException(status_code=500, detail="Could not save the uploaded file.") from exc

    # --- Step 2: Extract text ---
    try:
        document_text = extract_text(str(file_path))
    except ValueError as exc:
        _remove_file(file_path)
        status_code = 503 if "API_KEY" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        _remove_file(file_path)
        logger.exception("Text extraction failed for '%s'", filename)
        raise HTTPException(
            status_code=502,
            detail="Failed to extract text from the document. The AI provider may be unavailable.",
        ) from exc

    if not document_text.strip():
        _remove_file(file_path)
        raise HTTPException(status_code=400, detail="No readable text found in this document.")

    # --- Steps 3–5: Chunk → Embed → Index ---
    try:
        chunks     = chunk_text(document_text)
        embeddings = embed_chunks(chunks)
        add_chunks(chunks, embeddings, filename, document_id=source_id)
    except ValueError as exc:
        _remove_file(file_path)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        _remove_file(file_path)
        logger.exception("Chunking or embedding failed for '%s'", filename)
        raise HTTPException(
            status_code=502,
            detail="Failed to process and index the document. Please try again.",
        ) from exc

    # --- Step 6: Save to database ---
    new_document = Document(
        filename=filename,
        storage_filename=storage_filename,
        source_id=source_id,
    )
    try:
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
    except Exception as exc:
        db.rollback()
        remove_document_chunks(source_id)
        _remove_file(file_path)
        logger.exception("Database write failed for '%s'", filename)
        raise HTTPException(
            status_code=500,
            detail="Document was indexed but could not be saved to the database.",
        ) from exc

    return {
        "id":             new_document.id,
        "filename":       new_document.filename,
        "uploaded_at":    new_document.uploaded_at,
        "chunks_created": len(chunks),
        "status":         "uploaded and indexed",
    }


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)):
    """Return all uploaded documents, newest first."""
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Remove a document's database record, uploaded file, and FAISS index entries."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    source_id        = document.source_id or document.filename
    storage_filename = document.storage_filename or document.filename
    file_path        = Path(UPLOAD_FOLDER) / storage_filename

    # Remove vectors first — if this fails we don't touch the DB record.
    try:
        chunks_removed = remove_document_chunks(source_id)
    except Exception as exc:
        logger.exception("Could not remove indexed chunks for document %s", document_id)
        raise HTTPException(
            status_code=500, detail="Could not remove the document from the vector index."
        ) from exc

    file_deleted = _remove_file(file_path)

    try:
        db.delete(document)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Could not delete database record for document %s", document_id)
        raise HTTPException(status_code=500, detail="Could not delete the document record.") from exc

    return {
        "id":             document_id,
        "filename":       document.filename,
        "chunks_removed": chunks_removed,
        "file_deleted":   file_deleted,
        "status":         "deleted",
    }

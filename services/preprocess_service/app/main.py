import os
import hashlib
from datetime import datetime, timezone
from typing import Optional, List

import uuid

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .security import require_auth
from .ocr import extract_text_from_upload
from .ner import ner_entities, build_event_hints_from_text, clean_ocr_text
from .schemas import PreprocessResponse, NEREntity, ExtractedEvent

APP_NAME = "cse400-preprocess-service"

app = FastAPI(
    title="CSE400 Preprocess Service (Stage 1)",
    version="0.1.0",
    description="Upload medical documents → OCR + NER (best-effort) → structured extraction.",
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "25000000"))  # 25MB default

RETURN_OCR_TEXT = os.getenv("RETURN_OCR_TEXT", "false").strip().lower() in ("1","true","yes")


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}


@app.post("/v1/preprocess", response_model=PreprocessResponse)
async def preprocess(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(default=None),
    document_date: Optional[str] = Form(default=None),
    _: None = Depends(require_auth),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_BYTES} bytes.")

    sha256 = hashlib.sha256(data).hexdigest()

    # Parse document date if provided
    doc_dt = None
    if document_date:
        try:
            # Accept YYYY-MM-DD or full ISO
            doc_dt = datetime.fromisoformat(document_date.replace("Z", "+00:00"))
            if doc_dt.tzinfo is None:
                doc_dt = doc_dt.replace(tzinfo=timezone.utc)
        except Exception:
            doc_dt = None

    if doc_dt is None:
        doc_dt = datetime.now(tz=timezone.utc)

    # OCR / text extraction
    raw_text = extract_text_from_upload(
        data=data,
        filename=file.filename or "",
        content_type=file.content_type or "",
    )

    # Clean extracted text to reduce boilerplate-driven false positives.
    text = clean_ocr_text(raw_text)

    # NER (best effort)
    entities: List[NEREntity] = ner_entities(text)

    # Event hints (optional and intentionally conservative)
    # These are "hints" to Stage 2; in production you should map to SNOMED/RxNorm/LOINC.
    events: List[ExtractedEvent] = build_event_hints_from_text(text=text, event_dt=doc_dt)

    # IMPORTANT: do NOT log text or entities (PHI). Log only safe metadata.
    return PreprocessResponse(
        document_sha256=sha256,
        filename=file.filename,
        content_type=file.content_type,
        patient_id=patient_id,
        extracted_at_utc=datetime.now(tz=timezone.utc),
        text=(text if RETURN_OCR_TEXT else ""),
        entities=entities,
        events=events,
    )

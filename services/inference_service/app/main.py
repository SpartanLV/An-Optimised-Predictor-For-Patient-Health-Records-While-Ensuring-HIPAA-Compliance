import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

import uuid

from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel

from .security import require_api_key
from .schemas import PredictRequest, PredictResponse
from .model_runtime import ModelBundle, fetch_events_from_store

APP_NAME = "cse400-inference-service"

MODEL_BUNDLE_DIR = Path(os.getenv("MODEL_BUNDLE_DIR", "./model_bundle")).resolve()
PATIENT_STORE_URL = os.getenv("PATIENT_STORE_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()

bundle: ModelBundle | None = None

app = FastAPI(
    title="CSE400 Inference Service (Stage 3)",
    version="0.1.0",
    description="Hybrid inference: Deep CNN-BiLSTM + HMM features → fusion MLP → predictions + recommendations.",
)

@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response



@app.on_event("startup")
def _startup():
    global bundle
    if not MODEL_BUNDLE_DIR.exists():
        raise RuntimeError(f"MODEL_BUNDLE_DIR not found: {MODEL_BUNDLE_DIR}")
    bundle = ModelBundle(MODEL_BUNDLE_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME, "model_bundle_dir": str(MODEL_BUNDLE_DIR)}


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(request: Request, req: PredictRequest, _: None = Depends(require_api_key)):
    global bundle
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded.")

    events = None
    if req.events:
        events = [e.model_dump() for e in req.events]
    elif req.patient_id:
        if not PATIENT_STORE_URL:
            raise HTTPException(status_code=500, detail="PATIENT_STORE_URL not configured.")
        if not API_KEY:
            raise HTTPException(status_code=500, detail="API_KEY not configured for store fetch.")
        try:
            events = await fetch_events_from_store(PATIENT_STORE_URL, API_KEY, req.patient_id, request_id=getattr(request.state, "request_id", None))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch events from store: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either patient_id or events.")

    as_of = req.as_of or datetime.now(tz=timezone.utc)

    result = bundle.predict_from_events(
        events=events,
        as_of=as_of,
        top_n=req.top_n,
        return_debug=req.return_debug,
    )
    return result

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import uuid

from fastapi import FastAPI, Depends, HTTPException, Request

from .security import require_api_key
from .schemas import PredictRequest, PredictResponse, TelemetryPoint
from .model_runtime import ModelBundle, fetch_events_from_store

APP_NAME = "cse400-inference-service"

MODEL_BUNDLE_DIR = Path(os.getenv("MODEL_BUNDLE_DIR", "./model_bundle")).resolve()
PATIENT_STORE_URL = os.getenv("PATIENT_STORE_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()

bundle: ModelBundle | None = None
telemetry_log: List[TelemetryPoint] = []

app = FastAPI(
    title="CSE400 Inference Service (Stage 3)",
    version="0.2.0",
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
    return {"status": "ok", "service": APP_NAME, "model_bundle_dir": str(MODEL_BUNDLE_DIR), "telemetry_points": len(telemetry_log)}


def _build_explanations(events: List[Dict[str, Any]], top_n: int = 4) -> List[str]:
    scored: Dict[str, int] = {}
    for e in events[-80:]:
        desc = (e.get("description") or e.get("code") or e.get("source") or "").strip()
        if not desc:
            continue
        k = desc[:80]
        scored[k] = scored.get(k, 0) + 1
    return [k for k, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]


def _record_telemetry(patient_id: str | None, result: Dict[str, Any]) -> None:
    preds = result.get("predictions") or []
    top_prob = float(preds[0].get("probability", 0.0)) if preds else 0.0
    model_version = None
    try:
        model_version = str((bundle.metadata if bundle else {}).get("model_version") or "")
    except Exception:
        model_version = None
    telemetry_log.append(TelemetryPoint(
        ts_utc=datetime.now(tz=timezone.utc),
        patient_id=patient_id,
        threshold=float(result.get("threshold") or 0.0),
        top_probability=top_prob,
        prediction_count=len(preds),
        model_version=model_version,
    ))
    if len(telemetry_log) > 3000:
        del telemetry_log[:1000]


def _patient_timeline(patient_id: str) -> List[Dict[str, Any]]:
    pts = [p for p in telemetry_log if p.patient_id == patient_id]
    pts = sorted(pts, key=lambda x: x.ts_utc)[-90:]
    trend = "stable"
    if len(pts) >= 2:
        delta = pts[-1].top_probability - pts[0].top_probability
        if delta >= 0.10:
            trend = "rising"
        elif delta <= -0.10:
            trend = "falling"
    return [{"ts_utc": p.ts_utc.isoformat(), "top_probability": p.top_probability, "prediction_count": p.prediction_count, "trend": trend} for p in pts]


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

    explain = _build_explanations(events)
    for p in result.get("predictions") or []:
        p["explanation"] = explain

    _record_telemetry(req.patient_id, result)
    if req.patient_id:
        result["timeline"] = _patient_timeline(req.patient_id)
    return result


@app.get("/v1/inference/metrics", response_model=Dict[str, Any])
def inference_metrics(_: None = Depends(require_api_key)):
    now = datetime.now(tz=timezone.utc)
    last_24h = [p for p in telemetry_log if p.ts_utc >= now - timedelta(hours=24)]
    avg_top = sum(p.top_probability for p in last_24h) / len(last_24h) if last_24h else 0.0
    return {
        "total_requests": len(telemetry_log),
        "last_24h_requests": len(last_24h),
        "avg_top_probability_24h": avg_top,
    }


@app.get("/v1/inference/drift", response_model=Dict[str, Any])
def inference_drift(_: None = Depends(require_api_key)):
    now = datetime.now(tz=timezone.utc)
    recent = [p.top_probability for p in telemetry_log if p.ts_utc >= now - timedelta(days=7)]
    baseline = [p.top_probability for p in telemetry_log if now - timedelta(days=30) <= p.ts_utc < now - timedelta(days=7)]
    if not recent or not baseline:
        return {"status": "insufficient_data", "recent_points": len(recent), "baseline_points": len(baseline)}
    recent_avg = sum(recent) / len(recent)
    baseline_avg = sum(baseline) / len(baseline)
    shift = recent_avg - baseline_avg
    return {
        "status": "ok",
        "recent_avg_top_probability": recent_avg,
        "baseline_avg_top_probability": baseline_avg,
        "drift_delta": shift,
        "drift_level": "high" if abs(shift) >= 0.15 else ("medium" if abs(shift) >= 0.07 else "low"),
    }


@app.get("/v1/patients/{patient_id}/risk-timeline", response_model=List[Dict[str, Any]])
def risk_timeline(patient_id: str, _: None = Depends(require_api_key)):
    return _patient_timeline(patient_id)

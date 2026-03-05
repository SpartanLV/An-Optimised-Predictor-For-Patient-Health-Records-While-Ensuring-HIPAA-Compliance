import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import logging
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

APP_NAME = "cse400-web-portal"

# Internal service URLs (docker compose uses internal DNS)
AUTH_URL = os.getenv("AUTH_URL", "http://auth:8004").strip()
PREPROCESS_URL = os.getenv("PREPROCESS_URL", "http://preprocess:8001").strip()
PATIENT_STORE_URL = os.getenv("PATIENT_STORE_URL", "http://patient-store:8002").strip()
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8003").strip()

# Optional single-ingress gateway (recommended)
GATEWAY_URL = os.getenv("GATEWAY_URL", "").strip().rstrip("/")
GATEWAY_TLS_VERIFY = os.getenv("GATEWAY_TLS_VERIFY", "ca").strip().lower()
GATEWAY_CA_CERT = os.getenv("GATEWAY_CA_CERT", "/caddy/caddy/pki/authorities/local/root.crt").strip()

if GATEWAY_URL:
    # Route all server-side calls through the gateway
    AUTH_URL = f"{GATEWAY_URL}/auth"
    PREPROCESS_URL = f"{GATEWAY_URL}/api/preprocess"
    PATIENT_STORE_URL = f"{GATEWAY_URL}/api/store"
    INFERENCE_URL = f"{GATEWAY_URL}/api/inference"


def _gateway_verify():
    """Return httpx TLS verify setting for calls to the gateway.

    - false: disable verification (dev only)
    - ca: verify using the CA bundle at GATEWAY_CA_CERT (recommended for internal Caddy CA)
    - true: use system trust store
    """
    if not GATEWAY_URL:
        return True
    if GATEWAY_TLS_VERIFY in ("0", "false", "no", "off"):
        return False
    if GATEWAY_TLS_VERIFY in ("ca", "cafile", "cert"):
        return GATEWAY_CA_CERT if (GATEWAY_CA_CERT and os.path.exists(GATEWAY_CA_CERT)) else False
    if GATEWAY_TLS_VERIFY in ("1", "true", "yes", "on"):
        return True
    return False


def _client_kwargs(timeout_seconds: float) -> dict:
    return {"timeout": timeout_seconds, "verify": _gateway_verify()}

COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "access_token").strip()
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").strip().lower() in ("1", "true", "yes")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

app = FastAPI(title="CSE400 Web Portal", version="0.2.0")

logger = logging.getLogger("web-portal")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration_ms = int((time.time() - start) * 1000)
        # Add header if we have a response object in scope (normal path)
        try:
            response.headers["X-Request-ID"] = request_id  # type: ignore[name-defined]
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
        except Exception:
            pass
        logger.info(json.dumps({
            "msg": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": duration_ms,
        }))


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _auth_headers(token: str, request: Request | None = None) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {token}"}
    try:
        if request is not None and getattr(request.state, 'request_id', None):
            h["X-Request-ID"] = request.state.request_id
    except Exception:
        pass
    return h


async def _me(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    try:
        async with httpx.AsyncClient(**_client_kwargs(15.0)) as client:
            r = await client.get(f"{AUTH_URL.rstrip('/')}/v1/auth/me", headers=_auth_headers(token))
            if r.status_code != 200:
                return None
            return r.json()
    except Exception:
        return None



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    token = request.cookies.get(COOKIE_NAME, "")
    user = await _me(token) if token else None
    return templates.TemplateResponse("index.html", {"request": request, "app_name": APP_NAME, "user": user})


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    # Request JWT from auth service
    try:
        async with httpx.AsyncClient(**_client_kwargs(15.0)) as client:
            r = await client.post(
                f"{AUTH_URL.rstrip('/')}/v1/auth/token",
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if r.status_code != 200:
            return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials."})
        tok = r.json().get("access_token", "")
        if not tok:
            return templates.TemplateResponse("login.html", {"request": request, "error": "Auth failed."})

        resp = RedirectResponse(url="/patients", status_code=303)
        resp.set_cookie(
            key=COOKIE_NAME,
            value=tok,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite="lax",
        )
        return resp
    except Exception as e:
        return templates.TemplateResponse("login.html", {"request": request, "error": f"Auth service error: {e}"})


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


@app.get("/patients", response_class=HTMLResponse)
async def patients(request: Request):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    async with httpx.AsyncClient(**_client_kwargs(30.0)) as client:
        r = await client.get(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients", headers=_auth_headers(token, request), params={"limit": 50})
        r.raise_for_status()
        pts = r.json()

    query = request.query_params.get("q", "").strip().lower()
    if query:
        pts = [
            p for p in pts
            if query in (p.get("id", "").lower())
            or query in (p.get("mrn", "").lower())
            or query in ((p.get("first_name") or "").lower())
            or query in ((p.get("last_name") or "").lower())
        ]

    now = _utcnow()
    recent_cutoff = now - timedelta(days=7)
    with_mrn = 0
    created_recent = 0
    for p in pts:
        if p.get("mrn"):
            with_mrn += 1
        created_raw = p.get("created_at_utc")
        if created_raw:
            try:
                created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if created_dt >= recent_cutoff:
                    created_recent += 1
            except Exception:
                pass

    return templates.TemplateResponse(
        "patients.html",
        {
            "request": request,
            "patients": pts,
            "user": user,
            "query": query,
            "stats": {
                "total": len(pts),
                "with_mrn": with_mrn,
                "created_recent": created_recent,
            },
        },
    )


@app.get("/audit", response_class=HTMLResponse)
async def audit_log(request: Request):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    logs = []
    message = None
    async with httpx.AsyncClient(**_client_kwargs(30.0)) as client:
        try:
            r = await client.get(
                f"{PATIENT_STORE_URL.rstrip('/')}/v1/audit",
                headers=_auth_headers(token, request),
                params={"limit": 100},
            )
            r.raise_for_status()
            logs = r.json()
        except Exception as e:
            message = f"Unable to load audit logs: {e}"

    return templates.TemplateResponse(
        "audit.html",
        {"request": request, "user": user, "logs": logs, "message": message},
    )


@app.get("/system-health", response_class=HTMLResponse)
async def system_health(request: Request):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    checks = [
        ("Auth Service", f"{AUTH_URL.rstrip('/')}/health", False),
        ("Preprocess Service", f"{PREPROCESS_URL.rstrip('/')}/health", True),
        ("Patient Store Service", f"{PATIENT_STORE_URL.rstrip('/')}/health", True),
        ("Inference Service", f"{INFERENCE_URL.rstrip('/')}/health", True),
    ]
    statuses: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(**_client_kwargs(20.0)) as client:
        for name, url, needs_auth in checks:
            try:
                headers = _auth_headers(token, request) if needs_auth else {}
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    payload = resp.json()
                    statuses.append({"name": name, "status": "ok", "details": payload})
                else:
                    statuses.append({"name": name, "status": "error", "details": {"status_code": resp.status_code}})
            except Exception as e:
                statuses.append({"name": name, "status": "error", "details": {"error": str(e)}})

    return templates.TemplateResponse(
        "system_health.html",
        {"request": request, "user": user, "statuses": statuses},
    )


@app.get("/api-contracts", response_class=HTMLResponse)
async def api_contracts(request: Request):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    contracts_path = Path(__file__).resolve().parents[3] / "docs" / "API_CONTRACTS.md"
    content = "API contracts file not found."
    try:
        content = contracts_path.read_text(encoding="utf-8")
    except Exception:
        pass

    return templates.TemplateResponse(
        "api_contracts.html",
        {"request": request, "user": user, "contracts": content},
    )


@app.post("/patients")
async def create_patient(
    request: Request,
    mrn: str = Form(default=""),
    first_name: str = Form(default=""),
    last_name: str = Form(default=""),
    dob: str = Form(default=""),
    gender: str = Form(default=""),
):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)

    payload = {
        "mrn": mrn or None,
        "first_name": first_name or None,
        "last_name": last_name or None,
        "dob": dob or None,
        "gender": gender or None,
    }
    async with httpx.AsyncClient(**_client_kwargs(30.0)) as client:
        r = await client.post(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients", headers=_auth_headers(token, request), json=payload)
        r.raise_for_status()
        pid = r.json()["id"]
    return RedirectResponse(url=f"/patients/{pid}", status_code=303)


@app.get("/patients/{patient_id}", response_class=HTMLResponse)
async def patient_detail(request: Request, patient_id: str):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    msg = None
    preds = None
    do_predict = request.query_params.get("predict") == "1"

    async with httpx.AsyncClient(**_client_kwargs(60.0)) as client:
        p = await client.get(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}", headers=_auth_headers(token))
        p.raise_for_status()
        patient = p.json()

        d = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/documents",
            headers=_auth_headers(token, request),
            params={"limit": 50},
        )
        # Older versions may not expose this endpoint; fail soft.
        docs = d.json() if d.status_code == 200 else []

        ev = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/events",
            headers=_auth_headers(token, request),
            params={"limit": 200},
        )
        ev.raise_for_status()
        events = ev.json()

        if do_predict:
            try:
                r = await client.post(
                    f"{INFERENCE_URL.rstrip('/')}/v1/predict",
                    headers=_auth_headers(token, request),
                    json={"patient_id": patient_id, "top_n": 10, "return_debug": False},
                )
                r.raise_for_status()
                preds = r.json()
            except Exception as e:
                msg = f"Inference failed: {e}"

    return templates.TemplateResponse(
        "patient_detail.html",
        {"request": request, "patient": patient, "events": events, "documents": docs, "predictions": preds, "message": msg, "user": user},
    )

@app.get("/patients/{patient_id}/documents/{document_id}", response_class=HTMLResponse)
async def document_detail(request: Request, patient_id: str, document_id: str):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    async with httpx.AsyncClient(**_client_kwargs(60.0)) as client:
        p = await client.get(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}", headers=_auth_headers(token))
        p.raise_for_status()
        patient = p.json()

        docs_r = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/documents",
            headers=_auth_headers(token, request),
            params={"limit": 200},
        )
        docs = docs_r.json() if docs_r.status_code == 200 else []
        doc = next((d for d in docs if d.get("id") == document_id), None)

        ents_r = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/documents/{document_id}/entities",
            headers=_auth_headers(token, request),
            params={"limit": 500},
        )
        entities = ents_r.json() if ents_r.status_code == 200 else []

    return templates.TemplateResponse(
        "document_detail.html",
        {"request": request, "patient": patient, "document": doc, "entities": entities, "user": user},
    )

@app.post("/patients/{patient_id}/predict", response_class=HTMLResponse)
async def predict_only(request: Request, patient_id: str):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    msg = None
    preds = None
    # These two are only produced during Upload+Run. Keep them in the template context
    # so prediction-only renders cleanly.
    preprocess_payload = None
    stored_doc = None
    do_predict = request.query_params.get("predict") == "1"

    async with httpx.AsyncClient(**_client_kwargs(120.0)) as client:
        try:
            r = await client.post(
                f"{INFERENCE_URL.rstrip('/')}/v1/predict",
                headers=_auth_headers(token, request),
                json={"patient_id": patient_id, "top_n": 10, "return_debug": False},
            )
            r.raise_for_status()
            preds = r.json()
        except Exception as e:
            msg = f"Inference failed: {e}"

        p = await client.get(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}", headers=_auth_headers(token))
        p.raise_for_status()
        patient = p.json()

        d = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/documents",
            headers=_auth_headers(token, request),
            params={"limit": 50},
        )
        docs = d.json() if d.status_code == 200 else []

        ev = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/events",
            headers=_auth_headers(token, request),
            params={"limit": 200},
        )
        ev.raise_for_status()
        events = ev.json()

    return templates.TemplateResponse(
        "patient_detail.html",
        {
            "request": request,
            "patient": patient,
            "events": events,
            "documents": docs,
            "predictions": preds,
            "message": msg,
            "user": user,
            "extraction": preprocess_payload,
            "stored_doc": stored_doc,
        },
    )



@app.post("/patients/{patient_id}/upload", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    patient_id: str,
    file: UploadFile = File(...),
    document_date: str = Form(default=""),
):
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    user = await _me(token)

    msg = None
    preprocess_payload = None
    stored_doc = None
    preds = None

    file_bytes = await file.read()
    if not file_bytes:
        return RedirectResponse(url=f"/patients/{patient_id}", status_code=303)

    # 1) Stage 1: preprocess (OCR/NER)
    async with httpx.AsyncClient(**_client_kwargs(120.0)) as client:
        try:
            files = {"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")}
            data = {"patient_id": patient_id, "document_date": document_date or ""}
            r = await client.post(f"{PREPROCESS_URL.rstrip('/')}/v1/preprocess", headers=_auth_headers(token, request), data=data, files=files)
            r.raise_for_status()
            preprocess_payload = r.json()
        except Exception as e:
            msg = f"Preprocess failed: {e}"

    # 2) Stage 2: store document + extraction (if preprocess succeeded)
    if preprocess_payload:
        async with httpx.AsyncClient(**_client_kwargs(120.0)) as client:
            try:
                files = {"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")}
                data = {"extraction_json": json.dumps(preprocess_payload)}
                r = await client.post(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/documents", headers=_auth_headers(token, request), data=data, files=files)
                r.raise_for_status()
                stored_doc = r.json()
            except Exception as e:
                msg = f"Store failed: {e}"

    # 3) Stage 3: predict
    async with httpx.AsyncClient(**_client_kwargs(120.0)) as client:
        try:
            r = await client.post(f"{INFERENCE_URL.rstrip('/')}/v1/predict", headers=_auth_headers(token, request), json={"patient_id": patient_id, "top_n": 10, "return_debug": False})
            r.raise_for_status()
            preds = r.json()
        except Exception as e:
            msg = f"Inference failed: {e}"

        p = await client.get(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}", headers=_auth_headers(token))
        p.raise_for_status()
        patient = p.json()

        ev = await client.get(f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/events", headers=_auth_headers(token, request), params={"limit": 200})
        ev.raise_for_status()
        events = ev.json()

        d = await client.get(
            f"{PATIENT_STORE_URL.rstrip('/')}/v1/patients/{patient_id}/documents",
            headers=_auth_headers(token, request),
            params={"limit": 50},
        )
        docs = d.json() if d.status_code == 200 else []

    return templates.TemplateResponse(
        "patient_detail.html",
        {"request": request, "patient": patient, "events": events, "documents": docs, "predictions": preds, "message": msg, "user": user, "extraction": preprocess_payload, "stored_doc": stored_doc},
    )

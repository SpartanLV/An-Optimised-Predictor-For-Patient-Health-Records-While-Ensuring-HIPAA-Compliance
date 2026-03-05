import json
import os
import pathlib
import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import SessionLocal, init_db_with_retry
from .models import Patient, Event, Document, NEREntity, AuditLog, Cohort
from .schemas import (
    PatientCreate, PatientOut,
    BulkEventsIn, EventOut,
    DocumentOut, ExtractionPayload,
    NEREntityOut,
    CohortCreate,
    CohortOut,
)
from .security import require_api_key, actor_from_headers, require_roles, can_view_phi
from .crypto import encrypt, encryption_enabled
from .interop import patient_to_fhir_bundle, fhir_bundle_to_patient_events, detect_care_gaps, patient_matches_filters

APP_NAME = "cse400-patient-store-service"

STORAGE_DIR = pathlib.Path(os.getenv("STORAGE_DIR", "./data/documents")).resolve()


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    init_db_with_retry()
    yield


app = FastAPI(
    title="CSE400 Patient Store Service (Stage 2)",
    version="0.2.0",
    description="Stores patients, documents, extracted entities, and structured events.",
    lifespan=lifespan,
)


@app.middleware("http")
async def _request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME, "storage_dir": str(STORAGE_DIR), "file_encryption": encryption_enabled()}


def _audit(db: Session, request: Request, action: str, actor: str, patient_id: Optional[str] = None, document_id: Optional[str] = None, metadata: Optional[dict] = None):
    meta = dict(metadata or {})
    rid = getattr(request.state, 'request_id', None) or request.headers.get('X-Request-ID')
    if rid:
        meta['request_id'] = rid

    log = AuditLog(
        actor=actor,
        action=action,
        patient_id=patient_id,
        document_id=document_id,
        route=str(request.url.path),
        method=request.method,
        ip=request.client.host if request.client else None,
        meta=meta or None,
    )
    db.add(log)
    db.commit()


def _masked_patient_out(p: Patient, auth: Dict[str, Any]) -> PatientOut:
    out = PatientOut.model_validate(p)
    if not can_view_phi(auth):
        out.first_name = "REDACTED" if out.first_name else None
        out.last_name = "REDACTED" if out.last_name else None
        out.dob = "REDACTED" if out.dob else None
    return out


def _enforce_patient_read(auth: Dict[str, Any]) -> None:
    if auth.get("mode") == "api_key":
        return
    require_roles(auth, ["admin", "clinician", "auditor"])


@app.post("/v1/patients", response_model=PatientOut)
def create_patient(payload: PatientCreate, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "clinician"])
    p = Patient(
        mrn=payload.mrn,
        first_name=payload.first_name,
        last_name=payload.last_name,
        dob=payload.dob,
        gender=payload.gender,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    _audit(db, request, action="patient.create", actor=actor, patient_id=p.id)
    return _masked_patient_out(p, auth)


@app.get("/v1/patients", response_model=List[PatientOut])
def list_patients(limit: int = 50, offset: int = 0, request: Request = None, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    q = select(Patient).order_by(Patient.created_at_utc.desc()).limit(limit).offset(offset)
    patients = db.execute(q).scalars().all()
    _audit(db, request, action="patient.list", actor=actor, metadata={"limit": limit, "offset": offset})
    return [_masked_patient_out(p, auth) for p in patients]


@app.get("/v1/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    _audit(db, request, action="patient.read", actor=actor, patient_id=patient_id)
    return _masked_patient_out(p, auth)


@app.post("/v1/patients/{patient_id}/events:bulk", response_model=dict)
def add_events_bulk(patient_id: str, payload: BulkEventsIn, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "clinician"])
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")

    count = 0
    for e in payload.events:
        ev = Event(
            patient_id=patient_id,
            source=e.source,
            event_date=e.event_date,
            code=e.code,
            description=e.description,
            value=e.value,
            units=e.units,
            type=e.type,
            reason_code=e.reason_code,
            reason_description=e.reason_description,
            raw=e.raw,
        )
        db.add(ev)
        count += 1

    db.commit()
    _audit(db, request, action="event.bulk_create", actor=actor, patient_id=patient_id, metadata={"count": count})
    return {"inserted": count}


@app.get("/v1/patients/{patient_id}/events", response_model=List[EventOut])
def list_events(patient_id: str, request: Request, dt_from: Optional[str] = None, dt_to: Optional[str] = None, limit: int = 5000, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")

    q = select(Event).where(Event.patient_id == patient_id)

    def _parse_dt(x: str) -> Optional[datetime]:
        if not x:
            return None
        try:
            d = datetime.fromisoformat(x.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except Exception:
            return None

    d_from = _parse_dt(dt_from) if dt_from else None
    d_to = _parse_dt(dt_to) if dt_to else None
    if d_from:
        q = q.where(Event.event_date >= d_from)
    if d_to:
        q = q.where(Event.event_date <= d_to)

    q = q.order_by(Event.event_date.asc()).limit(limit)
    events = db.execute(q).scalars().all()

    _audit(db, request, action="event.list", actor=actor, patient_id=patient_id, metadata={"from": dt_from, "to": dt_to, "limit": limit})
    return [EventOut.model_validate(e) for e in events]


@app.post("/v1/patients/{patient_id}/documents", response_model=DocumentOut)
async def upload_document(
    patient_id: str,
    request: Request,
    file: UploadFile = File(...),
    extraction_json: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    auth: Dict[str, Any] = Depends(require_api_key),
    actor: str = Depends(actor_from_headers),
):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "clinician"])
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    sha256 = hashlib.sha256(data).hexdigest()

    doc = Document(
        patient_id=patient_id,
        filename=file.filename,
        content_type=file.content_type,
        sha256=sha256,
        created_at_utc=_utcnow(),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    patient_dir = STORAGE_DIR / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (file.filename or "document").replace("/", "_").replace("\\", "_")
    out_path = patient_dir / f"{doc.id}_{safe_name}"
    out_path.write_bytes(encrypt(data))
    doc.storage_path = str(out_path)
    db.add(doc)

    if extraction_json:
        try:
            payload = ExtractionPayload.model_validate(json.loads(extraction_json))
        except Exception:
            payload = None
        if payload:
            for ent in payload.entities:
                db.add(NEREntity(
                    document_id=doc.id,
                    label=ent.label,
                    text=ent.text,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=ent.confidence,
                ))
            for e in payload.events:
                db.add(Event(
                    patient_id=patient_id,
                    source=e.source,
                    event_date=e.event_date,
                    code=e.code,
                    description=e.description,
                    value=e.value,
                    units=e.units,
                    type=e.type,
                    reason_code=e.reason_code,
                    reason_description=e.reason_description,
                    raw=e.raw,
                ))

    db.commit()
    _audit(db, request, action="document.create", actor=actor, patient_id=patient_id, document_id=doc.id, metadata={"filename": file.filename, "sha256": sha256})

    return DocumentOut(
        id=doc.id,
        patient_id=patient_id,
        filename=doc.filename,
        content_type=doc.content_type,
        sha256=doc.sha256,
        created_at_utc=doc.created_at_utc,
    )


@app.get("/v1/patients/{patient_id}/documents", response_model=List[DocumentOut])
def list_documents(
    patient_id: str,
    request: Request,
    db: Session = Depends(get_db),
    auth: Dict[str, Any] = Depends(require_api_key),
    actor: str = Depends(actor_from_headers),
    limit: int = 100,
):
    _enforce_patient_read(auth)
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    q = select(Document).where(Document.patient_id == patient_id).order_by(Document.created_at_utc.desc()).limit(limit)
    docs = db.execute(q).scalars().all()
    _audit(db, request, action="document.list", actor=actor, patient_id=patient_id, metadata={"limit": limit})
    return [DocumentOut.model_validate(d) for d in docs]


@app.get("/v1/patients/{patient_id}/documents/{document_id}/entities", response_model=List[NEREntityOut])
def list_document_entities(
    patient_id: str,
    document_id: str,
    request: Request,
    db: Session = Depends(get_db),
    auth: Dict[str, Any] = Depends(require_api_key),
    actor: str = Depends(actor_from_headers),
    limit: int = 500,
):
    _enforce_patient_read(auth)
    doc = db.get(Document, document_id)
    if not doc or doc.patient_id != patient_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    q = select(NEREntity).where(NEREntity.document_id == document_id).order_by(NEREntity.created_at_utc.asc()).limit(limit)
    ents = db.execute(q).scalars().all()
    _audit(db, request, action="entity.list", actor=actor, patient_id=patient_id, document_id=document_id, metadata={"limit": limit})
    return [NEREntityOut.model_validate(e) for e in ents]


@app.get("/v1/patients/{patient_id}/fhir-bundle", response_model=dict)
def patient_fhir_bundle(patient_id: str, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    events = db.execute(select(Event).where(Event.patient_id == patient_id).order_by(Event.event_date.asc())).scalars().all()
    docs = db.execute(select(Document).where(Document.patient_id == patient_id).order_by(Document.created_at_utc.desc())).scalars().all()
    bundle = patient_to_fhir_bundle(p, events, docs)
    _audit(db, request, action="fhir.export", actor=actor, patient_id=patient_id, metadata={"entry_count": len(bundle.get("entry") or [])})
    return bundle


@app.post("/v1/fhir-bundle/import", response_model=dict)
def import_fhir_bundle(payload: Dict[str, Any], request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "clinician"])
    parsed = fhir_bundle_to_patient_events(payload)
    p_in = parsed["patient"]
    patient = Patient(
        mrn=p_in.get("mrn"),
        first_name=p_in.get("first_name"),
        last_name=p_in.get("last_name"),
        dob=p_in.get("dob"),
        gender=p_in.get("gender"),
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    inserted = 0
    for e in parsed["events"]:
        ev = Event(
            patient_id=patient.id,
            source=e.get("source") or "obs",
            event_date=datetime.fromisoformat(str(e.get("event_date")).replace("Z", "+00:00")),
            code=e.get("code"),
            description=e.get("description"),
            value=e.get("value"),
        )
        db.add(ev)
        inserted += 1
    db.commit()
    _audit(db, request, action="fhir.import", actor=actor, patient_id=patient.id, metadata={"events_inserted": inserted})
    return {"patient_id": patient.id, "events_inserted": inserted}


@app.get("/v1/patients/{patient_id}/care-gaps", response_model=List[dict])
def patient_care_gaps(patient_id: str, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    events = db.execute(select(Event).where(Event.patient_id == patient_id)).scalars().all()
    gaps = detect_care_gaps(events)
    _audit(db, request, action="care_gaps.list", actor=actor, patient_id=patient_id, metadata={"count": len(gaps)})
    return gaps


@app.post("/v1/cohorts", response_model=CohortOut)
def create_cohort(payload: CohortCreate, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "clinician", "auditor"])
    c = Cohort(name=payload.name, created_by=actor, filters=payload.filters or {})
    db.add(c)
    db.commit()
    db.refresh(c)
    _audit(db, request, action="cohort.create", actor=actor, metadata={"cohort_id": c.id})
    return CohortOut.model_validate(c)


@app.get("/v1/cohorts", response_model=List[CohortOut])
def list_cohorts(request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers), limit: int = 100):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "clinician", "auditor"])
    cohorts = db.execute(select(Cohort).order_by(Cohort.created_at_utc.desc()).limit(limit)).scalars().all()
    _audit(db, request, action="cohort.list", actor=actor, metadata={"limit": limit})
    return [CohortOut.model_validate(c) for c in cohorts]


@app.get("/v1/cohorts/{cohort_id}/patients", response_model=List[PatientOut])
def cohort_patients(cohort_id: str, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    cohort = db.get(Cohort, cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail="Cohort not found.")
    patients = db.execute(select(Patient)).scalars().all()
    out: List[PatientOut] = []
    for p in patients:
        events = db.execute(select(Event).where(Event.patient_id == p.id)).scalars().all()
        if patient_matches_filters(p, events, cohort.filters or {}):
            out.append(_masked_patient_out(p, auth))
    _audit(db, request, action="cohort.expand", actor=actor, metadata={"cohort_id": cohort_id, "matched": len(out)})
    return out


@app.get("/v1/patients/{patient_id}/summary", response_model=dict)
def patient_summary(patient_id: str, request: Request, db: Session = Depends(get_db), auth: Dict[str, Any] = Depends(require_api_key), actor: str = Depends(actor_from_headers)):
    _enforce_patient_read(auth)
    p = db.get(Patient, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    events = db.execute(select(Event).where(Event.patient_id == patient_id).order_by(Event.event_date.desc()).limit(20)).scalars().all()
    docs = db.execute(select(Document).where(Document.patient_id == patient_id).order_by(Document.created_at_utc.desc()).limit(10)).scalars().all()
    gaps = detect_care_gaps(events)
    _audit(db, request, action="summary.view", actor=actor, patient_id=patient_id)
    return {
        "patient": _masked_patient_out(p, auth).model_dump(),
        "recent_events": [EventOut.model_validate(e).model_dump() for e in events],
        "documents": [DocumentOut.model_validate(d).model_dump() for d in docs],
        "care_gaps": gaps,
        "generated_at_utc": _utcnow().isoformat(),
    }


@app.get("/v1/audit", response_model=List[dict])
def list_audit_logs(
    request: Request,
    db: Session = Depends(get_db),
    auth: Dict[str, Any] = Depends(require_api_key),
    actor: str = Depends(actor_from_headers),
    limit: int = 200,
):
    if auth.get("mode") != "api_key":
        require_roles(auth, ["admin", "auditor"])
    q = select(AuditLog).order_by(AuditLog.created_at_utc.desc()).limit(limit)
    logs = db.execute(q).scalars().all()
    _audit(db, request, action="audit.list", actor=actor, metadata={"limit": limit})
    return [
        {
            "id": l.id,
            "created_at_utc": l.created_at_utc,
            "actor": l.actor,
            "action": l.action,
            "patient_id": l.patient_id,
            "document_id": l.document_id,
            "route": l.route,
            "method": l.method,
            "ip": l.ip,
        }
        for l in logs
    ]

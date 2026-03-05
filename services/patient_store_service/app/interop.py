from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .models import Patient, Event, Document


def _iso(dt: datetime | None) -> str | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def patient_to_fhir_bundle(patient: Patient, events: List[Event], documents: List[Document]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    entries.append({
        "resource": {
            "resourceType": "Patient",
            "id": patient.id,
            "identifier": [{"system": "urn:mrn", "value": patient.mrn}] if patient.mrn else [],
            "name": [{"family": patient.last_name or "", "given": [patient.first_name or ""]}],
            "gender": (patient.gender or "unknown").lower(),
            "birthDate": patient.dob,
        }
    })

    for ev in events:
        rt = "Observation"
        if ev.source in {"med"}:
            rt = "MedicationStatement"
        elif ev.source in {"proc"}:
            rt = "Procedure"
        elif ev.source in {"enc", "cp", "alg"}:
            rt = "Condition"
        entries.append({
            "resource": {
                "resourceType": rt,
                "id": ev.id,
                "subject": {"reference": f"Patient/{patient.id}"},
                "code": {"text": ev.description or ev.code or ev.source},
                "effectiveDateTime": _iso(ev.event_date),
                "valueString": ev.value,
            }
        })

    for doc in documents:
        entries.append({
            "resource": {
                "resourceType": "DocumentReference",
                "id": doc.id,
                "status": "current",
                "subject": {"reference": f"Patient/{patient.id}"},
                "date": _iso(doc.created_at_utc),
                "description": doc.filename,
                "content": [{"attachment": {"contentType": doc.content_type, "hash": doc.sha256}}],
            }
        })

    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def fhir_bundle_to_patient_events(bundle: Dict[str, Any]) -> Dict[str, Any]:
    entries = bundle.get("entry") or []
    patient_payload: Dict[str, Any] = {"mrn": None, "first_name": None, "last_name": None, "dob": None, "gender": None}
    events: List[Dict[str, Any]] = []

    for e in entries:
        r = e.get("resource") or {}
        rt = r.get("resourceType")
        if rt == "Patient":
            names = r.get("name") or [{}]
            n0 = names[0] if names else {}
            given = n0.get("given") or []
            patient_payload.update({
                "mrn": ((r.get("identifier") or [{}])[0]).get("value"),
                "first_name": given[0] if given else None,
                "last_name": n0.get("family"),
                "dob": r.get("birthDate"),
                "gender": r.get("gender"),
            })
            continue

        source = "obs"
        if rt == "MedicationStatement":
            source = "med"
        elif rt == "Procedure":
            source = "proc"
        elif rt == "Condition":
            source = "enc"

        when = r.get("effectiveDateTime") or datetime.now(tz=timezone.utc).isoformat()
        events.append({
            "source": source,
            "event_date": when,
            "code": ((r.get("code") or {}).get("coding") or [{}])[0].get("code"),
            "description": (r.get("code") or {}).get("text"),
            "value": r.get("valueString"),
        })

    return {"patient": patient_payload, "events": events}


def detect_care_gaps(events: List[Event]) -> List[Dict[str, Any]]:
    now = datetime.now(tz=timezone.utc)
    obs = [e for e in events if e.source == "obs"]
    meds = [e for e in events if e.source == "med"]
    gaps: List[Dict[str, Any]] = []
    if not obs:
        gaps.append({"id": "missing_observation", "severity": "high", "title": "No recent observations", "rationale": "No vitals/labs found in timeline."})
    else:
        latest_obs = max((e.event_date for e in obs))
        if (now - latest_obs).days > 180:
            gaps.append({"id": "stale_observation", "severity": "medium", "title": "Observations outdated", "rationale": "Latest observation is older than 180 days."})
    if not meds:
        gaps.append({"id": "med_review", "severity": "low", "title": "Medication review needed", "rationale": "No medication events recorded."})
    return gaps


def patient_matches_filters(patient: Patient, events: List[Event], filters: Dict[str, Any]) -> bool:
    q = (filters.get("query") or "").strip().lower()
    if q:
        blob = " ".join([(patient.id or ""), (patient.mrn or ""), (patient.first_name or ""), (patient.last_name or "")]).lower()
        if q not in blob:
            return False

    min_events = int(filters.get("min_events") or 0)
    if len(events) < min_events:
        return False

    since_days = filters.get("recent_days")
    if since_days is not None:
        cutoff = datetime.now(tz=timezone.utc).timestamp() - int(since_days) * 86400
        if not any(e.event_date.timestamp() >= cutoff for e in events):
            return False

    return True

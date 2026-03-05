import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.patient_store_service.app import main
from services.patient_store_service.app.db import Base


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _client_with_db(tmp_path, monkeypatch):
    db_file = tmp_path / "patient_store_test.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{db_file}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(bind=engine)

    auth_state = {"mode": "jwt", "roles": ["clinician"], "sub": "tester"}

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def override_auth():
        return auth_state

    main.app.dependency_overrides[main.get_db] = override_get_db
    main.app.dependency_overrides[main.require_api_key] = override_auth
    main.app.dependency_overrides[main.actor_from_headers] = lambda: "tester"

    monkeypatch.setattr(main, "STORAGE_DIR", tmp_path / "documents")
    monkeypatch.setattr(main, "encrypt", lambda b: b)
    monkeypatch.setattr(main, "detect_care_gaps", lambda events: [{"id": "missing_observation"}] if not events else [])

    client = TestClient(main.app)
    return client, auth_state


def teardown_function():
    main.app.dependency_overrides.clear()


def test_create_and_list_patients_with_phi_masking(tmp_path, monkeypatch):
    client, auth_state = _client_with_db(tmp_path, monkeypatch)

    create = client.post(
        "/v1/patients",
        json={"mrn": "M-1", "first_name": "Alice", "last_name": "Ng", "dob": "1990-01-01", "gender": "female"},
    )
    assert create.status_code == 200
    patient_id = create.json()["id"]

    auth_state["roles"] = ["auditor"]
    listed = client.get("/v1/patients")
    assert listed.status_code == 200
    assert listed.json()[0]["first_name"] == "REDACTED"

    one = client.get(f"/v1/patients/{patient_id}")
    assert one.status_code == 200
    assert one.json()["last_name"] == "REDACTED"


def test_events_and_documents_endpoints(tmp_path, monkeypatch):
    client, _ = _client_with_db(tmp_path, monkeypatch)

    patient = client.post("/v1/patients", json={"first_name": "Bob", "last_name": "Li"}).json()
    patient_id = patient["id"]

    t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 2, 1, tzinfo=timezone.utc)
    inserted = client.post(
        f"/v1/patients/{patient_id}/events:bulk",
        json={
            "events": [
                {"source": "obs", "event_date": _iso(t1), "description": "bp"},
                {"source": "med", "event_date": _iso(t2), "description": "metformin"},
            ]
        },
    )
    assert inserted.status_code == 200
    assert inserted.json()["inserted"] == 2

    filtered = client.get(f"/v1/patients/{patient_id}/events?dt_from=2024-01-15T00:00:00Z")
    assert filtered.status_code == 200
    assert len(filtered.json()) == 1
    assert filtered.json()[0]["source"] == "med"

    extraction = {
        "entities": [{"label": "TEST", "text": "alpha", "start_char": 0, "end_char": 5}],
        "events": [{"source": "obs", "event_date": _iso(datetime.now(tz=timezone.utc)), "description": "spo2"}],
    }
    uploaded = client.post(
        f"/v1/patients/{patient_id}/documents",
        files={"file": ("note.txt", b"clinical note", "text/plain")},
        data={"extraction_json": json.dumps(extraction)},
    )
    assert uploaded.status_code == 200
    doc_id = uploaded.json()["id"]

    docs = client.get(f"/v1/patients/{patient_id}/documents")
    assert docs.status_code == 200
    assert len(docs.json()) == 1

    entities = client.get(f"/v1/patients/{patient_id}/documents/{doc_id}/entities")
    assert entities.status_code == 200
    assert entities.json()[0]["label"] == "TEST"

    summary = client.get(f"/v1/patients/{patient_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["documents"]
    assert summary.json()["recent_events"]

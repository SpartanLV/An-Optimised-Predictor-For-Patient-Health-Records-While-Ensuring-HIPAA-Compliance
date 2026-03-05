from fastapi.testclient import TestClient

from services.preprocess_service.app import main


client = TestClient(main.app)


def setup_function():
    main.app.dependency_overrides[main.require_auth] = lambda: {"sub": "tester", "roles": ["clinician"]}


def teardown_function():
    main.app.dependency_overrides.clear()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "cse400-preprocess-service"


def test_preprocess_rejects_empty_upload():
    response = client.post(
        "/v1/preprocess",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty upload."


def test_preprocess_rejects_file_too_large(monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 3)
    response = client.post(
        "/v1/preprocess",
        files={"file": ("data.txt", b"1234", "text/plain")},
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_preprocess_extracts_entities_and_events(monkeypatch):
    monkeypatch.setattr(main, "extract_text_from_upload", lambda **_: "raw OCR")
    monkeypatch.setattr(main, "clean_ocr_text", lambda text: f"clean::{text}")
    monkeypatch.setattr(main, "ner_entities", lambda text: [{"label": "TEST", "text": text}])
    monkeypatch.setattr(
        main,
        "build_event_hints_from_text",
        lambda text, event_dt: [
            {
                "source": "obs",
                "event_date": event_dt,
                "description": text,
                "value": "98",
                "units": "%",
            }
        ],
    )

    response = client.post(
        "/v1/preprocess",
        data={"patient_id": "p-1", "document_date": "2025-01-01"},
        files={"file": ("report.txt", b"abc", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == "p-1"
    assert body["filename"] == "report.txt"
    assert body["entities"][0]["label"] == "TEST"
    assert body["events"][0]["source"] == "obs"
    assert body["text"] == ""
    assert len(body["document_sha256"]) == 64

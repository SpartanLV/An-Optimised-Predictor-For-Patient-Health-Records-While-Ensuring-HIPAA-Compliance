from datetime import datetime, timezone

from fastapi.testclient import TestClient

from services.inference_service.app import main


class DummyBundle:
    metadata = {"model_version": "test-v2"}

    def predict_from_events(self, events, as_of=None, top_n=10, return_debug=False):
        return {
            "threshold": 0.42,
            "predictions": [
                {
                    "code": "RISK",
                    "description": "Risk",
                    "probability": 0.77,
                    "recommendations": {"medications": [], "procedures": [], "careplans": []},
                }
            ],
            "debug": {"count": len(events)},
        }


def setup_function():
    main.bundle = DummyBundle()
    main.telemetry_log.clear()
    main.app.dependency_overrides[main.require_api_key] = lambda: None


def teardown_function():
    main.app.dependency_overrides.clear()


def test_predict_with_events_payload_only():
    client = TestClient(main.app)

    response = client.post(
        "/v1/predict",
        json={
            "events": [
                {
                    "source": "obs",
                    "event_date": datetime.now(tz=timezone.utc).isoformat(),
                    "description": "blood pressure",
                }
            ],
            "top_n": 5,
        },
        headers={"X-API-Key": "dummy"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == 0.42
    assert body["predictions"][0]["explanation"] == ["blood pressure"]


def test_predict_requires_patient_or_events():
    client = TestClient(main.app)

    response = client.post("/v1/predict", json={}, headers={"X-API-Key": "dummy"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Provide either patient_id or events."


def test_predict_requires_store_configuration_for_patient_lookup(monkeypatch):
    monkeypatch.setattr(main, "PATIENT_STORE_URL", "")
    monkeypatch.setattr(main, "API_KEY", "")
    client = TestClient(main.app)

    response = client.post("/v1/predict", json={"patient_id": "p-1"}, headers={"X-API-Key": "dummy"})
    assert response.status_code == 500
    assert response.json()["detail"] == "PATIENT_STORE_URL not configured."


def test_predict_returns_502_when_store_fetch_fails(monkeypatch):
    async def failing_fetch(*args, **kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(main, "PATIENT_STORE_URL", "http://store")
    monkeypatch.setattr(main, "API_KEY", "k")
    monkeypatch.setattr(main, "fetch_events_from_store", failing_fetch)
    client = TestClient(main.app)

    response = client.post("/v1/predict", json={"patient_id": "p-2"}, headers={"X-API-Key": "dummy"})
    assert response.status_code == 502
    assert "Failed to fetch events from store" in response.json()["detail"]

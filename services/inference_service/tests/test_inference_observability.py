from datetime import datetime, timezone
import sys
import types

from fastapi.testclient import TestClient


fake_model_runtime = types.ModuleType("services.inference_service.app.model_runtime")


class DummyBundle:
    metadata = {"model_version": "test-v1"}

    def __init__(self, *args, **kwargs):
        pass

    def predict_from_events(self, events, as_of=None, top_n=10, return_debug=False):
        return {
            "threshold": 0.5,
            "predictions": [
                {
                    "code": "C1",
                    "description": "Condition 1",
                    "probability": 0.8,
                    "recommendations": {"medications": [], "procedures": [], "careplans": []},
                }
            ],
            "debug": None,
        }


async def _default_fetch(*args, **kwargs):
    return []


fake_model_runtime.ModelBundle = DummyBundle
fake_model_runtime.fetch_events_from_store = _default_fetch
sys.modules["services.inference_service.app.model_runtime"] = fake_model_runtime

from services.inference_service.app import main  # noqa: E402


def _setup(monkeypatch):
    main.bundle = DummyBundle()
    main.telemetry_log.clear()

    async def fake_fetch_events(*args, **kwargs):
        return [
            {"source": "obs", "event_date": datetime.now(tz=timezone.utc).isoformat(), "description": "blood pressure"},
            {"source": "obs", "event_date": datetime.now(tz=timezone.utc).isoformat(), "description": "blood pressure"},
            {"source": "med", "event_date": datetime.now(tz=timezone.utc).isoformat(), "description": "metformin"},
        ]

    monkeypatch.setattr(main, "fetch_events_from_store", fake_fetch_events)
    monkeypatch.setattr(main, "PATIENT_STORE_URL", "http://store")
    monkeypatch.setattr(main, "API_KEY", "k")


def test_predict_includes_explanations_and_timeline(monkeypatch):
    _setup(monkeypatch)
    main.app.dependency_overrides[main.require_api_key] = lambda: None
    client = TestClient(main.app)

    resp = client.post("/v1/predict", json={"patient_id": "p-1"}, headers={"X-API-Key": "k"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"][0]["explanation"]
    assert "timeline" in body


def test_metrics_and_drift_endpoints(monkeypatch):
    _setup(monkeypatch)
    main.app.dependency_overrides[main.require_api_key] = lambda: None
    client = TestClient(main.app)

    for _ in range(3):
        client.post("/v1/predict", json={"patient_id": "p-1"}, headers={"X-API-Key": "k"})

    m = client.get("/v1/inference/metrics", headers={"X-API-Key": "k"})
    assert m.status_code == 200
    assert m.json()["total_requests"] >= 3

    d = client.get("/v1/inference/drift", headers={"X-API-Key": "k"})
    assert d.status_code == 200
    assert "status" in d.json()



def teardown_module(module):
    main.app.dependency_overrides.clear()

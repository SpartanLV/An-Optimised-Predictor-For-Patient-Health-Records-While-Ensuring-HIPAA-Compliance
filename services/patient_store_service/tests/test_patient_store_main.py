from fastapi.testclient import TestClient

from services.patient_store_service.app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "cse400-patient-store-service"

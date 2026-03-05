import sys
import os
from pathlib import Path

# Add service root to path
service_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(service_root))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "cse400-patient-store-service"

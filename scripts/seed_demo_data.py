"""
Seed a demo patient + a few events, so you can test inference quickly.

Usage (with docker compose running):
  python scripts/seed_demo_data.py

Requires env:
  API_KEY
  PATIENT_STORE_URL (default http://localhost:8002)
"""
import os
from datetime import datetime, timedelta, timezone
import httpx

API_KEY = os.getenv("API_KEY", "change-me-please")
PATIENT_STORE_URL = os.getenv("PATIENT_STORE_URL", "http://localhost:8002").rstrip("/")

HEADERS = {"X-API-Key": API_KEY, "X-User-Id": "seed-script"}

def main():
    now = datetime.now(tz=timezone.utc)

    patient_payload = {"mrn": "DEMO-001", "first_name": "Demo", "last_name": "Patient", "dob": "1980-01-01", "gender": "M"}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{PATIENT_STORE_URL}/v1/patients", headers=HEADERS, json=patient_payload)
        r.raise_for_status()
        pid = r.json()["id"]
        print("Created patient:", pid)

        events = [
            {"source": "enc", "event_date": (now - timedelta(days=10)).isoformat(), "code": "99213", "description": "office_visit"},
            {"source": "obs", "event_date": (now - timedelta(days=9)).isoformat(), "description": "blood_pressure", "value": "130/85", "units": "mmHg", "type": "vital"},
            {"source": "med", "event_date": (now - timedelta(days=8)).isoformat(), "description": "metformin 500mg"},
            {"source": "proc", "event_date": (now - timedelta(days=7)).isoformat(), "description": "blood_test"},
        ]
        r = client.post(f"{PATIENT_STORE_URL}/v1/patients/{pid}/events:bulk", headers=HEADERS, json={"events": events})
        r.raise_for_status()
        print("Inserted events:", r.json())

        print("Now run inference via:")
        print(f"  curl -H 'X-API-Key: {API_KEY}' -X POST http://localhost:8003/v1/predict -H 'Content-Type: application/json' -d '{{\"patient_id\":\"{pid}\",\"top_n\":10}}'")

if __name__ == "__main__":
    main()

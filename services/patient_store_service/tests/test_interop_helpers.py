from datetime import datetime, timezone

from services.patient_store_service.app.interop import detect_care_gaps


class E:
    def __init__(self, source: str, event_date: datetime):
        self.source = source
        self.event_date = event_date


def test_detect_care_gaps_flags_missing_observations_and_meds():
    events = [E("enc", datetime.now(tz=timezone.utc))]
    gaps = detect_care_gaps(events)
    ids = {g["id"] for g in gaps}
    assert "missing_observation" in ids
    assert "med_review" in ids

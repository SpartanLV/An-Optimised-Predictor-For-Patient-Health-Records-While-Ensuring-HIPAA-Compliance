from datetime import datetime, timezone
from services.web_portal.app.main import _filter_patients, _parse_iso_utc, _patient_stats


def test_parse_iso_utc_handles_z_suffix():
    parsed = _parse_iso_utc("2024-01-15T10:30:00Z")
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_filter_patients_by_name_or_mrn():
    pts = [
        {"id": "aaa", "mrn": "M-100", "first_name": "Alice", "last_name": "Ng"},
        {"id": "bbb", "mrn": "M-200", "first_name": "Bob", "last_name": "Li"},
    ]
    filtered = _filter_patients(pts, "alice")
    assert len(filtered) == 1
    assert filtered[0]["id"] == "aaa"

    filtered_mrn = _filter_patients(pts, "m-200")
    assert len(filtered_mrn) == 1
    assert filtered_mrn[0]["id"] == "bbb"


def test_patient_stats_counts_recent_and_mrn():
    now = datetime(2024, 2, 1, tzinfo=timezone.utc)
    pts = [
        {"mrn": "M-100", "created_at_utc": "2024-01-30T00:00:00Z"},
        {"mrn": "", "created_at_utc": "2023-12-01T00:00:00Z"},
    ]
    stats = _patient_stats(pts, now=now)
    assert stats == {"total": 2, "with_mrn": 1, "created_recent": 1}

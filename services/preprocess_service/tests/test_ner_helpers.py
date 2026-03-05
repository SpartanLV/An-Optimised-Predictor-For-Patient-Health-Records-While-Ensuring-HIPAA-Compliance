from datetime import datetime, timezone

from services.preprocess_service.app.ner import build_event_hints_from_text, clean_ocr_text


def test_clean_ocr_text_removes_page_numbers_and_boilerplate():
    noisy = """
    1
    ACME Hospital
    Blood pressure 120/80
    Visit www.example.com
    """
    cleaned = clean_ocr_text(noisy)
    assert "ACME Hospital" not in cleaned
    assert "www.example.com" not in cleaned
    assert "Blood pressure 120/80" in cleaned


def test_build_event_hints_extracts_multiple_clinical_signals():
    text = """
    BP 130/85
    HR 98 bpm
    glucose 140 mg/dl
    Metformin 500 mg
    """
    events = build_event_hints_from_text(text=text, event_dt=datetime.now(tz=timezone.utc))
    sources = {e.source for e in events}
    assert "obs" in sources
    assert "med" in sources
    assert len(events) >= 3

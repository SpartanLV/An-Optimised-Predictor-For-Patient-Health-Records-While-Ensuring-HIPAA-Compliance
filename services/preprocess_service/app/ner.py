import os
import re
from datetime import datetime
from typing import List

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

from .schemas import NEREntity, ExtractedEvent

_SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

_nlp = None
if spacy is not None:
    try:
        _nlp = spacy.load(_SPACY_MODEL)
    except Exception:
        _nlp = None

# Simple patterns for MVP “event hints”
_RX_MED = re.compile(r"\b([A-Z][a-zA-Z]{2,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)\b")
_RX_BP = re.compile(r"\b(?:BP|blood pressure)\b[^0-9]{0,10}(\d{2,3})\s*/\s*(\d{2,3})", re.IGNORECASE)
_RX_TEMP = re.compile(r"\b(?:temp|temperature)\b[^0-9]{0,10}(\d{2,3}(?:\.\d+)?)\s*([CF])\b", re.IGNORECASE)

# Additional vitals/observations
_RX_HR = re.compile(r"\b(?:HR|heart rate|pulse)\b[^0-9]{0,10}(\d{2,3})\s*(?:bpm)?", re.IGNORECASE)
_RX_O2 = re.compile(r"\b(?:SpO2|O2 sat|oxygen saturation)\b[^0-9]{0,10}(\d{2,3})\s*%?", re.IGNORECASE)
_RX_WT = re.compile(r"\bweight\b[^0-9]{0,10}(\d{2,3}(?:\.\d+)?)\s*(kg|lb|lbs)\b", re.IGNORECASE)
_RX_HT = re.compile(r"\bheight\b[^0-9]{0,10}(\d{2,3}(?:\.\d+)?)\s*(cm|m|in|inch|inches)\b", re.IGNORECASE)

_RX_WT_NOUNIT = re.compile(r"(?im)^Weight\s*:?\s*(\d{1,3}(?:\.\d+)?)\b")
_RX_HT_NOUNIT = re.compile(r"(?im)^Height\s*:?\s*(\d{2,3}(?:\.\d+)?)\b")
_RX_GLU = re.compile(r"\b(?:glucose|blood sugar)\b[^0-9]{0,10}(\d{2,3})\s*(mg/dl|mmol/l)?", re.IGNORECASE)
_RX_A1C = re.compile(r"\b(?:hba1c|a1c)\b[^0-9]{0,10}(\d{1,2}(?:\.\d+)?)\s*%", re.IGNORECASE)

# Very small diagnosis keyword map (demo only)
_DIAG_KEYWORDS = [
    (re.compile(r"\bdiabetes\b", re.IGNORECASE), '44054006', 'Diabetes mellitus'),
    (re.compile(r"\bhypertension\b", re.IGNORECASE), '38341003', 'Hypertensive disorder'),
    (re.compile(r"\basthma\b", re.IGNORECASE), '195967001', 'Asthma'),
    (re.compile(r"\bbronchitis\b", re.IGNORECASE), '10509002', 'Acute bronchitis'),
    (re.compile(r"\bappendicitis\b", re.IGNORECASE), '74400008', 'Appendicitis'),
]

_COMMON_PROCS = [
    "colonoscopy",
    "spirometry",
    "x-ray",
    "xray",
    "ct scan",
    "mri",
    "ultrasound",
    "intramuscular injection",
]


# --- Text cleanup & lightweight form-field extraction (MVP) ---

# Common footer/boilerplate found in generated PDFs (e.g., form builders)
_BOILERPLATE_PATTERNS = [
    re.compile(r"\bNow create your own Jotform\b", re.IGNORECASE),
    re.compile(r"\bCreate your own PDF Document\b", re.IGNORECASE),
    re.compile(r"\bPDF document\b", re.IGNORECASE),
    re.compile(r"\bIt's Free\b", re.IGNORECASE),
    re.compile(r"\bVisit\s+www\.[^\s]+\b", re.IGNORECASE),
    re.compile(r"\bWebsite:\s*www\.[^\s]+\b", re.IGNORECASE),
    re.compile(r"\bACME Hospital\b", re.IGNORECASE),
    re.compile(r"\bJotform\b", re.IGNORECASE),
    # Demo/sample form footer block (ACME Hospital)
    re.compile(r"\bFor appointments, billing inquiries\b", re.IGNORECASE),
    re.compile(r"\bacme@example\.com\b", re.IGNORECASE),
    re.compile(r"\bHealthcare Avenue\b", re.IGNORECASE),
    re.compile(r"\bMetropolis\b", re.IGNORECASE),
]

# Very small lorem-ipsum/noise detector for demo/test PDFs.
_LOREM_RX = re.compile(r"\b(duis|sagittis|morbi|ipsum|nibh|nisl)\b", re.IGNORECASE)

_RX_DATE_LIKE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_RX_PHONE = re.compile(r"(?:(?:\+?1[\s\-\.]*)?)?(?:\(?\d{3}\)?[\s\-\.]*)\d{3}[\s\-\.]*(?:\d{4})")
_RX_ZIP5 = re.compile(r"\b\d{5}\b")
_RX_STREET = re.compile(r"^\s*\d{1,5}\s+[A-Za-z][A-Za-z0-9 .,'\-]{1,60}\s+(?:Drive|Dr|Trail|Trl|Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Lane|Ln|Way|Court|Ct|Circle|Cir)\b", re.IGNORECASE)

_RX_IMMUNE = re.compile(r"(?im)^(Chicken Pox|Measles|Mumps|Rubella|Varicella|Influenza|COVID-19|Polio|Tetanus)\b[^\n]*?(?:\n\s*|:\s*)(IMMUNE|NOT\s+IMMUNE|UNKNOWN|Yes|No)\b")

_RX_HEPB = re.compile(r"(?im)Hepatitis\s*B\s*vaccination\?\s*\n\s*(Yes|No)\b")
_RX_ALLERGIES_FIELD = re.compile(r"(?is)List any allergies:\s*\n\s*(.+?)(?:\n\s*List any medication taken regularly:|\n\s*List any medication taken regularly\b|\n\s*Medical Insurance Details\b|\Z)")
_RX_MED_FIELD = re.compile(r"(?is)List any medication taken regularly:\s*\n\s*(.+?)(?:\n\s*Name of Insurance Company:|\n\s*Medical Insurance Details\b|\Z)")


def clean_ocr_text(text: str, max_lines: int = 5000) -> str:
    """Remove common boilerplate, page numbers, and collapse whitespace.

    This is intentionally conservative: we avoid removing medically relevant
    content, but strip noisy footer blocks that inflate NER false-positives.
    """
    if not text:
        return ""

    # Normalize newlines
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    cleaned = []
    for ln in lines[:max_lines]:
        if not ln:
            continue
        # Drop bare page numbers
        if ln.isdigit() and len(ln) <= 3:
            continue
        # Drop boilerplate/footer lines
        if any(rx.search(ln) for rx in _BOILERPLATE_PATTERNS):
            continue
        cleaned.append(ln)

    out = "\n".join(cleaned)
    # Collapse excessive whitespace
    out = re.sub(r"[\t ]{2,}", " ", out)
    return out.strip()


def _looks_like_noise(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    # Common misfires
    if t.lower() in {"email", "expiry date", "birth date", "home phone", "work phone"}:
        return True
    # Lorem ipsum / placeholder content (demo PDFs)
    if _LOREM_RX.search(t):
        return True
    return False


def _norm_phone(s: str) -> str:
    # normalize common phone formats to (XXX) XXX-XXXX when possible
    digits = re.sub(r"\D+", "", s or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    return (s or "").strip()


def _clean_possible_street(line: str) -> str:
    ln = (line or "").strip()
    # If OCR glued phone tail + street number (e.g., "9964 1195 Holly Street"), drop the first 3-4 digit chunk
    m = re.match(r"^(\d{3,4})\s+(\d{3,5}\s+.+)$", ln)
    if m and re.search(r"\b(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way|Circle|Cir)\b", ln, re.IGNORECASE):
        ln = m.group(2).strip()
    return ln


def _extract_form_entities(text: str) -> List[NEREntity]:
    """Lightweight extraction for common form fields.

    Goal: produce a *small set* of high-value structured entities from intake-form style docs,
    without needing a clinical NER model. These entities are returned first, with spaCy entities after.
    """
    out: List[NEREntity] = []
    if not text:
        return out

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    def add(label: str, value: str):
        v = (value or "").strip()
        if v and not _looks_like_noise(v):
            out.append(NEREntity(label=label, text=v))

    # --- Patient Information block ---
    pi_start = None
    for i, ln in enumerate(lines):
        if re.search(r"\bPatient Information\b", ln, re.IGNORECASE):
            pi_start = i
            break

    if pi_start is not None:
        stop_words = re.compile(r"\b(In Case of Emergency|General Medical History|Medical Insurance Details|Name of Insurance Company)\b", re.IGNORECASE)
        pi_lines: List[str] = []
        for ln in lines[pi_start + 1:]:
            if stop_words.search(ln):
                break
            pi_lines.append(ln)

        # name (first reasonable "Firstname ... Lastname")
        for ln in pi_lines[:8]:
            ln2 = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "", ln).strip()
            if not ln2 or re.search(r"\d", ln2):
                continue
            if re.search(r"\b(Patient Information|General Medical History|In Case of Emergency|United States)\b", ln2, re.IGNORECASE):
                continue
            tokens = ln2.split()
            # allow middle initials like "R" / "R."
            if len(tokens) < 2:
                continue
            if not all(re.match(r"^[A-Z][A-Za-z'\-]*\.?$", t) for t in tokens):
                continue
            multi_letter = [t for t in tokens if len(t.strip(".-")) > 1]
            if len(multi_letter) >= 2:
                add("PATIENT_NAME", ln2)
                break

# patient phone
        for ln in pi_lines[:12]:
            m = _RX_PHONE.search(ln)
            if m:
                add("PATIENT_PHONE", _norm_phone(m.group(0)))
                break
            dg = re.findall(r"\d+", ln)
            if len(dg) >= 3 and sum(len(x) for x in dg[:3]) == 10:
                add("PATIENT_PHONE", _norm_phone("".join(dg[:3])))
                break

        # address
        for ln in pi_lines:
            ln2 = _clean_possible_street(ln)
            if _RX_STREET.search(ln2):
                add("ADDRESS", ln2)
                break

        # city/state/zip
        for ln in pi_lines:
            m = re.search(r"^([A-Za-z .'-]+),\s*([A-Za-z .'-]+)?\(?([A-Z]{2})\)?\s*,?\s*(\d{5})\b", ln)
            if m:
                city = m.group(1).strip()
                state = (m.group(3) or "").strip()
                zip5 = m.group(4).strip()
                add("CITY", city)
                if state:
                    add("STATE", state)
                add("POSTCODE", zip5)
                break

        if any(re.search(r"\bUnited States\b", ln, re.IGNORECASE) for ln in pi_lines):
            add("COUNTRY", "United States")

    # --- Birth date / weight / height ---
    dob_val: Optional[str] = None
    m = re.search(r"Birth Date\s*[:\s]*\n?\s*(\d{1,2}/\d{1,2}/\d{4})", text, flags=re.IGNORECASE)
    if m:
        dob_val = m.group(1)
        add("DOB", dob_val)

    m = re.search(r"(?im)^Weight\s*:?\s*(\d{1,3}(?:\.\d+)?)\b", text)
    if not m:
        m = re.search(r"Weight\s*:?\s*\n\s*(\d{1,3}(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    if m:
        add("WEIGHT", m.group(1))

    m = re.search(r"(?im)^Height\s*:?\s*(\d{2,3}(?:\.\d+)?)\b", text)
    if not m:
        m = re.search(r"Height\s*:?\s*\n\s*(\d{2,3}(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    if m:
        add("HEIGHT", m.group(1))

    # Document date (top-right)
    # pick a date near the top that is *not* the DOB
    top = "\n".join(lines[:18])
    candidates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", top)
    for dt in candidates:
        if dob_val and dt == dob_val:
            continue
        add("DOCUMENT_DATE", dt)
        break

    # --- Emergency contact block ---
    em_start = None
    for i, ln in enumerate(lines):
        if re.search(r"\bIn Case of Emergency\b", ln, re.IGNORECASE):
            em_start = i
            break
    if em_start is not None:
        stop_words = re.compile(r"\b(General Medical History|Medical Insurance Details|Name of Insurance Company)\b", re.IGNORECASE)
        em_lines: List[str] = []
        for ln in lines[em_start + 1:]:
            if stop_words.search(ln):
                break
            em_lines.append(ln)

        for ln in em_lines[:8]:
            ln2 = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "", ln).strip()
            if not ln2 or re.search(r"\d", ln2):
                continue
            if re.search(r"\b(In Case of Emergency|General Medical History|United States)\b", ln2, re.IGNORECASE):
                continue
            tokens = ln2.split()
            if len(tokens) < 2:
                continue
            if not all(re.match(r"^[A-Z][A-Za-z'\-]*\.?$", t) for t in tokens):
                continue
            multi_letter = [t for t in tokens if len(t.strip(".-")) > 1]
            # allow "Bertie C Rowell" etc
            if len(multi_letter) >= 2:
                add("EMERGENCY_CONTACT_NAME", ln2)
                break
            m2 = re.match(r"^([A-Z][A-Za-z'\-]+(?:\s+[A-Z]\.?)*\s+[A-Z][A-Za-z'\-]+)\s+(\d{1,5}\s+.+)$", ln2)
            if m2:
                add("EMERGENCY_CONTACT_NAME", m2.group(1))
                ln_addr = _clean_possible_street(m2.group(2))
                if _RX_STREET.search(ln_addr):
                    add("EMERGENCY_ADDRESS", ln_addr)
                break


        if not any(e.label == "EMERGENCY_ADDRESS" for e in out):
            for ln in em_lines:
                ln2 = _clean_possible_street(ln)
                if _RX_STREET.search(ln2):
                    add("EMERGENCY_ADDRESS", ln2)
                    break

        for i, ln in enumerate(em_lines):
            if re.search(r"home phone", ln, re.IGNORECASE):
                for k in range(i, min(i + 3, len(em_lines))):
                    m3 = _RX_PHONE.search(em_lines[k])
                    if m3:
                        add("EMERGENCY_HOME_PHONE", _norm_phone(m3.group(0)))
                        break
            if re.search(r"work phone", ln, re.IGNORECASE):
                for k in range(i, min(i + 3, len(em_lines))):
                    m3 = _RX_PHONE.search(em_lines[k])
                    if m3:
                        add("EMERGENCY_WORK_PHONE", _norm_phone(m3.group(0)))
                        break

    # Insurance yes/no
    m = re.search(r"Do you have medical insurance\?\s*\n\s*(Yes|No)", text, flags=re.IGNORECASE)
    if m:
        add("HAS_INSURANCE", m.group(1).capitalize())

    # Hep B vaccination yes/no
    m = re.search(r"Hepatitis\s*B\s*vaccination\?\s*\n\s*(Yes|No)", text, flags=re.IGNORECASE)
    if m:
        add("HEP_B_VACCINATED", m.group(1).capitalize())

    # Insurance company / policy / expiry
    m = re.search(r"Name of Insurance Company:\s*\n\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        add("INSURANCE_COMPANY", m.group(1).strip())
    m = re.search(r"Policy Number:\s*\n\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        add("POLICY_NUMBER", m.group(1).strip())
    m = re.search(r"Expiry Date:\s*\n\s*(\d{1,2}/\d{1,2}/\d{4})", text, flags=re.IGNORECASE)
    if m:
        add("POLICY_EXPIRY_DATE", m.group(1).strip())

    # Allergies / regular meds (multi-line)
    m = _RX_ALLERGIES_FIELD.search(text)
    if m:
        items = [it.strip(" -•\t") for it in re.split(r"\n|\r|\s*\u2022\s*", m.group(1)) if it.strip()]
        if items:
            add("ALLERGIES_TEXT", "; ".join(items)[:800])
    m = _RX_MED_FIELD.search(text)
    if m:
        items = [it.strip(" -•\t") for it in re.split(r"\n|\r|\s*\u2022\s*", m.group(1)) if it.strip()]
        if items:
            add("MEDICATIONS_TEXT", "; ".join(items)[:800])

    # Medical problems
    m = re.search(r"(?is)List any Medical Problems[^:]*:\s*\n\s*(.+?)(?:\n\s*Name of Insurance Company:|\n\s*Medical Insurance Details\b|\Z)", text)
    if m:
        items = [it.strip(" -•\t") for it in re.split(r"\n|\r", m.group(1)) if it.strip()]
        if items:
            add("MEDICAL_PROBLEMS_TEXT", "; ".join(items)[:800])

    return out


def ner_entities(text: str, max_chars: int = 200_000) -> List[NEREntity]:
    """
    Best-effort NER:
      - Adds lightweight form-field entities first (custom labels).
      - If spaCy model available: returns spaCy entities (PERSON/ORG/DATE etc) after.
    """
    if not text:
        return []

    txt = text[:max_chars]

    out: List[NEREntity] = []
    seen = set()

    # 1) Form-field entities first
    for fe in _extract_form_entities(txt):
        key = (fe.label, (fe.text or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(fe)

    # 2) spaCy entities after
    if _nlp is None:
        return out

    doc = _nlp(txt)

    for ent in doc.ents:
        t = (ent.text or "").strip()
        if _looks_like_noise(t):
            continue

        label = ent.label_

        # Common template mislabels
        if label in {"CARDINAL", "QUANTITY"} and _RX_DATE_LIKE.fullmatch(t):
            label = "DATE"
        if label in {"DATE", "CARDINAL"} and _RX_ZIP5.fullmatch(t):
            label = "POSTCODE"
        if label == "PERSON" and any(k in t.lower() for k in [" lane", " drive", " trail", " avenue", " street", " suite", " rd", " road"]):
            label = "ADDRESS"

        # Reduce some obvious heading merges
        if label in {"ORG", "WORK_OF_ART"} and re.search(r"\b(Patient Medical Record|Patient Information|General Medical History)\b", t, re.IGNORECASE):
            continue

        key = (label, t.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(NEREntity(label=label, text=t, start_char=ent.start_char, end_char=ent.end_char))

    return out


def build_event_hints_from_text(text: str, event_dt: datetime, max_hits: int = 50) -> List[ExtractedEvent]:
    """
    Extract a SMALL number of structured “hints” from raw text.
    These are **not** production-grade clinical coding, but they let Stage-2/3 be exercised end-to-end.

    Returns events with `source` in {enc, obs, med, proc, imm, alg}.
    """
    if not text:
        return []

    events: List[ExtractedEvent] = []

    # Observations
    for m in _RX_BP.finditer(text):
        sys, dia = m.group(1), m.group(2)
        events.append(ExtractedEvent(
            source="obs",
            event_date=event_dt,
            description="blood_pressure",
            value=f"{sys}/{dia}",
            units="mmHg",
            type="vital",
            raw={"pattern": "bp"}
        ))
        if len(events) >= max_hits:
            return events

    for m in _RX_TEMP.finditer(text):
        val, unit = m.group(1), m.group(2)
        events.append(ExtractedEvent(
            source="obs",
            event_date=event_dt,
            description="temperature",
            value=str(val),
            units=unit,
            type="vital",
            raw={"pattern": "temp"}
        ))
        if len(events) >= max_hits:
            return events


    # Additional vitals/observations (demo)
    for rx, desc, unit_default, typ in [
        (_RX_HR, 'heart_rate', 'bpm', 'vital'),
        (_RX_O2, 'oxygen_saturation', '%', 'vital'),
        (_RX_GLU, 'glucose', 'mg/dL', 'lab'),
    ]:
        for m in rx.finditer(text):
            val = m.group(1)
            unit = (m.group(2) if m.lastindex and m.lastindex >= 2 else None) or unit_default
            events.append(ExtractedEvent(
                source='obs',
                event_date=event_dt,
                description=desc,
                value=str(val),
                units=unit,
                type=typ,
                raw={'pattern': desc}
            ))
            if len(events) >= max_hits:
                return events

    for m in _RX_WT.finditer(text):
        val, unit = m.group(1), m.group(2)
        events.append(ExtractedEvent(
            source='obs',
            event_date=event_dt,
            description='weight',
            value=str(val),
            units=unit,
            type='vital',
            raw={'pattern': 'weight'}
        ))
        if len(events) >= max_hits:
            return events

    
    # Weight without explicit units (common in intake templates)
    for m in _RX_WT_NOUNIT.finditer(text):
        val = m.group(1)
        events.append(ExtractedEvent(
            source='obs',
            event_date=event_dt,
            description='weight',
            value=str(val),
            units='kg',
            type='vital',
            raw={'pattern': 'weight_nounit'}
        ))
        if len(events) >= max_hits:
            return events

    for m in _RX_HT.finditer(text):
        val, unit = m.group(1), m.group(2)
        events.append(ExtractedEvent(
            source='obs',
            event_date=event_dt,
            description='height',
            value=str(val),
            units=unit,
            type='vital',
            raw={'pattern': 'height'}
        ))
        if len(events) >= max_hits:
            return events

    
    # Height without explicit units (common in intake templates)
    for m in _RX_HT_NOUNIT.finditer(text):
        val = m.group(1)
        events.append(ExtractedEvent(
            source='obs',
            event_date=event_dt,
            description='height',
            value=str(val),
            units='cm',
            type='vital',
            raw={'pattern': 'height_nounit'}
        ))
        if len(events) >= max_hits:
            return events

    for m in _RX_A1C.finditer(text):
        val = m.group(1)
        events.append(ExtractedEvent(
            source='obs',
            event_date=event_dt,
            description='hba1c',
            value=str(val),
            units='%',
            type='lab',
            raw={'pattern': 'a1c'}
        ))
        if len(events) >= max_hits:
            return events

    # Diagnoses (very naive keyword match)
    for rx, code, desc in _DIAG_KEYWORDS:
        m = rx.search(text)
        if m:
            # Avoid intake-form checkbox questions like "Asthma\nYes\nNo".
            ctx = text[max(0, m.start()-20): m.end()+50]
            if re.search(r"\bYes\b", ctx, re.IGNORECASE) and re.search(r"\bNo\b", ctx, re.IGNORECASE):
                continue
            # Skip examples/prompts, e.g., "List any Medical Problems (asthma, seizures...)"
            ctx2 = text[max(0, m.start()-100): m.end()+100]
            if re.search(r"\bList any\b", ctx2, re.IGNORECASE) and re.search(r"\bMedical Problems\b", ctx2, re.IGNORECASE):
                continue
            if re.search(r"\([^\)]{0,60}$", ctx2[:100], re.IGNORECASE) and re.search(r"^[^\(]{0,60}\)", ctx2[100:], re.IGNORECASE):
                continue
            events.append(ExtractedEvent(
                source='enc',
                event_date=event_dt,
                code=code,
                description=desc,
                raw={'pattern': 'diag'}
            ))
            if len(events) >= max_hits:
                return events

    # Immunization / immunity hints (common in intake forms)
    for m in _RX_IMMUNE.finditer(text):
        vaccine = (m.group(1) or "").strip()
        status = (m.group(2) or "").strip()
        events.append(ExtractedEvent(
            source='imm',
            event_date=event_dt,
            description=vaccine.lower().replace(" ", "_"),
            value=status.upper().replace(" ", "_"),
            type='immunity',
            raw={'pattern': 'immune'}
        ))
        if len(events) >= max_hits:
            return events

    m = _RX_HEPB.search(text)
    if m:
        status = (m.group(1) or "").strip()
        events.append(ExtractedEvent(
            source='imm',
            event_date=event_dt,
            description='hepatitis_b_vaccination',
            value=status.upper(),
            type='immunization',
            raw={'pattern': 'hepb'}
        ))
        if len(events) >= max_hits:
            return events

    # Allergy free-text field
    m = _RX_ALLERGIES_FIELD.search(text)
    if m and not _looks_like_noise(m.group(1)):
        v = re.sub(r"\s+", " ", m.group(1)).strip()[:800]
        events.append(ExtractedEvent(
            source='alg',
            event_date=event_dt,
            description='allergies_reported',
            value=v,
            type='allergy',
            raw={'pattern': 'alg'}
        ))
        if len(events) >= max_hits:
            return events

    # Regular medications free-text field (when not in "Drug 5 mg" pattern)
    m = _RX_MED_FIELD.search(text)
    if m and not _looks_like_noise(m.group(1)):
        v = re.sub(r"\s+", " ", m.group(1)).strip()[:800]
        events.append(ExtractedEvent(
            source='med',
            event_date=event_dt,
            description=v,
            raw={'pattern': 'med_field'}
        ))
        if len(events) >= max_hits:
            return events

    # Medications (very naive)
    for m in _RX_MED.finditer(text):
        name, dose, unit = m.group(1), m.group(2), m.group(3)
        events.append(ExtractedEvent(
            source="med",
            event_date=event_dt,
            description=f"{name} {dose}{unit}",
            raw={"pattern": "med"}
        ))
        if len(events) >= max_hits:
            return events

    # Procedures (string match)
    lowered = text.lower()
    for proc in _COMMON_PROCS:
        if proc in lowered:
            events.append(ExtractedEvent(
                source="proc",
                event_date=event_dt,
                description=proc,
                raw={"pattern": "proc"}
            ))
            if len(events) >= max_hits:
                return events

    return events

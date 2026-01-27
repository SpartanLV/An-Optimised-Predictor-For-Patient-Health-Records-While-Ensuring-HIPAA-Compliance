from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class NEREntity(BaseModel):
    label: str
    text: str
    start_char: int = 0
    end_char: int = 0
    confidence: Optional[float] = None


class ExtractedEvent(BaseModel):
    # Must match Stage-3 tokenization prefixes: enc/obs/med/proc/cp/imm/alg
    source: str = Field(..., examples=["enc", "obs", "med", "proc", "cp", "imm", "alg"])
    event_date: datetime

    code: Optional[str] = None
    description: Optional[str] = None

    value: Optional[str] = None
    units: Optional[str] = None
    type: Optional[str] = None

    reason_code: Optional[str] = None
    reason_description: Optional[str] = None

    raw: Optional[Dict[str, Any]] = None


class PreprocessResponse(BaseModel):
    document_sha256: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    patient_id: Optional[str] = None
    extracted_at_utc: datetime

    text: str
    entities: List[NEREntity] = []
    events: List[ExtractedEvent] = []

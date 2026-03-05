from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PatientCreate(BaseModel):
    mrn: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    mrn: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    created_at_utc: datetime


class NEREntityIn(BaseModel):
    label: str
    text: str
    start_char: int = 0
    end_char: int = 0
    confidence: Optional[float] = None


class NEREntityOut(NEREntityIn):
    model_config = ConfigDict(from_attributes=True)
    id: str
    document_id: str
    created_at_utc: datetime


class EventIn(BaseModel):
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


class EventOut(EventIn):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at_utc: datetime


class BulkEventsIn(BaseModel):
    events: List[EventIn]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    patient_id: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    sha256: Optional[str] = None
    created_at_utc: datetime


class ExtractionPayload(BaseModel):
    # Mirror preprocess response subset
    document_sha256: Optional[str] = None
    text: Optional[str] = None
    entities: List[NEREntityIn] = []
    events: List[EventIn] = []


class CohortCreate(BaseModel):
    name: str
    filters: Dict[str, Any] = {}


class CohortOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    created_by: Optional[str] = None
    filters: Dict[str, Any] = {}
    created_at_utc: datetime

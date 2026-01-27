from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


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


class PredictRequest(BaseModel):
    patient_id: Optional[str] = None
    events: Optional[List[EventIn]] = None

    as_of: Optional[datetime] = None
    top_n: int = 10
    return_debug: bool = False


class RecommendationItem(BaseModel):
    item: str
    count: int


class RecommendationsOut(BaseModel):
    medications: List[RecommendationItem] = []
    procedures: List[RecommendationItem] = []
    careplans: List[RecommendationItem] = []


class PredictionOut(BaseModel):
    code: str
    description: Optional[str] = None
    probability: float
    recommendations: Optional[RecommendationsOut] = None


class PredictResponse(BaseModel):
    threshold: float
    predictions: List[PredictionOut]
    # optional debug
    debug: Optional[Dict[str, Any]] = None

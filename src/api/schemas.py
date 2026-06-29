"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel
from typing import Optional, List


class DiagnosisRequest(BaseModel):
    image_url: Optional[str] = None
    patient_id: Optional[str] = None


class DiagnosisResponse(BaseModel):
    wound_detected: bool
    wound_area_pixels: int
    wound_area_percentage: float
    bbox: Optional[List[int]] = None
    confidence: float


class PatientResponse(BaseModel):
    patient_id: str
    total_evaluations: int

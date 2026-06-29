"""Patient CRUD endpoint."""
from fastapi import APIRouter

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/{patient_id}")
async def get_patient(patient_id: str):
    return {"patient_id": patient_id, "total_evaluations": 0}

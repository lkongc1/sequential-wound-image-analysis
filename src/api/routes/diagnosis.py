"""Diagnosis endpoint for wound segmentation."""
from fastapi import APIRouter
from starlette.requests import Request

router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])


@router.post("/")
async def diagnose(request: Request):
    return {"status": "ok"}

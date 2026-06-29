"""FastAPI application entry point."""
from fastapi import FastAPI

from src.api.routes.diagnosis import router as diagnosis_router
from src.api.routes.health import router as health_router
from src.api.routes.patients import router as patients_router

app = FastAPI(
    title="Wound Segmentation API",
    description="Medical AI system for wound detection and tracking",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(diagnosis_router)
app.include_router(patients_router)

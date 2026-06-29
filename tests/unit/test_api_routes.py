"""Unit tests for API router registration in src.api.main.

Run with: pytest tests/unit/test_api_routes.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health router registration."""

    def test_health_returns_ok_and_version(self, client):
        """GET /health returns status ok and version from the router."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestDiagnosisEndpoint:
    """Tests for /diagnosis router registration."""

    def test_diagnosis_post_returns_ok(self, client):
        """POST /diagnosis/ returns a status response."""
        response = client.post("/diagnosis/", params={"request": "test"})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestPatientsEndpoint:
    """Tests for /patients router registration."""

    def test_get_patient_returns_id(self, client):
        """GET /patients/{id} returns the patient_id."""
        response = client.get("/patients/123")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "123"


class TestDocsAvailability:
    """Tests that /docs and OpenAPI schema are reachable."""

    def test_openapi_schema_available(self, client):
        """GET /openapi.json returns 200 and contains registered paths."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/health" in paths
        assert "/diagnosis/" in paths
        assert "/patients/{patient_id}" in paths

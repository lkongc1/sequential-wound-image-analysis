"""Integration tests for FastAPI application endpoints.

Run with: pytest tests/integration/test_api.py -v
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
    """Tests for /health."""

    def test_health_returns_200_and_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestDocsEndpoint:
    """Tests for OpenAPI documentation."""

    def test_docs_returns_200(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_schema_present(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/health" in paths
        assert "/diagnosis/" in paths
        assert "/patients/{patient_id}" in paths


class TestDiagnosisEndpoint:
    """Tests for /diagnosis router."""

    def test_diagnosis_post_returns_200(self, client):
        """POST /diagnosis/ should not crash (stub implementation)."""
        response = client.post("/diagnosis/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestPatientsEndpoint:
    """Tests for /patients router."""

    def test_get_patient_returns_200_and_id(self, client):
        """GET /patients/{id} should return the requested patient_id."""
        response = client.get("/patients/123")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "123"
        assert "total_evaluations" in data

    def test_get_patient_with_different_id(self, client):
        response = client.get("/patients/456")
        assert response.status_code == 200
        assert response.json()["patient_id"] == "456"

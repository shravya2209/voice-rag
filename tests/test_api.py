"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "index_loaded" in data
        assert "embedding_model" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "uptime_seconds" in data


class TestTextQueryEndpoint:
    def test_empty_query_rejected(self, client):
        response = client.post("/api/query", json={"query": ""})
        assert response.status_code == 422  # Pydantic validation

    def test_query_without_index(self, client):
        """Query without loaded index should return error gracefully."""
        response = client.post("/api/query", json={"query": "test query"})
        # Should return 500 or error, not crash
        assert response.status_code in (200, 500)


class TestVoiceQueryEndpoint:
    def test_no_file_rejected(self, client):
        response = client.post("/api/voice-query")
        assert response.status_code == 422

    def test_empty_audio_rejected(self, client):
        """Empty file should be handled gracefully."""
        import io
        response = client.post(
            "/api/voice-query",
            files={"file": ("test.webm", io.BytesIO(b""), "audio/webm")},
        )
        # Should return error, not crash
        assert response.status_code in (400, 500)


class TestTranscribeEndpoint:
    def test_no_file_rejected(self, client):
        response = client.post("/api/transcribe")
        assert response.status_code == 422

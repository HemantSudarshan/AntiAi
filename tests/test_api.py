"""
API Tests for TruthTracker
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_client():
    """Get test client - import here to avoid startup issues"""
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_returns_200(self):
        """Health check should return 200"""
        client = get_client()
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self):
        """Health check should include status field"""
        client = get_client()
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestInfoEndpoint:
    """Tests for /api/v1/info endpoint"""
    
    def test_info_returns_200(self):
        """Info endpoint should return 200"""
        client = get_client()
        response = client.get("/api/v1/info")
        assert response.status_code == 200
    
    def test_info_contains_version(self):
        """Info should contain version"""
        client = get_client()
        response = client.get("/api/v1/info")
        data = response.json()
        assert "version" in data


class TestAnalyzeNewsEndpoint:
    """Tests for /api/v1/analyze-news endpoint"""
    
    def test_analyze_news_valid_text(self):
        """Should analyze valid news text"""
        client = get_client()
        response = client.post(
            "/api/v1/analyze-news",
            data={"text": "According to Reuters, officials confirmed the new policy changes."}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["REAL", "FAKE"]
    
    def test_analyze_news_short_text_fails(self):
        """Should reject text that's too short"""
        client = get_client()
        response = client.post(
            "/api/v1/analyze-news",
            data={"text": "Short"}
        )
        assert response.status_code == 400
    
    def test_analyze_news_returns_confidence(self):
        """Should return confidence score"""
        client = get_client()
        response = client.post(
            "/api/v1/analyze-news",
            data={"text": "Breaking news! You won't believe what happened next!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1


class TestAnalyzeImageEndpoint:
    """Tests for /api/v1/analyze-image endpoint"""
    
    def test_analyze_image_invalid_type(self):
        """Should reject non-image files"""
        client = get_client()
        response = client.post(
            "/api/v1/analyze-image",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

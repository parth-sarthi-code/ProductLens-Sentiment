import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "available_experts" in data


def test_api_analyze():
    payload = {
        "text": "The laptop screen is stunning, but battery life is poor.",
        "category": "Electronics",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["aspect_count"] >= 1
    assert len(data["results"]) >= 1
    assert data["results"][0]["sentiment"] in ["positive", "neutral", "negative"]

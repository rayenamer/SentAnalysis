
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_valid_analysis_request():
    response = client.post(
        "/analyze/validate",
        json={
            "documents": [
                {
                    "document_id": "doc-001",
                    "title": "Apple earnings",
                    "text": "Apple reports strong profits",
                    "tickers": ["AAPL"]
                }
            ],
            "model": "vader"
        }
    )

    assert response.status_code == 200
    assert response.json()["documents_count"] == 1


def test_invalid_analysis_request():
    response = client.post(
        "/analyze/validate",
        json={"documents": []}
    )

    assert response.status_code == 422
    assert "detail" in response.json()

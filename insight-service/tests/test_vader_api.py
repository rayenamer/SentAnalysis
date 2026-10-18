import pytest
from fastapi.testclient import TestClient

from app.main import app as original_app
from app.models.sentiment import AnalyzeResponse
from app.vader_api import app

client = TestClient(app)


def document(index=0, text="Excellent profits and strong growth!"):
    return {"document_id": str(index), "title": "Earnings", "text": text,
            "tickers": ["AAPL"]}


def test_analyze_batch_uses_existing_response_contract():
    response = client.post("/analyze", json={"documents": [
        document(), document(1, "Terrible losses and disastrous results."),
        document(2, "The report was filed on Tuesday."),
    ]})
    assert response.status_code == 200
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.schema_version == "1.0"
    assert [r.document_id for r in parsed.results] == ["0", "1", "2"]
    assert [r.label for r in parsed.results] == ["positive", "negative", "neutral"]
    assert all(r.model == "vader" and r.tickers == ["AAPL"]
               and r.confidence is None for r in parsed.results)


@pytest.mark.parametrize("payload", [
    {}, {"documents": []}, {"documents": [document(text="")]},
    {"documents": [document()], "model": "unknown"},
    {"documents": [document(i) for i in range(101)]},
])
def test_analyze_rejects_invalid_contracts(payload):
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_maximum_batch_size():
    response = client.post("/analyze", json={
        "documents": [document(i) for i in range(100)], "model": "vader",
    })
    assert response.status_code == 200
    assert len(response.json()["results"]) == 100


def test_finbert_returns_not_implemented():
    response = client.post("/analyze", json={
        "documents": [document()], "model": "finbert",
    })
    assert response.status_code == 501


def test_existing_routes_are_available_and_original_app_is_unchanged():
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/analyze/validate", json={
        "documents": [document()],
    }).json() == {"status": "valid", "documents_count": 1, "model": "vader"}
    assert "/analyze" not in {route.path for route in original_app.routes}
    schema = client.get("/openapi.json").json()
    assert "AnalyzeResponse" in schema["components"]["schemas"]
    assert "/analyze" in schema["paths"]

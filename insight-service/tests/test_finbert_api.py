from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import finbert_api
from app.finbert_models import FinBertResponse, FinBertResult, SentimentProbabilities
from app.models.sentiment import AnalyzeResponse
from app.vader_api import app as vader_app

client = TestClient(finbert_api.app)
document = {"document_id": "earnings-1", "title": "Earnings",
            "text": "Strong profits", "tickers": ["AAPL"]}


def test_finbert_response_retains_probabilities(monkeypatch):
    service = Mock()
    service.analyze.return_value = FinBertResponse(results=[FinBertResult(
        document_id="earnings-1", label="positive", confidence=0.8,
        sentiment_score=0.75, tickers=["AAPL"],
        probabilities=SentimentProbabilities(positive=0.8, negative=0.05, neutral=0.15),
    )])
    monkeypatch.setattr(finbert_api, "get_finbert_service", lambda: service)
    response = client.post("/analyze", json={"documents": [document], "model": "finbert"})
    assert response.status_code == 200
    assert response.json()["results"][0]["probabilities"]["positive"] == 0.8
    assert AnalyzeResponse.model_validate(response.json()).results[0].model == "finbert"
    assert service.analyze.call_args.args[0].documents[0].document_id == "earnings-1"


def test_vader_is_unchanged_and_does_not_load_finbert(monkeypatch):
    loader = Mock(side_effect=AssertionError("FinBERT must stay unloaded"))
    monkeypatch.setattr(finbert_api, "get_finbert_service", loader)
    payload = {"documents": [document]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    assert response.json() == TestClient(vader_app).post("/analyze", json=payload).json()
    loader.assert_not_called()


def test_finbert_failure_returns_503_without_vader_fallback(monkeypatch):
    monkeypatch.setattr(finbert_api, "get_finbert_service", Mock(side_effect=RuntimeError("failure")))
    vader = Mock(side_effect=AssertionError("no fallback"))
    monkeypatch.setattr(finbert_api, "analyze_documents", vader)
    response = client.post("/analyze", json={"documents": [document], "model": "finbert"})
    assert response.status_code == 503
    assert response.json() == {"detail": "FinBERT CPU inference is unavailable"}
    vader.assert_not_called()


def test_validation_and_existing_routes():
    assert client.post("/analyze", json={"documents": []}).status_code == 422
    assert client.get("/health").status_code == 200
    assert client.post("/analyze/validate", json={"documents": [document], "model": "finbert"}).status_code == 200
    schema = client.get("/openapi.json").json()
    assert "FinBertResponse" in schema["components"]["schemas"]
    assert len([r for r in finbert_api.app.routes if getattr(r, "path", None) == "/analyze"]) == 1

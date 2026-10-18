from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import batch_analysis
from app.extensions_api import app
from app.models.sentiment import AnalyzeResponse, SentimentResult
from app.finbert_models import FinBertResult, FinBertResponse, SentimentProbabilities

client = TestClient(app)


def documents(count=2):
    return [dict(document_id=str(i), title="Earnings", text="Earnings statement", tickers=["AAPL"])
            for i in range(count)]


def inference(request):
    values = [dict(document_id=d.document_id, model=request.model, label="neutral",
                   sentiment_score=0, tickers=d.tickers) for d in request.documents]
    if request.model == "finbert":
        return FinBertResponse(results=[FinBertResult(**v, confidence=0.8,
            probabilities=SentimentProbabilities(positive=0.1, negative=0.1, neutral=0.8)) for v in values])
    return AnalyzeResponse(results=[SentimentResult(**v) for v in values])


@pytest.mark.parametrize("model", ["vader", "finbert"])
def test_batch_api_uses_existing_models_and_preserves_results(monkeypatch, model):
    monkeypatch.setattr(batch_analysis, "_run_model", inference)
    response = client.post("/analyze/batch", json=dict(documents=documents(), model=model))
    assert response.status_code == 200
    body = response.json()
    assert body["success_count"] == 2 and body["error_count"] == 0
    assert [o["document_id"] for o in body["outcomes"]] == ["0", "1"]
    assert all(o["result"]["tickers"] == ["AAPL"] for o in body["outcomes"])
    if model == "finbert":
        assert body["outcomes"][0]["result"]["probabilities"]["neutral"] == 0.8


def test_document_errors_remain_explicit_in_api_response(monkeypatch):
    def infer(request):
        if any(d.document_id == "1" for d in request.documents):
            raise RuntimeError("failure")
        return inference(request)
    monkeypatch.setattr(batch_analysis, "_run_model", infer)
    response = client.post("/analyze/batch", json=dict(documents=documents(3)))
    assert response.status_code == 200
    assert [o["status"] for o in response.json()["outcomes"]] == ["success", "error", "success"]
    assert response.json()["outcomes"][1]["error"]["code"] == "inference_failed"


@pytest.mark.parametrize("changes", [dict(documents=[]), dict(documents=documents(101)),
    dict(chunk_size=0), dict(chunk_size=21), dict(model="unknown"),
    dict(documents=[dict(document_id="bad", title="Report", text="")])])
def test_invalid_batch_does_not_run_inference(monkeypatch, changes):
    runner = Mock(side_effect=AssertionError("invalid input must not infer"))
    monkeypatch.setattr(batch_analysis, "_run_model", runner)
    payload = dict(documents=documents()); payload.update(changes)
    assert client.post("/analyze/batch", json=payload).status_code == 422
    runner.assert_not_called()


def weighted_payload():
    return dict(as_of="2026-10-18T12:00:00Z", results=[dict(
        document_id="one", model="vader", label="positive", sentiment_score=0.8,
        tickers=["AAPL", "MSFT"], published_at="2026-10-18T12:00:00Z",
        source="wire", engagement=10)],
        weighting=dict(source_weights={"wire": 2}, engagement_factor=0.1))


def test_weighted_api_exposes_weight_and_raw_statistics():
    response = client.post("/aggregate/weighted", json=weighted_payload())
    assert response.status_code == 200
    value = response.json()["windows"][0]["aggregates"][0]
    assert value["article_count"] == 1 and value["total_weight"] == 4
    assert value["weighted_mean_sentiment"] == value["mean_sentiment"] == 0.8
    assert value["weighted_distribution"]["positive"] == 1


@pytest.mark.parametrize("change", ["engagement", "source_weight", "conflict", "naive"])
def test_invalid_weighted_data(change):
    data = weighted_payload()
    if change == "engagement":
        data["results"][0]["engagement"] = -1
    elif change == "source_weight":
        data["weighting"]["source_weights"]["wire"] = -1
    elif change == "conflict":
        data["results"].append(dict(data["results"][0], engagement=11))
    else:
        data["as_of"] = "2026-10-18T12:00:00"
    assert client.post("/aggregate/weighted", json=data).status_code == 422


def test_empty_weighted_dataset_and_zero_weight_response():
    assert client.post("/aggregate/weighted", json=dict(results=[])).status_code == 200
    data = weighted_payload(); data["weighting"]["source_weights"]["wire"] = 0
    response = client.post("/aggregate/weighted", json=data)
    assert response.status_code == 200
    assert response.json()["windows"][0]["aggregates"][0]["weighted_mean_sentiment"] is None


def test_all_routes_are_available_without_duplicate_registration():
    schema = client.get("/openapi.json").json()
    for path in ["/analyze", "/analyze/batch", "/aggregate", "/aggregate/weighted", "/health"]:
        assert path in schema["paths"]
    assert client.get("/health").status_code == 200
    response = client.post("/aggregate", json=dict(results=[]))
    assert response.status_code == 200
    assert "weighting" not in response.json()

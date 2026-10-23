import json

import pytest
from fastapi.testclient import TestClient

from app.correlation_api import app
from app.correlation_models import CorrelationResponse

client = TestClient(app)


def payload():
    return dict(time_bucket="daily", lags=[0], sentiments=[
        dict(ticker="AAPL", model="finbert", aggregated_sentiment=s,
             timestamp=f"2026-10-{day:02d}T00:00:00Z") for day, s in [(21, -.5), (22, 0), (23, .5)]],
        prices=[dict(ticker="AAPL", close_price=p, timestamp=f"2026-10-{day:02d}T00:00:00Z")
                for day, p in [(20, 100), (21, 50), (22, 50), (23, 75)]])


def test_valid_api_returns_correlation_contract_and_warnings():
    response = client.post("/signals/correlation", json=payload())
    assert response.status_code == 200
    body = CorrelationResponse.model_validate(response.json())
    assert body.results[0].pearson_r == pytest.approx(1)
    assert body.results[0].paired_observations == 3
    assert body.results[0].correlation_method == "pearson"
    assert "Autocorrelated" in body.warnings[0]
    json.dumps(response.json(), allow_nan=False)


@pytest.mark.parametrize("kind,changes", [("prices", dict(close_price=0)),
    ("prices", dict(close_price=-1)), ("prices", dict(timestamp="bad")),
    ("sentiments", dict(aggregated_sentiment=2)),
    ("sentiments", dict(timestamp="2026-10-21T00:00:00")),
    ("sentiments", dict(timestamp="2026-10-21T00:30:00Z"))])
def test_invalid_data_returns_422(kind, changes):
    data = payload(); data[kind][0].update(changes)
    assert client.post("/signals/correlation", json=data).status_code == 422


@pytest.mark.parametrize("changes", [dict(lags=[]), dict(lags=[0, 0]), dict(min_pairs=1),
    dict(time_bucket="weekly"), dict(start_timestamp="2026-11-01T00:00:00Z",
                                     end_timestamp="2026-10-01T00:00:00Z")])
def test_invalid_configuration_returns_422(changes):
    data = payload(); data.update(changes)
    assert client.post("/signals/correlation", json=data).status_code == 422


def test_duplicate_prices_and_sentiment_conflicts():
    data = payload(); data["prices"].append(dict(data["prices"][0]))
    response = client.post("/signals/correlation", json=data)
    assert response.status_code == 200
    assert response.json()["diagnostics"]["price_duplicates_removed"] == 1
    data["prices"][-1]["close_price"] = 200
    assert client.post("/signals/correlation", json=data).status_code == 422
    data = payload(); data["sentiments"].append(dict(data["sentiments"][0], aggregated_sentiment=1))
    assert client.post("/signals/correlation", json=data).status_code == 422


def test_empty_missing_and_constant_cases_are_null_not_zero():
    response = client.post("/signals/correlation", json=dict(sentiments=[], prices=[]))
    assert response.status_code == 200 and response.json()["results"] == []
    assert client.post("/signals/correlation", json={}).status_code == 422
    data = payload()
    for row in data["sentiments"]:
        row["aggregated_sentiment"] = 0
    result = client.post("/signals/correlation", json=data).json()["results"][0]
    assert result["pearson_r"] is None and result["diagnostic_reason"] == "constant_sentiment"
    data["prices"] = []
    result = client.post("/signals/correlation", json=data).json()["results"][0]
    assert result["pearson_r"] is None and result["diagnostic_reason"] == "insufficient_pairs"


def test_mixed_weighting_and_horizons_are_distinct_api_series():
    data = payload()
    data["sentiments"] += [dict(r, score_kind="weighted", aggregation_window="7d")
                           for r in list(data["sentiments"])]
    response = client.post("/signals/correlation", json=data)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert {r["score_kind"] for r in response.json()["results"]} == {"raw", "weighted"}


def test_existing_endpoints_remain_accessible_and_no_inference(monkeypatch):
    from app import finbert
    monkeypatch.setattr(finbert, "_load_backend", lambda *_: pytest.fail("must not load model"))
    assert client.post("/signals/correlation", json=payload()).status_code == 200
    schema = client.get("/openapi.json").json()
    for path in ["/health", "/analyze", "/analyze/batch", "/aggregate", "/aggregate/weighted",
                 "/signals/velocity", "/signals/outliers", "/signals/correlation"]:
        assert path in schema["paths"]
    assert "CorrelationResponse" in schema["components"]["schemas"]
    assert client.get("/health").status_code == 200
    assert client.post("/signals/outliers", json=dict(observations=[])).status_code == 200
    assert client.post("/signals/velocity", json=dict(observations=[])).status_code == 200

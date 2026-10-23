import json

import pytest
from fastapi.testclient import TestClient

from app.outliers_api import app
from app.outliers_models import OutliersResponse

client = TestClient(app)


def payload():
    return dict(min_history=3, observations=[dict(ticker="AAPL", model="finbert",
        sentiment_score=score, observed_at=f"2026-10-23T{hour}:00:00Z")
        for hour, score in [("10", -0.125), ("11", 0), ("12", 0.125), ("13", 0.375)]])


def test_api_returns_strong_positive_outlier_and_valid_json():
    response = client.post("/signals/outliers", json=payload())
    assert response.status_code == 200
    result = OutliersResponse.model_validate(response.json())
    final = result.results[-1]
    assert final.z_score == 3 and final.historical_std == 0.125
    assert final.severity == "strong_outlier" and final.anomaly_direction == "positive"
    assert result.std_ddof == 1
    json.dumps(response.json(), allow_nan=False)


def test_empty_and_missing_input():
    assert client.post("/signals/outliers", json=dict(observations=[])).json()["results"] == []
    assert client.post("/signals/outliers", json={}).status_code == 422
    data = dict(observations=[dict(ticker="AAPL", model="vader")])
    response = client.post("/signals/outliers", json=data)
    assert response.status_code == 200
    assert response.json()["results"][0]["diagnostic_reason"] == "missing_timestamp"


@pytest.mark.parametrize("changes", [dict(sentiment_score=2), dict(model="unknown"),
    dict(observed_at="2026-10-23T10:00:00"), dict(observed_at="bad"), dict(ticker="")])
def test_invalid_observations_return_422(changes):
    data = payload(); data["observations"][0].update(changes)
    assert client.post("/signals/outliers", json=data).status_code == 422


@pytest.mark.parametrize("changes", [dict(min_history=1), dict(lookback_hours=-1),
    dict(strong_threshold=2), dict(moderate_threshold=-1)])
def test_invalid_configuration_returns_422(changes):
    data = payload(); data.update(changes)
    assert client.post("/signals/outliers", json=data).status_code == 422


def test_duplicates_and_context_conflicts():
    data = payload(); data["observations"].append(dict(data["observations"][0]))
    assert client.post("/signals/outliers", json=data).json()["diagnostics"]["duplicates_removed"] == 1
    data["observations"][-1]["sentiment_score"] = 0.9
    assert client.post("/signals/outliers", json=data).status_code == 422
    data = payload(); data["observations"][-1]["score_kind"] = "weighted"
    assert client.post("/signals/outliers", json=data).status_code == 422


def test_zero_variance_is_null_not_a_false_normal_result():
    data = payload()
    for row in data["observations"]:
        row["sentiment_score"] = 0.1
    response = client.post("/signals/outliers", json=data)
    final = response.json()["results"][-1]
    assert final["z_score"] is None and final["anomaly_detected"] is None
    assert final["diagnostic_reason"] == "zero_variance"


def test_existing_routes_preserved_and_no_inference(monkeypatch):
    from app import finbert
    monkeypatch.setattr(finbert, "_load_backend", lambda *_: pytest.fail("must not infer"))
    assert client.post("/signals/outliers", json=payload()).status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in ["/health", "/analyze", "/analyze/batch", "/aggregate", "/aggregate/weighted",
                 "/signals/velocity", "/signals/outliers"]:
        assert path in paths
    assert "/signals/correlation" not in paths
    assert client.get("/health").status_code == 200
    assert client.post("/aggregate", json=dict(results=[])).status_code == 200

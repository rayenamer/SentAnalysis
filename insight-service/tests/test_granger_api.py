import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import granger
from app.granger_api import app
from app.granger_models import GrangerResponse, StationarityDiagnostic

client = TestClient(app)


def payload():
    return dict(time_bucket="daily", sentiments=[dict(ticker="AAPL", model="vader",
        timestamp="2026-10-23T00:00:00Z", aggregated_sentiment=0.2)],
        prices=[dict(ticker="AAPL", timestamp="2026-10-22T00:00:00Z", close_price=100),
                dict(ticker="AAPL", timestamp="2026-10-23T00:00:00Z", close_price=101)])


def test_valid_but_insufficient_data_has_null_inference_and_all_requested_orders():
    response = client.post("/signals/granger", json=payload())
    assert response.status_code == 200
    body = GrangerResponse.model_validate(response.json())
    assert [r.lag for r in body.results] == [1, 2, 3]
    assert all(r.observation_count == 1 and r.p_value is None for r in body.results)
    assert all(r.diagnostic_reason == "insufficient_observations" for r in body.results)
    assert body.warnings
    json.dumps(response.json(), allow_nan=False)


@pytest.mark.parametrize("changes", [dict(max_lag=0), dict(max_lag=13), dict(alpha=0),
    dict(directions=["unknown"]), dict(min_observations=1)])
def test_invalid_config_returns_422(changes):
    data = payload(); data.update(changes)
    assert client.post("/signals/granger", json=data).status_code == 422


@pytest.mark.parametrize("kind,changes", [("prices", dict(close_price=0)),
    ("sentiments", dict(aggregated_sentiment=2)),
    ("sentiments", dict(timestamp="2026-10-23T00:00:00"))])
def test_invalid_data_returns_422(kind, changes):
    data = payload(); data[kind][0].update(changes)
    assert client.post("/signals/granger", json=data).status_code == 422


def test_empty_request_contract_and_reverse_directions():
    assert client.post("/signals/granger", json={}).status_code == 422
    assert client.post("/signals/granger", json=dict(sentiments=[], prices=[])).json()["results"] == []
    data = payload(); data["directions"] = ["sentiment_to_returns", "returns_to_sentiment"]
    assert len(client.post("/signals/granger", json=data).json()["results"]) == 6


def test_conflicting_price_duplicate_is_rejected():
    data = payload(); data["prices"].append(dict(data["prices"][0], close_price=200))
    assert client.post("/signals/granger", json=data).status_code == 422


def test_all_existing_routes_and_schema_are_preserved_without_loading_finbert(monkeypatch):
    from app import finbert
    monkeypatch.setattr(finbert, "_load_backend", Mock(side_effect=AssertionError("must not infer")))
    assert client.post("/signals/granger", json=payload()).status_code == 200
    schema = client.get("/openapi.json").json()
    for path in ["/health", "/analyze", "/analyze/batch", "/aggregate", "/aggregate/weighted",
                 "/signals/velocity", "/signals/outliers", "/signals/correlation", "/signals/granger"]:
        assert path in schema["paths"]
    assert "GrangerResponse" in schema["components"]["schemas"]
    assert client.post("/signals/correlation", json=dict(sentiments=[], prices=[])).status_code == 200
    assert client.get("/health").status_code == 200


def eligible_payload():
    start = datetime(2026, 10, 23, tzinfo=timezone.utc)
    prices, sentiments = [], []
    close = 100
    for i in range(13):
        timestamp = (start+timedelta(hours=i)).isoformat()
        close *= 1 + (i % 5) * 0.001
        prices.append(dict(ticker="AAPL", timestamp=timestamp, close_price=close))
        if i:
            sentiments.append(dict(ticker="AAPL", model="vader", timestamp=timestamp,
                                    aggregated_sentiment=0.1 * ((i % 7)-3)))
    return dict(time_bucket="hourly", min_observations=8, sentiments=sentiments, prices=prices)


def test_api_eligible_input_reports_missing_dependency_without_invented_p_values(monkeypatch):
    monkeypatch.setattr(granger, "_backend", Mock(side_effect=granger.StatisticsUnavailable("missing")))
    response = client.post("/signals/granger", json=eligible_payload())
    assert response.status_code == 200
    assert all(r["diagnostic_reason"] == "dependency_unavailable" for r in response.json()["results"])
    assert all(r["f_statistic"] is r["p_value"] is r["reject_null_hypothesis"] is None
               for r in response.json()["results"])


def test_api_serializes_mocked_inference_for_both_directions(monkeypatch):
    monkeypatch.setattr(granger, "_backend", lambda: SimpleNamespace(np=np))
    monkeypatch.setattr(granger, "_adf", lambda *_: StationarityDiagnostic(
        statistic=-5, p_value=0.001, reject_unit_root=True, diagnostic_reason=None))
    monkeypatch.setattr(granger, "_fit", lambda *_: (4, 0.01, {}, None))
    data = eligible_payload(); data["directions"] = ["sentiment_to_returns", "returns_to_sentiment"]
    response = client.post("/signals/granger", json=data)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 6
    assert all(r["p_value"] == 0.01 and r["reject_null_hypothesis"] is True for r in results)

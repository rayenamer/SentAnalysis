from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.aggregation_models import AggregationRequest
from app.datetime_utils import normalize_utc
from app.granger_api import app

client = TestClient(app, raise_server_exceptions=False)
OVERFLOWS = ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59.999999-01:00"]


def record(timestamp):
    return dict(document_id="1", model="vader", label="neutral", sentiment_score=0,
                tickers=["AAPL"], published_at=timestamp)


@pytest.mark.parametrize("timestamp", OVERFLOWS)
@pytest.mark.parametrize("endpoint", ["/aggregate", "/aggregate/weighted"])
@pytest.mark.parametrize("field", ["as_of", "published_at"])
def test_aggregation_utc_overflow_is_a_structured_422(timestamp, endpoint, field):
    payload = dict(results=[])
    if field == "as_of":
        payload["as_of"] = timestamp
    else:
        payload["results"] = [record(timestamp)]
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field
    assert "Traceback" not in response.text


@pytest.mark.parametrize("timestamp", OVERFLOWS)
@pytest.mark.parametrize("endpoint", ["/signals/velocity", "/signals/outliers",
    "/signals/correlation", "/signals/granger"])
def test_signal_observation_utc_overflow_remains_422(timestamp, endpoint):
    row = dict(ticker="AAPL", model="vader", sentiment_score=0, observed_at=timestamp)
    payload = dict(observations=[row]) if endpoint in {"/signals/velocity", "/signals/outliers"} else dict(sentiments=[row], prices=[])
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422 and "detail" in response.json()


@pytest.mark.parametrize("timestamp", OVERFLOWS)
@pytest.mark.parametrize("endpoint", ["/signals/correlation", "/signals/granger"])
@pytest.mark.parametrize("field", ["timestamp", "start_timestamp", "end_timestamp"])
def test_market_timestamp_and_analysis_bounds_use_safe_utc_conversion(timestamp, endpoint, field):
    payload = dict(sentiments=[], prices=[])
    if field == "timestamp":
        payload["prices"] = [dict(ticker="AAPL", timestamp=timestamp, close_price=100)]
    else:
        payload[field] = timestamp
    assert client.post(endpoint, json=payload).status_code == 422


@pytest.mark.parametrize("timestamp", ["0001-01-01T00:00:00Z", "0001-01-01T00:00:00-01:00",
    "9999-12-31T23:59:59.999999Z", "9999-12-31T23:59:59.999999+01:00"])
def test_representable_publication_boundaries_remain_valid(timestamp):
    response = client.post("/aggregate", json=dict(results=[record(timestamp)],
        as_of="2026-10-23T12:00:00Z"))
    assert response.status_code == 200


@pytest.mark.parametrize("timestamp", ["0001-01-01T01:00:00Z", "9999-12-31T23:59:59.999999Z"])
def test_reference_boundaries_that_allow_requested_window_remain_valid(timestamp):
    assert client.post("/aggregate", json=dict(results=[], windows=["1h"], as_of=timestamp)).status_code == 200


def test_utc_minimum_conversion_is_valid_but_underflowing_window_remains_invalid():
    assert normalize_utc(datetime.min.replace(tzinfo=timezone.utc)) == datetime.min.replace(tzinfo=timezone.utc)
    response = client.post("/aggregate", json=dict(results=[], windows=["1h"], as_of="0001-01-01T00:00:00Z"))
    assert response.status_code == 422


def test_ordinary_offsets_and_missing_publication_time_keep_existing_behavior():
    request = AggregationRequest(as_of="2026-10-23T14:00:00+02:00", results=[record(None)])
    assert request.as_of == datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
    assert request.results[0].published_at is None
    assert request.as_of.utcoffset() == timedelta(0)

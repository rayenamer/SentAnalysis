import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.aggregation_api import app
from app.aggregation_models import AggregationResponse

client = TestClient(app)
AS_OF = "2026-10-18T12:00:00Z"


def payload():
    return {"as_of": AS_OF, "results": [{
        "document_id": "article-1", "model": "vader", "label": "positive",
        "sentiment_score": 0.7, "tickers": ["aapl", "MSFT"], "published_at": AS_OF,
    }]}


def test_endpoint_calculations_and_response_contract():
    response = client.post("/aggregate", json=payload())
    assert response.status_code == 200
    parsed = AggregationResponse.model_validate(response.json())
    assert parsed.schema_version == "1.0"
    assert [w.window for w in parsed.windows] == ["1h", "24h", "7d"]
    for window in parsed.windows:
        assert [a.ticker for a in window.aggregates] == ["AAPL", "MSFT"]
        for row in window.aggregates:
            assert row.mean_sentiment == 0.7
            assert row.article_count == 1
            assert row.distribution.model_dump() == dict(positive=1, neutral=0, negative=0)


def test_empty_input():
    response = client.post("/aggregate", json={"results": [], "as_of": AS_OF})
    assert response.status_code == 200
    assert all(w["aggregates"] == [] for w in response.json()["windows"])


@pytest.mark.parametrize("changes", [{"results": None}, {"windows": []},
    {"windows": ["2h"]}, {"windows": ["1h", "1h"]},
    {"as_of": "2026-10-18T12:00:00"}, {"as_of": "bad-date"}])
def test_invalid_requests_return_422(changes):
    data = payload(); data.update(changes)
    response = client.post("/aggregate", json=data)
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize("changes", [{"sentiment_score": -1.1}, {"label": "unknown"},
    {"model": "unknown"}, {"published_at": "2026-10-18T12:00:00"},
    {"published_at": "bad-date"}, {"tickers": [""]}, {"document_id": ""}])
def test_invalid_results_return_422(changes):
    data = payload(); data["results"][0].update(changes)
    assert client.post("/aggregate", json=data).status_code == 422


def test_duplicate_and_conflict_handling():
    data = payload()
    data["results"].append(dict(data["results"][0]))
    response = client.post("/aggregate", json=data)
    assert response.status_code == 200
    assert response.json()["diagnostics"]["duplicates_removed"] == 1
    assert response.json()["windows"][0]["aggregates"][0]["article_count"] == 1
    data["results"][1]["sentiment_score"] = 0.1
    assert client.post("/aggregate", json=data).status_code == 422


def test_missing_publication_time_is_excluded_without_substitution():
    data = payload(); del data["results"][0]["published_at"]
    response = client.post("/aggregate", json=data)
    assert response.status_code == 200
    assert response.json()["diagnostics"]["missing_timestamp_count"] == 1
    assert all(w["aggregates"] == [] for w in response.json()["windows"])


def test_schema_and_independent_endpoint():
    schema = client.get("/openapi.json").json()
    assert "/aggregate" in schema["paths"]
    assert "AggregationRequest" in schema["components"]["schemas"]
    assert "AggregationResponse" in schema["components"]["schemas"]
    assert client.post("/analyze", json={}).status_code == 404


def test_import_does_not_import_any_scoring_stack():
    # A fresh interpreter verifies isolation even when the full suite already
    # imported inference modules. No files, downloads, or model loads are needed.
    code = """
import sys
from app.aggregation_api import app
blocked = {'app.vader', 'app.finbert', 'vaderSentiment', 'transformers',
           'torch', 'optimum', 'onnxruntime'}
assert not (blocked & set(sys.modules)), blocked & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

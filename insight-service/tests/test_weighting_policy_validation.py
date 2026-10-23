import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.extensions_api import app
from app.weighted_aggregation import WeightingPolicy

client = TestClient(app)


def payload():
    return dict(as_of="2026-10-23T12:00:00Z", results=[
        dict(document_id="1", model="finbert", label="positive", sentiment_score=0.8,
            tickers=["AAPL"], published_at="2026-10-23T12:00:00Z", engagement=10),
        dict(document_id="2", model="finbert", label="negative", sentiment_score=-0.4,
            tickers=["AAPL"], published_at="2026-10-23T12:00:00Z", engagement=0)])


@pytest.mark.parametrize("name", ["engagement_factorr", "engagement_weight", "engagement_capp",
    "default_source_weights", "source_weight", "unknown"])
def test_unknown_weighting_policy_fields_return_422(name):
    data = payload(); data["weighting"] = {name: 0.1}
    response = client.post("/aggregate/weighted", json=data)
    assert response.status_code == 422
    assert any(error["type"] == "extra_forbidden" and error["loc"] == ["body", "weighting", name]
               for error in response.json()["detail"])


@pytest.mark.parametrize("name", ["weightings", "weightingg", "engagement_factor"])
def test_misplaced_or_misspelled_top_level_configuration_is_rejected(name):
    data = payload(); data[name] = {}
    response = client.post("/aggregate/weighted", json=data)
    assert response.status_code == 422
    assert any(error["type"] == "extra_forbidden" and error["loc"][-1] == name
               for error in response.json()["detail"])


def test_unknown_fields_are_rejected_even_alongside_valid_configuration():
    with pytest.raises(ValidationError, match="Extra inputs"):
        WeightingPolicy(engagement_factor=0.1, engagement_factorr=0.2)


def test_documented_policy_calculation_is_unchanged():
    data = payload(); data["weighting"] = dict(engagement_factor=0.1,
        engagement_cap=100, default_source_weight=1, source_weights={})
    response = client.post("/aggregate/weighted", json=data)
    assert response.status_code == 200
    row = response.json()["windows"][0]["aggregates"][0]
    assert row["weighted_mean_sentiment"] == pytest.approx(0.4)
    assert row["total_weight"] == 3
    assert row["article_count"] == 2


def test_missing_configuration_keeps_equal_weight_defaults():
    response = client.post("/aggregate/weighted", json=payload())
    assert response.status_code == 200
    row = response.json()["windows"][0]["aggregates"][0]
    assert row["weighted_mean_sentiment"] == pytest.approx(0.2)


def test_dynamic_source_names_and_finbert_probability_metadata_remain_supported():
    data = payload()
    data["results"][0].update(source=" New Wire ", confidence=0.8,
        probabilities=dict(positive=0.85, negative=0.05, neutral=0.1))
    data["weighting"] = dict(source_weights={"NEW WIRE": 2})
    response = client.post("/aggregate/weighted", json=data)
    assert response.status_code == 200
    assert response.json()["windows"][0]["aggregates"][0]["weighted_mean_sentiment"] == pytest.approx(0.4)

import pytest
from fastapi.testclient import TestClient

from app.velocity_api import app
from app.velocity_models import VelocityResponse

client = TestClient(app)


def payload():
    return {"observations": [
        {"ticker": "AAPL", "model": "finbert", "sentiment_score": -0.2,
         "observed_at": "2026-10-23T10:00:00Z"},
        {"ticker": "AAPL", "model": "finbert", "sentiment_score": 0.4,
         "observed_at": "2026-10-23T12:00:00Z"}]}


def test_velocity_api_formula_and_contract():
    response = client.post("/signals/velocity", json=payload())
    assert response.status_code == 200
    result = VelocityResponse.model_validate(response.json()).results[-1]
    assert result.sentiment_change == pytest.approx(0.6)
    assert result.elapsed_hours == 2 and result.velocity_per_hour == pytest.approx(0.3)
    assert result.direction == "increasing"


@pytest.mark.parametrize("changes", [{"sentiment_score": 2}, {"ticker": ""},
    {"observed_at": "2026-10-23T10:00:00"}, {"model": "unknown"}])
def test_invalid_observations_return_422(changes):
    data = payload(); data["observations"][0].update(changes)
    assert client.post("/signals/velocity", json=data).status_code == 422


def test_conflict_returns_422_and_identical_duplicate_returns_null_velocity():
    data = payload(); data["observations"][1]["observed_at"] = data["observations"][0]["observed_at"]
    assert client.post("/signals/velocity", json=data).status_code == 422
    data["observations"][1]["sentiment_score"] = -0.2
    response = client.post("/signals/velocity", json=data)
    assert response.status_code == 200
    assert response.json()["results"][0]["velocity_per_hour"] is None


def test_empty_and_missing_observations():
    assert client.post("/signals/velocity", json={"observations": []}).json()["results"] == []
    assert client.post("/signals/velocity", json={}).status_code == 422
    response = client.post("/signals/velocity", json={"observations": [{"ticker": "AAPL", "model": "vader"}]})
    assert response.status_code == 200
    assert response.json()["results"][0]["direction"] is None


def test_configurable_threshold_and_display_unit():
    data = payload(); data.update(time_unit="days", stability_threshold_per_hour=0.5)
    response = client.post("/signals/velocity", json=data)
    assert response.status_code == 200
    value = response.json()["results"][-1]
    assert value["velocity_per_unit"] == pytest.approx(7.2)
    assert value["direction"] == "stable"


def test_existing_endpoints_and_schema_preserved_without_loading_model(monkeypatch):
    from app import finbert
    monkeypatch.setattr(finbert, "_load_backend", lambda *_: pytest.fail("must not load FinBERT"))
    assert client.post("/signals/velocity", json=payload()).status_code == 200
    schema = client.get("/openapi.json").json()
    for path in ["/health", "/analyze", "/analyze/batch", "/aggregate", "/aggregate/weighted", "/signals/velocity"]:
        assert path in schema["paths"]
    assert client.get("/health").status_code == 200
    assert client.post("/aggregate", json={"results": []}).status_code == 200

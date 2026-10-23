import asyncio
import importlib
import json

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.granger_api import app
from app.validation_errors import request_validation_error_handler

client = TestClient(app, raise_server_exceptions=False)
TOKENS = ["NaN", "Infinity", "-Infinity"]
DOCUMENT = dict(document_id="1", title="Earnings", text="Company reports earnings", tickers=["AAPL"])


def strict_json(text):
    def reject_constant(value):
        pytest.fail(f"Non-finite literal leaked into JSON response: {value}")
    return json.loads(text, parse_constant=reject_constant)


def assert_safe_validation(response):
    assert response.status_code == 422
    assert isinstance(strict_json(response.text)["detail"], list)
    assert "Traceback" not in response.text
    assert response.headers["content-type"].startswith("application/json")


def payload_for(endpoint, token):
    number = float(token)
    if endpoint in {"/analyze", "/analyze/validate"}:
        return dict(documents=[dict(DOCUMENT, text=number)])
    if endpoint == "/analyze/batch":
        return dict(documents=[DOCUMENT], chunk_size=number)
    if endpoint == "/aggregate":
        return dict(results=[dict(document_id="1", model="vader", label="neutral",
                                 sentiment_score=number, tickers=["AAPL"])])
    if endpoint == "/aggregate/weighted":
        return dict(results=[], weighting=dict(default_source_weight=number))
    if endpoint in {"/signals/velocity", "/signals/outliers"}:
        return dict(observations=[dict(ticker="AAPL", model="vader", sentiment_score=number)])
    if endpoint == "/signals/correlation":
        return dict(sentiments=[dict(ticker="AAPL", model="vader", aggregated_sentiment=number)], prices=[])
    return dict(sentiments=[], prices=[], alpha=number)


ENDPOINTS = ["/analyze/validate", "/analyze", "/analyze/batch", "/aggregate", "/aggregate/weighted",
             "/signals/velocity", "/signals/outliers", "/signals/correlation", "/signals/granger"]


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_nonfinite_numbers_return_serializable_422_on_composed_app(endpoint, token):
    # Send raw text: HTTPX's json= encoder itself refuses these literals.
    response = client.post(endpoint, content=json.dumps(payload_for(endpoint, token)),
                           headers={"content-type": "application/json"})
    assert_safe_validation(response)


STANDALONE_APPS = [("main", "/analyze/validate"), ("vader_api", "/analyze"),
    ("finbert_api", "/analyze"), ("aggregation_api", "/aggregate"),
    ("extensions_api", "/aggregate/weighted"), ("velocity_api", "/signals/velocity"),
    ("outliers_api", "/signals/outliers"), ("correlation_api", "/signals/correlation"),
    ("granger_api", "/signals/granger")]


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("module,endpoint", STANDALONE_APPS)
def test_handler_registered_on_each_standalone_application(module, endpoint, token):
    application = importlib.import_module(f"app.{module}").app
    response = TestClient(application, raise_server_exceptions=False).post(endpoint,
        content=json.dumps(payload_for(endpoint, token)), headers={"content-type": "application/json"})
    assert_safe_validation(response)


@pytest.mark.parametrize("token", TOKENS)
def test_nested_invalid_inputs_are_safely_serialized(token):
    response = client.post("/aggregate/weighted", content=json.dumps(dict(results=[],
        weighting=dict(source_weights=[dict(nested=[float(token)])]))),
        headers={"content-type": "application/json"})
    assert_safe_validation(response)


def test_handler_handles_nonfinite_error_context_and_hides_exception_internals():
    exc = RequestValidationError([dict(type="value_error", loc=("body", "value"), msg="invalid value",
        input=dict(nested=[float("nan"), float("inf"), -float("inf")]),
        ctx=dict(error=ValueError("private internal trace details"), numeric=float("nan")))])
    response = asyncio.run(request_validation_error_handler(Request({"type": "http"}), exc))
    body = strict_json(response.body.decode())
    assert body["detail"][0]["input"]["nested"] == ["NaN", "Infinity", "-Infinity"]
    assert "private internal trace" not in response.body.decode()


@pytest.mark.parametrize("score", [-1, -0.25, 0, 0.25, 1])
def test_valid_finite_sentiment_inputs_remain_accepted(score):
    response = client.post("/signals/velocity", json=dict(observations=[
        dict(ticker="AAPL", model="vader", sentiment_score=score, observed_at="2026-10-23T12:00:00Z")]))
    assert response.status_code == 200
    assert response.json()["results"][0]["current_sentiment"] == score


def test_valid_strings_containing_nonfinite_words_are_not_rejected():
    response = client.post("/analyze/validate", json=dict(documents=[dict(DOCUMENT,
        text="NaN, Infinity and -Infinity are quoted words in this financial report.")]))
    assert response.status_code == 200


def test_ordinary_422_detail_shape_is_preserved():
    response = client.post("/signals/velocity", json={})
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["type"] == "missing" and detail["loc"] == ["body", "observations"]

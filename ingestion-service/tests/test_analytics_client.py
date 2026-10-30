import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from pydantic import ValidationError

from app.core.analytics_client import AnalyticsClient
from app.integration_config import InsightConfig
from app.models.analytics import AnalyticsOptions
from app.models.insight import AnalyzeIngestionRequest, AnalysisReport
from app.models.ingestion import Document

MODEL = "loughran_mcdonald"


def inputs(missing_date=False, failed=False, no_tickers=False):
    now = datetime.now(timezone.utc)
    document = Document(document_id="doc", title="Excellent", text="Excellent success.",
        language="en", tickers=[] if no_tickers else ["AAPL"], source="news",
        original_url=None, canonical_url=None, collected_at=now,
        published_at=None if missing_date else now - timedelta(minutes=1))
    result = {"document_id": "doc", "model": MODEL, "label": "positive",
              "sentiment_score": 1., "confidence": None, "tickers": document.tickers}
    report = AnalysisReport(status="error" if failed else "success", model=MODEL,
        success_count=0 if failed else 1, error_count=1 if failed else 0,
        outcomes=[{"input_index": 0, "document_id": "doc", "tickers": document.tickers,
                   "status": "error" if failed else "success", "result": None if failed else result}])
    return [document], report


def aggregate(body):
    end = datetime.fromisoformat(body["as_of"])
    intervals = {"1h": timedelta(hours=1), "24h": timedelta(days=1), "7d": timedelta(days=7)}
    return {"schema_version": "1.0", "as_of": body["as_of"], "windows": [
        {"window": window, "start": (end-intervals[window]).isoformat(), "end": end.isoformat(),
         "aggregates": [{
             "ticker": "AAPL", "model": MODEL, "mean_sentiment": 1., "article_count": 1,
             "label_counts": {"positive": 1, "neutral": 0, "negative": 0},
             "distribution": {"positive": 1., "neutral": 0., "negative": 0.},
         }] if body["results"][0]["published_at"] and body["results"][0]["tickers"] else []}
        for window in body["windows"]], "diagnostics": {"input_count": len(body["results"])}}


def run(handler, *, options=None, **input_options):
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            docs, report = inputs(**input_options)
            return await AnalyticsClient(InsightConfig(), client).enrich(
                docs, report, options or AnalyticsOptions())
    return asyncio.run(execute())


def test_projection_and_single_snapshot_insufficiency():
    import json
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path.startswith("/aggregate"):
            assert body["results"][0]["published_at"] is not None
            assert "probabilities" not in body["results"][0]
            if request.url.path.endswith("weighted"):
                assert body["results"][0]["source"] == "news"
            return httpx.Response(200, json=aggregate(body))
        assert len(body["observations"]) == 1
        row = body["observations"][0]
        assert row["ticker"] == "AAPL"
        if request.url.path.endswith("velocity"):
            result = {**row, "current_observed_at": row["observed_at"],
                      "current_sentiment": 1., "velocity_per_hour": None}
        else:
            result = {**row, "z_score": None, "historical_count": 0,
                      "diagnostic_reason": "insufficient_history"}
        return httpx.Response(200, json={"schema_version": "1.0", "results": [result], "diagnostics": {}})
    raw, weighted, signals = run(handler)
    assert raw.status == weighted.status == "success"
    assert signals.velocity.status == signals.outliers.status == "unavailable"
    assert signals.outliers.reason == "insufficient_history"
    assert signals.correlation.reason == signals.granger.reason == "prices_not_provided"
    assert calls == ["/aggregate", "/aggregate/weighted", "/signals/velocity", "/signals/outliers"]


@pytest.mark.parametrize("options", [{"missing_date": True}, {"no_tickers": True}])
def test_missing_metadata_not_fabricated(options):
    import json
    def handler(request):
        assert request.url.path.startswith("/aggregate")
        return httpx.Response(200, json=aggregate(json.loads(request.content)))
    raw, _, signals = run(handler, **options)
    assert raw.status == "success"
    assert not raw.data["windows"][0]["aggregates"]
    assert signals.velocity.reason == "no_timestamped_ticker_aggregates"


def test_failed_scoring_makes_no_downstream_calls():
    raw, weighted, signals = run(lambda _: pytest.fail("Unexpected network"), failed=True)
    assert raw.reason == weighted.reason == "no_successful_analysis"
    assert signals.velocity.status == "unavailable"


@pytest.mark.parametrize("mode", ["timeout", "malformed", "wrong_model", "wrong_window", "http_error"])
def test_downstream_failure_is_isolated_and_sanitized(mode):
    import json
    def handler(request):
        if request.url.path == "/aggregate":
            if mode == "timeout":
                raise httpx.ReadTimeout("secret-key")
            if mode == "http_error":
                return httpx.Response(503, json={"detail": "secret-key"})
            if mode == "malformed":
                return httpx.Response(200, json={"schema_version": "1.0"})
            payload = aggregate(json.loads(request.content))
            if mode == "wrong_model":
                payload["windows"][0]["aggregates"][0]["model"] = "vader"
            else:
                payload["windows"][0]["end"] = "2000-01-01T00:00:00Z"
            return httpx.Response(200, json=payload)
        assert request.url.path == "/aggregate/weighted"
        return httpx.Response(200, json=aggregate(json.loads(request.content)))
    raw, weighted, signals = run(handler)
    assert raw.status == "error"
    assert weighted.status == "success"
    assert signals.velocity.status == "unavailable"
    assert "secret" not in raw.model_dump_json()


@pytest.mark.parametrize("change", ["future", "conflict", "model", "window", "alignment"])
def test_invalid_history_rejected(change):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    row = {"ticker": "aapl", "model": MODEL, "observed_at": now.isoformat(), "sentiment_score": .3}
    options = {"history": [row]}
    if change == "future":
        row["observed_at"] = (now + timedelta(days=1)).isoformat()
    elif change == "conflict":
        options["history"].append({**row, "sentiment_score": .4})
    elif change == "model":
        row["model"] = "vader"
    elif change == "window":
        row["aggregation_window"] = "7d"
    else:
        options["prices"] = [{"ticker": "AAPL", "timestamp": (now+timedelta(minutes=1)).isoformat(),
                              "close_price": 100.}]
    with pytest.raises(ValidationError):
        AnalyzeIngestionRequest(model=MODEL, analytics=options)


def test_current_snapshot_not_backdated_to_market_bucket():
    import json
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    options = AnalyticsOptions(prices=[{"ticker": "AAPL", "timestamp": now, "close_price": 100.}])
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path not in {"/signals/correlation", "/signals/granger"}
        if request.url.path.startswith("/aggregate"):
            return httpx.Response(200, json=aggregate(body))
        # A broken signal response must be reported safely.
        return httpx.Response(200, json={"schema_version": "1.0", "results": [{}], "diagnostics": {}})
    _, _, signals = run(handler, options=options)
    assert signals.velocity.reason == "invalid_response"
    assert signals.correlation.reason == "no_bucket_aligned_sentiment_history"

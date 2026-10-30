import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.insight_client import InsightClient
from app.integration_config import InsightConfig
from app.main import create_app
from app.models.ingestion import Document
from app.models.insight import WireBatch


def document(index=0):
    return Document(document_id=f"doc-{index}", title="Earnings", text="The company reports excellent profits.",
                    tickers=["AAPL"], language="en", source="news", original_url="https://example.com/a",
                    canonical_url="https://example.com/a", published_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
                    collected_at=datetime.now(timezone.utc), metadata={"score": 10})


def batch_for(documents, failed=False, model="vader"):
    outcomes = []
    for index, doc in enumerate(documents):
        outcome = {"input_index": index, "document_id": doc["document_id"], "tickers": doc["tickers"],
                   "status": "error" if failed and index == 0 else "success"}
        if outcome["status"] == "success":
            outcome["result"] = {"document_id": doc["document_id"], "model": model, "tickers": doc["tickers"],
                                 "label": "positive", "sentiment_score": 0.6, "confidence": None}
            if model == "finbert":
                outcome["result"]["probabilities"] = {"positive": .8, "neutral": .1, "negative": .1}
        else:
            outcome["error"] = {"code": "inference_failed", "message": "upstream-secret-must-be-hidden"}
        outcomes.append(outcome)
    errors = int(failed and bool(documents))
    return {"schema_version": "1.0", "model": model, "outcomes": outcomes,
            "success_count": len(documents) - errors, "error_count": errors}


def analyze(handler, documents=None, **options):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await InsightClient(InsightConfig(retry_backoff_seconds=0, **options), client).analyze(
                [document()] if documents is None else documents)
    return asyncio.run(run())


def test_success_projection_and_metadata_untouched():
    original = document()
    before = original.model_dump()
    def handler(request):
        assert request.url.path == "/analyze/batch"
        body = json.loads(request.content)
        assert set(body["documents"][0]) == {"document_id", "title", "text", "language", "tickers"}
        assert body["documents"][0]["text"] == original.text
        return httpx.Response(200, json=batch_for(body["documents"]))
    report = analyze(handler, [original])
    assert report.status == "success" and report.success_count == 1
    assert report.outcomes[0].document_id == "doc-0"
    assert report.outcomes[0].tickers == ["AAPL"]
    assert original.model_dump() == before


def test_empty_does_not_call_insight():
    result = analyze(lambda _: pytest.fail("network"), [])
    assert result.status == "empty" and not result.outcomes


def test_partial_failure_no_remote_error_leak():
    report = analyze(lambda r: httpx.Response(200, json=batch_for(json.loads(r.content)["documents"], failed=True)), [document(), document(1)])
    assert report.status == "partial" and report.error_count == 1
    assert report.outcomes[0].error_code == "inference_failed"
    assert "upstream-secret" not in report.model_dump_json()


@pytest.mark.parametrize("status,attempts", [(429, 2), (500, 2), (502, 2), (503, 2), (504, 2), (400, 1), (401, 1), (404, 1), (422, 1)])
def test_retry_status_policy(status, attempts):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"detail": "internal secret"})
    report = analyze(handler)
    assert len(calls) == attempts
    assert report.status == "error" and "internal secret" not in report.model_dump_json()


def test_transient_then_success():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=batch_for(json.loads(request.content)["documents"]))
    assert analyze(handler).status == "success"
    assert len(calls) == 2


@pytest.mark.parametrize("kind", [httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout])
def test_network_errors(kind):
    calls = []
    def handler(request):
        calls.append(request)
        raise kind("internal credentials", request=request)
    report = analyze(handler)
    assert report.status == "error" and len(calls) == 2
    assert "internal credentials" not in report.model_dump_json()


def test_long_retry_after_is_not_shortened():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "600"})
    assert analyze(handler).outcomes[0].error_code == "rate_limited"
    assert len(calls) == 1


@pytest.mark.parametrize("content,code", [(b"invalid", "invalid_json"), (b'[]', "invalid_response"), (b'{}', "invalid_response")])
def test_invalid_json_and_contract_no_retry(content, code):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=content)
    assert analyze(handler).outcomes[0].error_code == code
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["id", "ticker", "index", "count", "result_id", "nan", "model", "missing"])
def test_response_identity_and_validation(mutation):
    def handler(request):
        result = batch_for(json.loads(request.content)["documents"])
        outcome = result["outcomes"][0]
        if mutation == "id": outcome["document_id"] = "other"
        if mutation == "ticker": outcome["tickers"] = ["OTHER"]
        if mutation == "index": outcome["input_index"] = 4
        if mutation == "count": result["success_count"] = 99
        if mutation == "result_id": outcome["result"]["document_id"] = "other"
        if mutation == "nan": outcome["result"]["sentiment_score"] = "NaN"
        if mutation == "model": result["model"] = "finbert"
        if mutation == "missing": result["outcomes"] = []
        return httpx.Response(200, json=result)
    assert analyze(handler).outcomes[0].error_code == "invalid_response"


def test_chunking_preserves_global_indices_and_duplicate_ids():
    calls = []
    def handler(request):
        docs = json.loads(request.content)["documents"]
        calls.append(len(docs))
        return httpx.Response(200, json=batch_for(docs))
    report = analyze(handler, [document()] * 101)
    assert calls == [100, 1]
    assert [o.input_index for o in report.outcomes] == list(range(101))


@pytest.mark.parametrize("values", [{"service_url": "https://user:secret@internal"}, {"service_url": "file:///x"},
                                   {"service_url": "http://localhost?key=secret"}, {"max_retries": 4}, {"request_timeout_seconds": float("inf")}])
def test_configuration_validation(values):
    with pytest.raises(ValidationError):
        InsightConfig(**values)


def test_environment(monkeypatch):
    monkeypatch.setenv("INSIGHT_SERVICE_URL", "http://localhost:8009")
    monkeypatch.setenv("INSIGHT_MAX_RETRIES", "2")
    assert Settings.from_env().insight.service_url == "http://localhost:8009"
    assert Settings.from_env().insight.max_retries == 2


def test_new_api_preserves_ingestion_when_insight_unavailable(rss):
    async def run():
        settings = Settings(sources=[{"name": "news", "url": "https://feeds.example/news"}], insight={"max_retries": 0})
        def handler(request):
            if request.url.host == "feeds.example":
                return httpx.Response(200, content=rss)
            raise httpx.ConnectError("secret", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as api:
                response = await api.post("/ingest/analyze", json={})
                assert response.status_code == 200
                body = response.json()
                assert body["analysis"]["status"] == "error"
                doc = body["ingestion"]["documents"][0]
                assert doc["source"] == "news" and doc["published_at"] == "2026-10-27T10:00:00Z"
                assert doc["document_id"] == body["analysis"]["outcomes"][0]["document_id"]
                assert (await api.post("/ingest/run", json={})).status_code == 200
    asyncio.run(run())


@pytest.mark.parametrize("model", ["vader", "loughran_mcdonald"])
def test_actual_insight_request_response_contracts(model):
    documents = [{name: getattr(document(), name) for name in ("document_id", "title", "text", "language", "tickers")}]
    code = """
import json,sys
from app.models.batch import BatchAnalyzeRequest
from app.core.batch import analyze_batch
body=json.load(sys.stdin)
request=BatchAnalyzeRequest.model_validate(body)
print(analyze_batch(request).model_dump_json())
"""
    service = Path(__file__).resolve().parents[2] / "insight-service"
    interpreter = service / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    result = subprocess.run([str(interpreter) if interpreter.exists() else sys.executable, "-B", "-c", code], cwd=service,
                            input=json.dumps({"documents": documents, "model": model, "chunk_size": 4}),
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    response = WireBatch.model_validate_json(result.stdout)
    assert response.success_count == 1 and response.outcomes[0].tickers == ["AAPL"]
    assert response.model == response.outcomes[0].result.model == model


def test_loughran_mcdonald_projection_and_model_mismatch():
    async def run(wrong_model):
        def handler(request):
            body = json.loads(request.content)
            assert body["model"] == "loughran_mcdonald"
            assert body["chunk_size"] == 2
            return httpx.Response(200, json=batch_for(body["documents"],
                model="vader" if wrong_model else "loughran_mcdonald"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await InsightClient(InsightConfig(), client).analyze(
                [document()], "loughran_mcdonald", 2)
    result = asyncio.run(run(False))
    assert result.status == "success" and result.outcomes[0].result.confidence is None
    rejected = asyncio.run(run(True))
    assert rejected.status == "error"
    assert rejected.outcomes[0].error_code == "invalid_response"


def test_unknown_analysis_model_rejected_before_network():
    from app.models.insight import AnalyzeIngestionRequest
    with pytest.raises(ValidationError):
        AnalyzeIngestionRequest(model="unknown")
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _: pytest.fail("Must not call network"))) as client:
            with pytest.raises(ValueError, match="invalid_analysis_options"):
                await InsightClient(InsightConfig(), client).analyze([document()], "unknown")
    asyncio.run(run())

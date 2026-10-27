from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app


def test_health_and_openapi():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok", "service": "ingestion-service"}
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {"/health", "/ingest/run", "/ingest/market"}
        assert schema["components"]["schemas"]["IngestRequest"]["additionalProperties"] is False
        assert "Document" in schema["components"]["schemas"]


def test_valid_ingestion(client_factory, rss):
    client = client_factory(lambda request: httpx.Response(200, content=rss))
    response = client.post("/ingest/run", json={})
    assert response.status_code == 200
    payload = response.json()
    document = payload["documents"][0]
    assert document["tickers"] == ["AAPL"]
    assert document["language"] == "en"
    assert document["published_at"] == "2026-10-27T10:00:00Z"
    assert document["original_url"].endswith("?utm_source=rss")
    assert document["canonical_url"] == "https://example.com/story"
    assert document["collected_at"] == payload["collected_at"]
    assert payload["diagnostics"]["returned"] == 1


@pytest.mark.parametrize("body", [
    {"max_documents": 0}, {"max_documents": 101}, {"max_documents": True},
    {"max_documents": "2"}, {"source_names": []}, {"source_names": [1]},
    {"url": "http://127.0.0.1"}, {"source_names": ["unknown"]},
])
def test_invalid_requests(client_factory, body):
    def forbidden(request):
        pytest.fail("Invalid requests must not access the network")
    response = client_factory(forbidden).post("/ingest/run", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_nonfinite_validation(client_factory):
    response = client_factory(lambda _: pytest.fail("network")).post(
        "/ingest/run", content='{"max_documents": NaN}', headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_empty_and_missing_content(client_factory):
    client = client_factory(lambda _: httpx.Response(200, content=b"<rss><channel><item><title>Empty</title></item></channel></rss>"))
    response = client.post("/ingest/run", json={})
    assert response.status_code == 200
    assert response.json()["documents"] == []
    assert response.json()["diagnostics"]["rejected"] == 1


def test_empty_feed(client_factory):
    response = client_factory(lambda _: httpx.Response(200, content=b"<rss><channel/></rss>")).post("/ingest/run", json={})
    assert response.status_code == 200
    assert response.json()["diagnostics"]["collected"] == 0


def test_partial_failure_and_selection(rss):
    settings = Settings(sources=[{"name": "bad", "url": "https://feeds.example/bad"},
                                 {"name": "good", "url": "https://feeds.example/good"}])
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(500) if request.url.path == "/bad" else httpx.Response(200, content=rss)
    import asyncio
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as client:
                result = (await client.post("/ingest/run", json={})).json()
                assert [s["status"] for s in result["sources"]] == ["error", "success"]
                assert len(result["documents"]) == 1
                calls.clear()
                selected = await client.post("/ingest/run", json={"source_names": ["good"]})
                assert selected.status_code == 200
                assert calls == ["/good"]
    asyncio.run(run())


def test_total_failure(client_factory):
    response = client_factory(lambda _: httpx.Response(500)).post("/ingest/run", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "all_sources_failed"
    assert response.json()["detail"]["sources"][0]["error_code"] == "http_error"


@pytest.mark.parametrize("raw", ["not-json", '[{"name":"bad","url":"file:///etc/passwd"}]'])
def test_invalid_environment(monkeypatch, raw):
    monkeypatch.setenv("INGESTION_SOURCES", raw)
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    response = client.post("/ingest/run", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "invalid_configuration"


def test_no_sources(monkeypatch):
    monkeypatch.setenv("INGESTION_SOURCES", "[]")
    response = TestClient(create_app()).post("/ingest/run", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "no_sources_configured"


@pytest.mark.parametrize("values", [
    {"timeout_seconds": 0}, {"timeout_seconds": float("inf")}, {"max_documents": 101},
    {"max_entries_per_source": 1001}, {"max_response_bytes": 100},
    {"sources": [{"name": "x", "url": "http://a"}, {"name": "x", "url": "http://b"}]},
    {"company_tickers": {"": "ABC"}}, {"company_tickers": {"Name": "not a ticker"}},
])
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        Settings(**values)


def test_environment_configuration(monkeypatch):
    monkeypatch.setenv("INGESTION_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("INGESTION_COMPANY_TICKERS", '{"Microsoft":"msft"}')
    assert Settings.from_env().timeout_seconds == 3.5
    assert Settings.from_env().company_tickers == {"Microsoft": "MSFT"}


def test_deduplication_limits_and_repeatability(client_factory):
    body = b'''<rss><channel>
    <item><title>Same title</title><description>Body one</description><link>https://example.com/one</link></item>
    <item><title>Same title</title><description>Body two</description><link>https://example.com/two</link></item>
    <item><title>Duplicate URL</title><description>Changed</description><link>https://example.com/one?utm_source=x</link></item>
    <item><title>Duplicate body</title><description>Body one</description><link>https://example.com/three</link></item>
    <item><title>Other</title><description>Body four</description></item>
    </channel></rss>'''
    client = client_factory(lambda _: httpx.Response(200, content=body), max_documents=2)
    first = client.post("/ingest/run", json={}).json()
    second = client.post("/ingest/run", json={}).json()
    assert first["diagnostics"] == {"collected": 5, "rejected": 0, "deduplicated": 2,
                                     "returned": 2, "limited": 1, "published_at_missing_or_invalid": 2}
    assert [d["text"] for d in first["documents"]] == ["Body one", "Body two"]
    assert [d["document_id"] for d in first["documents"]] == [d["document_id"] for d in second["documents"]]
    assert len(client.post("/ingest/run", json={"max_documents": 1}).json()["documents"]) == 1


def test_insight_contract_compatibility(client_factory, rss):
    documents = client_factory(lambda _: httpx.Response(200, content=rss)).post("/ingest/run", json={}).json()["documents"]
    # Separate interpreter avoids collisions between the two services' app packages.
    service = Path(__file__).resolve().parents[2] / "insight-service"
    code = """
import json, sys
from app.models.sentiment import AnalyzeRequest
from app.models.batch import BatchAnalyzeRequest
documents = json.load(sys.stdin)
for contract in (AnalyzeRequest, BatchAnalyzeRequest):
    validated = contract.model_validate({'documents': documents})
    assert validated.documents[0].tickers == ['AAPL']
print('Both existing Insight contracts accept ingestion documents')
"""
    import json
    result = subprocess.run([sys.executable, "-c", code], input=json.dumps(documents), cwd=service,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr

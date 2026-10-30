"""Cross-provider pipeline and unchanged downstream contract checks, all offline."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.helper.provider_http import SourceError, fetch_json


def test_mixed_sources_share_transform_dedupe_and_insight_contracts(rss):
    async def run():
        settings = Settings(
            sources=[{"name": "feed", "url": "https://feeds.example/rss"}],
            newsapi={"enabled": True, "api_key": "secret-key", "keywords": ["earnings"]},
            gdelt={"enabled": True, "keywords": ["earnings"]},
            hackernews={"enabled": True, "keywords": ["ETF"]},
        )
        calls = []
        def handler(request):
            calls.append(str(request.url))
            host = request.url.host
            if host == "feeds.example":
                return httpx.Response(200, content=rss)
            if host == "newsapi.org":
                return httpx.Response(200, json={"status": "ok", "articles": [{"title": "Republished", "description": "Different body",
                    "url": "https://example.com/story?utm_source=newsapi", "publishedAt": "2026-10-27T10:00:00Z"}]})
            if host == "api.gdeltproject.org":
                return httpx.Response(200, json={"articles": [{"title": "Microsoft announces $MSFT results", "url": "https://example.com/other", "language": "English"}]})
            assert host == "hn.algolia.com"
            return httpx.Response(200, json={"hits": [{"objectID": "99999", "title": "ETF $SPY discussion",
                "story_text": "<p>Financial report &amp; market news</p>", "url": "https://example.com/etf",
                "author": "example_author", "created_at": "2026-10-27T12:00:00Z", "points": 15, "num_comments": 3}],
                "page": 0, "nbPages": 1})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as api:
                response = await api.post("/ingest/run", json={"providers": ["hackernews", "gdelt", "rss", "newsapi"]})
                assert response.status_code == 200
                payload = response.json()
                assert [source["name"] for source in payload["sources"]] == ["feed", "newsapi", "gdelt", "hackernews"]
                assert len(payload["documents"]) == 3
                assert payload["diagnostics"]["deduplicated"] == 1
                assert payload["documents"][2]["text"] == "Financial report & market news"
                assert all("secret" not in url for url in calls)
                return payload["documents"]
    documents = asyncio.run(run())
    service = Path(__file__).resolve().parents[2] / "insight-service"
    code = """
import json,sys
from app.models.sentiment import AnalyzeRequest
from app.models.batch import BatchAnalyzeRequest
documents = json.load(sys.stdin)
for contract in (AnalyzeRequest, BatchAnalyzeRequest):
    parsed = contract.model_validate({'documents': documents})
    assert len(parsed.documents) == 3
    assert parsed.documents[2].tickers == ['SPY']
"""
    result = subprocess.run([sys.executable, "-c", code], input=json.dumps(documents), cwd=service,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("provider", ["newsapi", "gdelt", "hackernews", "market"])
def test_invalid_provider_environment_does_not_leak(monkeypatch, provider):
    monkeypatch.setenv(f"INGESTION_{provider.upper()}_TIMEOUT_SECONDS", "secret-invalid-value")
    response = TestClient(create_app()).post("/ingest/market" if provider == "market" else "/ingest/run", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "invalid_configuration"
    assert "secret-invalid-value" not in response.text


@pytest.mark.parametrize("content,headers,error", [(b"x" * 1025, {}, "response_too_large"),
                                                  (b"", {"content-encoding": "gzip"}, "unsupported_encoding")])
def test_shared_json_transport_bounds(content, headers, error):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=content, headers=headers))) as client:
            with pytest.raises(SourceError, match=error):
                await fetch_json(client, "GET", "https://provider.example", timeout=1, max_bytes=1024)
    asyncio.run(run())


def test_market_empty_api_and_caps():
    async def run():
        settings = Settings(market={"enabled": True, "api_key": "secret", "outputsize": 2,
                                    "instruments": [{"ticker": "AAPL", "provider_symbol": "AAPL"}]})
        def handler(request):
            assert request.url.params["outputsize"] == "2"
            return httpx.Response(200, json={"status": "ok", "meta": {"symbol": "AAPL", "interval": "1h"}, "values": []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as client:
                response = await client.post("/ingest/market", json={"max_candles": 100})
                assert response.status_code == 200 and response.json()["candles"] == []
    asyncio.run(run())



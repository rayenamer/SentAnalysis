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
            reddit={"enabled": True, "client_id": "client-id", "client_secret": "secret-client", "subreddits": ["stocks"]},
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
            if request.url.path == "/api/v1/access_token":
                return httpx.Response(200, json={"access_token": "test-token", "token_type": "bearer"})
            return httpx.Response(200, json={"kind": "Listing", "data": {"after": None, "children": [{"kind": "t3", "data": {
                "id": "xyz", "title": "ETF $SPY discussion", "selftext": "<p>Financial report &amp; market news</p>", "subreddit": "stocks",
                "permalink": "/r/stocks/comments/xyz/etf/", "created_utc": 1793102400, "score": 15, "num_comments": 3}}]}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as api:
                response = await api.post("/ingest/run", json={"providers": ["reddit", "gdelt", "rss", "newsapi"]})
                assert response.status_code == 200
                payload = response.json()
                assert [source["name"] for source in payload["sources"]] == ["feed", "newsapi", "gdelt", "reddit:stocks"]
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


@pytest.mark.parametrize("provider", ["newsapi", "gdelt", "reddit", "market"])
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


def test_reddit_entry_cap_and_token_reuse():
    from app.core.extractors.reddit import RedditExtractor
    from app.config import FeedSource
    async def run():
        settings = Settings(max_entries_per_source=1, reddit={"enabled": True, "client_id": "client", "client_secret": "secret", "subreddits": ["stocks", "investing"]})
        auth_calls = []
        def handler(request):
            if request.url.path == "/api/v1/access_token":
                auth_calls.append(request)
                return httpx.Response(200, json={"access_token": "token", "token_type": "bearer"})
            return httpx.Response(200, json={"kind": "Listing", "data": {"children": [{"kind": "t3", "data": {}}] * 2, "after": "t3_abc"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            extractor = RedditExtractor(http, settings)
            for subreddit in settings.reddit.subreddits:
                rows, truncated = await extractor.fetch(FeedSource(name="reddit:" + subreddit, url="https://oauth.reddit.com"))
                assert len(rows) == 1 and truncated
            assert len(auth_calls) == 1
    asyncio.run(run())

"""Offline provider and API tests; all external HTTP uses MockTransport."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.core.extractors.hackernews import HackerNewsExtractor, normalize_story
from app.helper.provider_http import SourceError
from app.main import create_app

STORY = {"objectID": "12345", "title": "Apple earnings $AAPL", "url": "https://example.com/apple?utm_source=hn",
         "author": "example_author", "created_at": "2026-10-27T12:00:00+02:00", "points": 42, "num_comments": 9,
         "story_text": "<p>The company reports strong earnings and profits with revenue growth for investors in the market.</p>",
         "_tags": ["story", "author_example_author"]}


def page(hits=None, number=0, pages=1):
    return {"hits": [STORY] if hits is None else hits, "page": number, "nbPages": pages}


def settings(**values):
    return Settings(hackernews={"enabled": True, "keywords": ["Apple"], **values})


def extract(handler, settings_override=None, **values):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await HackerNewsExtractor(http, settings_override or settings(**values)).fetch(None)
    return asyncio.run(run())


def test_success_and_public_request():
    def handler(request):
        assert str(request.url).startswith("https://hn.algolia.com/api/v1/search_by_date?")
        assert dict(request.url.params) == {"query": "Apple", "tags": "story", "hitsPerPage": "10", "page": "0",
                                            "typoTolerance": "false", "queryType": "prefixNone"}
        assert "authorization" not in request.headers and "x-api-key" not in request.headers
        return httpx.Response(200, json=page())
    records, truncated = extract(handler)
    assert len(records) == 1 and not truncated
    raw = records[0]
    assert raw.metadata["hn_id"] == "12345" and raw.metadata["author"] == "example_author"
    assert raw.metadata["points"] == 42 and raw.metadata["num_comments"] == 9
    assert raw.url == STORY["url"] and raw.published_at == "2026-10-27T10:00:00+00:00"


def test_separate_keywords_with_stable_pagination():
    calls = []
    def handler(request):
        keyword, number = request.url.params["query"], int(request.url.params["page"])
        calls.append((keyword, number, request.url.params["hitsPerPage"]))
        return httpx.Response(200, json=page([{**STORY, "objectID": str(100 + len(calls))}], number, 2))
    records, truncated = extract(handler, keywords=["Apple", "Microsoft"], max_pages=2)
    assert calls == [("Apple", 0, "10"), ("Apple", 1, "10"), ("Microsoft", 0, "10"), ("Microsoft", 1, "10")]
    assert len(records) == 4 and not truncated


def test_page_cap():
    records, truncated = extract(lambda _: httpx.Response(200, json=page(pages=5)))
    assert len(records) == 1 and truncated


def test_record_and_global_entry_limits():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=page([STORY] * 10, pages=10))
    config = Settings(max_entries_per_source=2, hackernews={"enabled": True, "keywords": ["Apple", "Tesla"], "max_records": 5})
    rows, truncated = extract(handler, settings_override=config)
    assert len(rows) == 2 and truncated and len(calls) == 1
    assert calls[0].url.params["hitsPerPage"] == "2"


def test_empty():
    assert extract(lambda _: httpx.Response(200, json=page([], pages=0))) == ([], False)


@pytest.mark.parametrize("status,code", [(429, "rate_limited"), (500, "http_error"), (503, "http_error"),
                                       (403, "authentication_failed"), (302, "http_error")])
def test_http_errors_no_retry_or_redirect(status, code):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "http://127.0.0.1/private", "retry-after": "120"})
    with pytest.raises(SourceError, match=code) as exc:
        extract(handler)
    assert len(calls) == 1
    if status == 429:
        assert exc.value.retry_after_seconds == 120


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
def test_network_errors(error):
    def handler(request):
        raise error("internal secret", request=request)
    with pytest.raises(SourceError) as exc:
        extract(handler)
    assert str(exc.value) in {"timeout", "http_error"} and "secret" not in str(exc.value)


@pytest.mark.parametrize("body,code", [(b"not-json", "invalid_json"), (b"[]", "invalid_response"), (b"{}", "invalid_response"),
                                      (b'{"hits":{},"page":0,"nbPages":1}', "invalid_response"),
                                      (b'{"hits":[],"page":1,"nbPages":2}', "invalid_response"),
                                      (b'{"hits":[],"page":0,"nbPages":true}', "invalid_response")])
def test_invalid_response(body, code):
    with pytest.raises(SourceError, match=code):
        extract(lambda _: httpx.Response(200, content=body))


def test_oversized_response():
    config = Settings(max_response_bytes=1024, hackernews={"enabled": True})
    with pytest.raises(SourceError, match="response_too_large"):
        extract(lambda _: httpx.Response(200, content=b"x" * 1025), settings_override=config)


def test_title_only_and_missing_link_fallback():
    raw = normalize_story({**STORY, "story_text": None, "url": None}, "Apple")
    assert raw.text == STORY["title"] and raw.metadata["content_kind"] == "title_only"
    assert raw.url == "https://news.ycombinator.com/item?id=12345"
    assert normalize_story({**STORY, "story_text": "<p> </p>"}, "Apple").text == STORY["title"]


@pytest.mark.parametrize("changes", [{"objectID": "bad"}, {"objectID": None}, {"_tags": ["comment"]}, {"deleted": True}, {"dead": True}])
def test_rejected_items(changes):
    assert normalize_story({**STORY, **changes}, "Apple").title is None


def test_missing_engagement_and_date_fallback():
    raw = normalize_story({**STORY, "created_at": None, "created_at_i": 0, "points": True, "num_comments": -1, "author": None}, "Apple")
    assert raw.published_at == "1970-01-01T00:00:00+00:00"
    assert raw.metadata["points"] is None and raw.metadata["num_comments"] is None
    assert raw.metadata["author"] is None
    assert normalize_story({**STORY, "created_at": "bad", "created_at_i": 10**100}, "Apple").published_at is None


@pytest.mark.parametrize("values", [{"keywords": []}, {"keywords": [" "]}, {"keywords": ["bad\nquery"]},
                                   {"hits_per_page": 101}, {"hits_per_page": 0}, {"max_pages": 0}, {"max_pages": 6},
                                   {"max_records": 1001}, {"timeout_seconds": 0}, {"timeout_seconds": float("inf")}])
def test_config_validation(values):
    with pytest.raises(ValidationError):
        settings(**values)


def test_defaults_and_environment(monkeypatch):
    assert Settings().hackernews.keywords == ["Apple", "Microsoft", "Tesla", "Nvidia", "Bitcoin"]
    assert not Settings().hackernews.enabled
    monkeypatch.setenv("INGESTION_HACKERNEWS_ENABLED", "true")
    monkeypatch.setenv("INGESTION_HACKERNEWS_KEYWORDS", '["Apple","apple","Bitcoin"]')
    monkeypatch.setenv("INGESTION_HACKERNEWS_HITS_PER_PAGE", "3")
    config = Settings.from_env().hackernews
    assert config.enabled and config.keywords == ["Apple", "Bitcoin"] and config.hits_per_page == 3


def test_disabled_provider_api():
    client = TestClient(create_app(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("network")))))
    response = client.post("/ingest/run", json={"providers": ["hackernews"]})
    assert response.status_code == 503
    assert response.json()["detail"]["sources"][0]["error_code"] == "provider_disabled"


def test_api_normalization_deduplication_and_insight_contracts():
    async def run():
        def handler(request):
            return httpx.Response(200, json=page([STORY, STORY, {**STORY, "objectID": "bad"}]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings(), http)), base_url="http://test") as api:
                response = await api.post("/ingest/run", json={"providers": ["hackernews"], "source_names": ["hackernews"]})
                assert response.status_code == 200
                body = response.json()
                assert body["diagnostics"]["deduplicated"] == 1 and body["diagnostics"]["rejected"] == 1
                doc = body["documents"][0]
                assert doc["source"] == "hackernews" and doc["tickers"] == ["AAPL"] and doc["language"] == "en"
                assert "<p>" not in doc["text"]
                assert doc["published_at"] == "2026-10-27T10:00:00Z"
                assert doc["metadata"]["hn_id"] == "12345"
                return body["documents"]
    documents = asyncio.run(run())
    code = """
import json,sys
from app.models.sentiment import AnalyzeRequest
from app.models.batch import BatchAnalyzeRequest
documents=json.load(sys.stdin)
for contract in (AnalyzeRequest,BatchAnalyzeRequest):
    parsed=contract.model_validate({'documents':documents})
    assert parsed.documents[0].tickers==['AAPL']
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2] / "insight-service",
                            input=json.dumps(documents), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_mixed_provider_partial_failure():
    async def run():
        config = Settings(sources=[{"name": "feed", "url": "https://feed.example/rss"}], hackernews={"enabled": True, "keywords": ["Apple"]})
        def handler(request):
            return httpx.Response(500) if request.url.host == "feed.example" else httpx.Response(200, json=page())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(config, http)), base_url="http://test") as api:
                response = await api.post("/ingest/run", json={"providers": ["rss", "hackernews"]})
                assert response.status_code == 200
                assert [s["status"] for s in response.json()["sources"]] == ["error", "success"]
    asyncio.run(run())


def test_scheduler_support():
    from app.core.scheduler import IngestionScheduler
    config = settings()
    config.scheduler.providers = ["hackernews"]
    scheduler = IngestionScheduler(config)
    assert scheduler.active_providers() == ["hackernews"]
    assert scheduler.interval("hackernews") == 900

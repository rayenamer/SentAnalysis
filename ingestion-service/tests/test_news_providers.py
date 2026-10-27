import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.core.extractors.news_providers import NewsExtractor
from app.core.extractors.news import SourceError
from app.main import create_app


ARTICLE = {"title": "Apple earnings $AAPL", "description": "<p>The company reports strong earnings and profits with revenue growth for investors in the market.</p>",
           "url": "https://example.com/a?utm_source=x", "publishedAt": "2026-10-27T12:00:00+02:00", "source": {"name": "Publisher"}}


def settings_for(provider, **extra):
    config = {"enabled": True, "keywords": ["earnings"], "tickers": ["AAPL"]}
    if provider == "newsapi":
        config["api_key"] = "secret-test-key"
    return Settings(**{provider: {**config, **extra}})


def extract(provider, handler, **overrides):
    async def run():
        settings = settings_for(provider, **overrides)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await NewsExtractor(client, settings, provider).fetch(None)
    return asyncio.run(run())


@pytest.mark.parametrize("provider", ["newsapi", "gdelt"])
def test_success_and_query(provider):
    def handler(request):
        assert "earnings" in request.url.params.get("q", request.url.params.get("query", ""))
        assert "AAPL" in str(request.url)
        if provider == "newsapi":
            assert request.headers["x-api-key"] == "secret-test-key"
            return httpx.Response(200, json={"status": "ok", "articles": [ARTICLE], "totalResults": 1})
        return httpx.Response(200, json={"articles": [{"title": "Financial report", "url": "https://example.com/g", "domain": "Publisher", "seendate": "20261027T120000Z", "language": "English"}]})
    records, truncated = extract(provider, handler)
    assert len(records) == 1 and not truncated
    assert records[0].metadata["publisher"] == "Publisher"
    if provider == "gdelt":
        assert records[0].text == records[0].title
        assert records[0].published_at is None


@pytest.mark.parametrize("provider", ["newsapi", "gdelt"])
@pytest.mark.parametrize("status,payload,code", [
    (429, {}, "rate_limited"), (500, {}, "http_error"), (401, {}, "authentication_failed"),
    (200, [], "invalid_response"), (200, {"status": "ok", "articles": {}}, "invalid_response"),
])
def test_failures(provider, status, payload, code):
    with pytest.raises(SourceError, match=code):
        extract(provider, lambda _: httpx.Response(status, json=payload))


@pytest.mark.parametrize("provider", ["newsapi", "gdelt"])
def test_invalid_json(provider):
    with pytest.raises(SourceError, match="invalid_json"):
        extract(provider, lambda _: httpx.Response(200, content=b"not JSON"))


@pytest.mark.parametrize("provider", ["newsapi", "gdelt"])
def test_timeout(provider):
    def handler(request):
        raise httpx.ReadTimeout("secret must not escape", request=request)
    with pytest.raises(SourceError, match="timeout"):
        extract(provider, handler)


@pytest.mark.parametrize("provider", ["newsapi", "gdelt"])
def test_empty(provider):
    assert extract(provider, lambda _: httpx.Response(200, json={"status": "ok", "articles": []}))[0] == []


def test_article_limits():
    records, truncated = extract("newsapi", lambda _: httpx.Response(200, json={"status": "ok", "articles": [ARTICLE] * 3, "totalResults": 3}), max_records=2)
    assert len(records) == 2 and truncated


def test_newsapi_body_rate_limit():
    with pytest.raises(SourceError, match="rate_limited"):
        extract("newsapi", lambda _: httpx.Response(200, json={"status": "error", "code": "rateLimited", "message": "secret"}))


def test_missing_credentials():
    with pytest.raises(SourceError, match="missing_credentials"):
        extract("newsapi", lambda _: pytest.fail("network"), api_key=None)


@pytest.mark.parametrize("config", [{"enabled": True}, {"enabled": True, "keywords": [" "]}, {"timeout_seconds": 0}, {"tickers": ['bad"query']}])
def test_invalid_config(config):
    with pytest.raises(ValidationError):
        Settings(newsapi=config)


def test_news_api_normalization_dedupe_and_partial():
    async def run():
        settings = Settings(newsapi={"enabled": True, "keywords": ["earnings"], "api_key": "secret"}, gdelt={"enabled": True, "keywords": ["earnings"]})
        def handler(request):
            if request.url.host == "api.gdeltproject.org":
                return httpx.Response(429)
            return httpx.Response(200, json={"status": "ok", "articles": [ARTICLE, ARTICLE]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as api:
                response = await api.post("/ingest/run", json={"providers": ["newsapi", "gdelt"]})
                assert response.status_code == 200
                body = response.json()
                assert body["diagnostics"]["deduplicated"] == 1
                assert body["documents"][0]["published_at"] == "2026-10-27T10:00:00Z"
                assert body["documents"][0]["tickers"] == ["AAPL"]
                assert body["documents"][0]["metadata"]["publisher"] == "Publisher"
                assert body["sources"][1]["error_code"] == "rate_limited"
                assert "secret" not in response.text
    asyncio.run(run())


def test_disabled_provider():
    response = TestClient(create_app(Settings())).post("/ingest/run", json={"providers": ["newsapi"]})
    assert response.status_code == 503
    assert response.json()["detail"]["sources"][0]["error_code"] == "provider_disabled"


def test_provider_env(monkeypatch):
    monkeypatch.setenv("INGESTION_NEWSAPI_API_KEY", "secret")
    monkeypatch.setenv("INGESTION_NEWSAPI_KEYWORDS", '["earnings"]')
    monkeypatch.setenv("INGESTION_NEWSAPI_ENABLED", "true")
    settings = Settings.from_env()
    assert settings.newsapi.enabled
    assert settings.newsapi.keywords == ["earnings"]
    assert "secret" not in repr(settings)

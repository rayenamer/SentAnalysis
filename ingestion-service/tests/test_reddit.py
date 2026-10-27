import asyncio

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings, FeedSource
from app.core.extractors.reddit import RedditExtractor
from app.helper.provider_http import SourceError
from app.main import create_app

POST = {"id": "abc", "title": "Apple $AAPL earnings", "selftext": "The company reports strong earnings and profits with revenue growth for investors in the market.",
        "created_utc": 1793102400, "subreddit": "stocks", "permalink": "/r/stocks/comments/abc/apple/", "score": 42, "num_comments": 7}


def listing(posts=None, after=None):
    return {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": p} for p in (posts or [])], "after": after}}


def reddit_settings(**overrides):
    return Settings(reddit={"enabled": True, "client_id": "client-test", "client_secret": "secret-test", "subreddits": ["stocks"], **overrides})


def extract(handler, **overrides):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await RedditExtractor(client, reddit_settings(**overrides)).fetch(FeedSource(name="reddit:stocks", url="https://oauth.reddit.com/r/stocks/new"))
    return asyncio.run(run())


def auth_handler(request):
    if request.url.path == "/api/v1/access_token":
        assert request.method == "POST"
        assert request.headers["authorization"].startswith("Basic ")
        assert b"grant_type=client_credentials" in request.content
        return httpx.Response(200, json={"access_token": "test-token", "token_type": "bearer"})


def test_success_and_pagination():
    calls = []
    def handler(request):
        auth = auth_handler(request)
        if auth is not None:
            return auth
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        if request.url.params.get("after"):
            return httpx.Response(200, json=listing([]))
        return httpx.Response(200, json=listing([POST], "t3_abc"))
    records, truncated = extract(handler)
    assert len(calls) == 2 and not truncated
    assert records[0].metadata == {"provider": "reddit", "post_id": "abc", "subreddit": "stocks", "score": 42, "num_comments": 7}
    assert records[0].published_at.endswith("+00:00")
    assert records[0].url == "https://www.reddit.com/r/stocks/comments/abc/apple/"


@pytest.mark.parametrize("status,code", [(429, "rate_limited"), (500, "http_error"), (403, "authentication_failed")])
def test_http_failures(status, code):
    def handler(request):
        return auth_handler(request) or httpx.Response(status)
    with pytest.raises(SourceError, match=code):
        extract(handler)


def test_timeout():
    def handler(request):
        if (auth := auth_handler(request)) is not None:
            return auth
        raise httpx.ReadTimeout("secret", request=request)
    with pytest.raises(SourceError, match="timeout"):
        extract(handler)


@pytest.mark.parametrize("body,code", [(b"not-json", "invalid_json"), (b'{}', "invalid_response"), (b'[]', "invalid_response")])
def test_invalid_responses(body, code):
    with pytest.raises(SourceError, match=code):
        extract(lambda r: auth_handler(r) or httpx.Response(200, content=body))


def test_empty():
    assert extract(lambda r: auth_handler(r) or httpx.Response(200, json=listing())) == ([], False)


@pytest.mark.parametrize("changes", [{"selftext": "[deleted]"}, {"selftext": "[removed]"}, {"created_utc": "bad"},
                                    {"created_utc": True}, {"created_utc": 1e100}, {"permalink": "https://evil.test"}, {"subreddit": "other"}])
def test_rejected_posts(changes):
    records, _ = extract(lambda r: auth_handler(r) or httpx.Response(200, json=listing([{**POST, **changes}])))
    assert records[0].title is None


def test_link_post_and_invalid_engagement():
    records, _ = extract(lambda r: auth_handler(r) or httpx.Response(200, json=listing([{**POST, "selftext": "", "num_comments": -1, "score": "bad"}])))
    assert records[0].text == POST["title"]
    assert records[0].metadata["score"] is None
    assert records[0].metadata["num_comments"] is None


def test_pagination_bound():
    calls = []
    def handler(r):
        if (auth := auth_handler(r)) is not None:
            return auth
        calls.append(r)
        return httpx.Response(200, json=listing([POST], "t3_abc"))
    records, truncated = extract(handler, max_pages=1)
    assert len(records) == 1 and truncated and len(calls) == 1


def test_repeated_cursor():
    with pytest.raises(SourceError, match="invalid_pagination"):
        extract(lambda r: auth_handler(r) or httpx.Response(200, json=listing([POST], "t3_abc")))


def test_missing_credentials():
    with pytest.raises(SourceError, match="missing_credentials"):
        extract(lambda r: pytest.fail("network"), client_secret=None)


@pytest.mark.parametrize("payload", [{"access_token": "bad\ntoken", "token_type": "bearer"}, {"error": "invalid_client"}])
def test_invalid_auth(payload):
    with pytest.raises(SourceError, match="authentication_failed"):
        extract(lambda r: httpx.Response(200, json=payload))


@pytest.mark.parametrize("values", [{"subreddits": ["../private"]}, {"page_size": 101}, {"max_pages": 6}, {"user_agent": "bad\nheader"}])
def test_config_validation(values):
    with pytest.raises(ValidationError):
        reddit_settings(**values)


def test_api_deduplication_and_engagement():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: auth_handler(r) or httpx.Response(200, json=listing([POST, POST, {**POST, "selftext": "[deleted]"}])))) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(reddit_settings(), http)), base_url="http://test") as api:
                response = await api.post("/ingest/run", json={"providers": ["reddit"]})
                assert response.status_code == 200
                payload = response.json()
                assert payload["diagnostics"]["deduplicated"] == 1
                assert payload["diagnostics"]["rejected"] == 1
                assert payload["documents"][0]["metadata"]["score"] == 42
                assert payload["documents"][0]["tickers"] == ["AAPL"]
                assert "secret-test" not in response.text and "test-token" not in response.text
    asyncio.run(run())

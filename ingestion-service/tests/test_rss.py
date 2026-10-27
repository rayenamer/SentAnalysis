import asyncio
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.core.extractors.news import RSSExtractor, SourceError, parse_feed
from app.core.transform import clean_text


def fetch(handler, **overrides):
    async def run():
        settings = Settings(sources=[{"name": "news", "url": "https://feeds.example/rss"}], **overrides)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await RSSExtractor(client, settings).fetch(settings.sources[0])
    return asyncio.run(run())


def test_rss(rss):
    records, truncated = fetch(lambda request: httpx.Response(200, content=rss))
    assert not truncated
    assert records[0].title == "Apple earnings $aapl"
    assert records[0].url == "https://example.com/story?utm_source=rss"


def test_atom():
    records, _ = parse_feed((Path(__file__).parent / "fixtures/atom.xml").read_bytes(), 10)
    assert records[0].url == "https://example.com/article"
    assert clean_text(records[0].text) == "Revenue rises. Profit grows."
    assert records[0].published_at == "2026-10-27T12:00:00-04:00"


@pytest.mark.parametrize("data,code", [
    (b"<rss>", "invalid_feed"), (b"<html/>", "invalid_feed"),
    (b'<?xml version="1.0" encoding="unknown-codec"?><rss/>', "invalid_feed"),
    (b"<rss>" + b"<a>" * 129 + b"</a>" * 129 + b"</rss>", "unsafe_xml"),
    (b'<!DOCTYPE rss [<!ENTITY x "EXPAND">]><rss><channel>&x;</channel></rss>', "unsafe_xml"),
    ('<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE rss><rss/>'.encode("utf-16"), "unsafe_xml"),
])
def test_invalid_xml(data, code):
    with pytest.raises(SourceError) as exc:
        parse_feed(data, 10)
    assert exc.value.code == code


def test_missing_fields():
    records, _ = parse_feed(b"<rss><channel><item><title>T</title></item></channel></rss>", 10)
    assert records[0].text is None
    assert records[0].url is None
    assert records[0].published_at is None


def test_entry_limit():
    records, truncated = parse_feed(b"<rss><channel>" + b"<item/>" * 10 + b"</channel></rss>", 2)
    assert len(records) == 2 and truncated


@pytest.mark.parametrize("status", [301, 404, 500])
def test_http_errors_no_redirects(status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "http://127.0.0.1/private"})
    with pytest.raises(SourceError, match="http_error"):
        fetch(handler)
    assert len(calls) == 1


def test_http_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(SourceError, match="timeout"):
        fetch(handler)


def test_overall_timeout():
    async def handler(request):
        await asyncio.sleep(0.05)
        return httpx.Response(200, content=b"<rss/>")
    with pytest.raises(SourceError, match="timeout"):
        fetch(handler, timeout_seconds=0.01)


@pytest.mark.parametrize("headers,body,code", [
    ({}, b"x" * 1025, "response_too_large"),
    ({"content-length": "2000"}, b"", "response_too_large"),
    ({"content-encoding": "gzip"}, b"", "unsupported_encoding"),
])
def test_response_bounds(headers, body, code):
    with pytest.raises(SourceError, match=code):
        fetch(lambda request: httpx.Response(200, headers=headers, content=body), max_response_bytes=1024)

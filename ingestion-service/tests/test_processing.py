from datetime import timezone

import pytest

from app.core.deduplication import Deduplicator, content_hash
from app.core.ticker_tagging import tag_tickers
from app.core.transform import canonical_url, clean_text, detect_language
from app.helper.datetime_utils import parse_utc


@pytest.mark.parametrize("value,expected", [
    (None, ""), ("  \n", ""), ("<p>A &amp; B</p><p>$AAPL €10 +2.5%</p>", "A & B $AAPL €10 +2.5%"),
    ("Cafe\u0301 — 東京", "Café — 東京"),
    ("<script>bad()</script><style>bad</style>Profit", "Profit"),
])
def test_cleaning(value, expected):
    assert clean_text(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("2026-10-27T12:00:00+02:00", "2026-10-27T10:00:00+00:00"),
    ("Tue, 27 Oct 2026 12:00:00 -0400", "2026-10-27T16:00:00+00:00"),
    ("2026-10-27T12:00:00Z", "2026-10-27T12:00:00+00:00"),
])
def test_dates(value, expected):
    parsed = parse_utc(value)
    assert parsed.tzinfo == timezone.utc
    assert parsed.isoformat() == expected


@pytest.mark.parametrize("value", [None, "", "invalid", "2026-10-27", "2026-10-27T12:00:00", "0001-01-01T00:00:00+14:00"])
def test_unknown_dates(value):
    assert parse_utc(value) is None


def test_url_dedupe_and_order():
    dedupe = Deduplicator()
    urls = ["https://EXAMPLE.com:443/article?utm_source=rss#section", "https://example.com/article"]
    assert canonical_url(urls[0]) == canonical_url(urls[1])
    assert not dedupe.duplicate(canonical_url(urls[0]), "Original report")
    assert dedupe.duplicate(canonical_url(urls[1]), "Updated report")
    assert dedupe.duplicate("https://example.com/other", " Original  report ")
    assert not dedupe.duplicate("https://example.com/different", "Different report")
    assert content_hash("A  B") == content_hash("A B")
    assert content_hash("US") != content_hash("us")


@pytest.mark.parametrize("value", ["javascript:alert(1)", "http://[invalid", "https://a:bad", "https://user:pass@example.com", "/relative", None])
def test_invalid_urls(value):
    assert canonical_url(value) is None


def test_url_preserves_identity_parameters():
    assert canonical_url("https://EXAMPLE.com/a/?id=2&x=1&utm_medium=test") == "https://example.com/a/?id=2&x=1"
    assert canonical_url("https://example.com/a") != canonical_url("https://example.com/a/")


@pytest.mark.parametrize("text,expected", [
    ("$aapl $MSFT $AAPL", ["AAPL", "MSFT"]),
    ("Apple Inc. and MICROSOFT report", ["AAPL", "MSFT"]),
    ("UnknownCorp and IT ON A stock rise", []),
    ("pineapple Micros oft", []),
    ("$brk.b $100 $ON", ["BRK.B", "ON"]),
])
def test_tickers(text, expected):
    assert tag_tickers(text, {"Apple Inc.": "AAPL", "Microsoft": "MSFT", "IT": "IT"}) == expected


@pytest.mark.parametrize("text,code", [
    ("The company reports strong earnings and profits with revenue growth for investors in the market.", "en"),
    ("Les investisseurs dans le marché et les actions avec une hausse des bénéfices pour entreprise.", "fr"),
    ("Los inversores en el mercado y las acciones con beneficios para una empresa con ingresos.", "es"),
    ("Strong earnings", "und"),
    ("未知のニュースです " * 15, "und"),
    ("alpha beta gamma delta " * 10, "und"),
])
def test_language(text, code):
    assert detect_language(text) == code

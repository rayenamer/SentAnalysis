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
from app.core.extractors.market import MarketExtractor, normalize_candles
from app.helper.provider_http import SourceError
from app.main import create_app
from app.models.market import MarketRequest
from app.provider_config import Instrument

NOW = datetime(2026, 10, 27, 16, tzinfo=timezone.utc)
ROW = {"datetime": "2026-10-27 12:00:00", "open": "100", "high": "110", "low": "90", "close": "105", "volume": "1200"}
INSTRUMENT = Instrument(ticker="AAPL", provider_symbol="AAPL")


def payload(rows=None, **extra):
    return {"status": "ok", "meta": {"symbol": "AAPL", "interval": "1h"}, "values": [ROW] if rows is None else rows, **extra}


def market_settings(**overrides):
    return Settings(market={"enabled": True, "api_key": "secret-market", "instruments": [INSTRUMENT], **overrides})


def extract(handler, **overrides):
    async def run():
        settings = market_settings(**overrides)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await MarketExtractor(client, settings).fetch(INSTRUMENT, NOW, settings.market.outputsize)
    return asyncio.run(run())


def test_success_and_auth():
    def handler(request):
        assert request.url.host == "api.twelvedata.com"
        assert request.headers["authorization"] == "apikey secret-market"
        assert "secret-market" not in str(request.url)
        assert request.url.params["timezone"] == "UTC"
        assert request.url.params["order"] == "asc"
        return httpx.Response(200, json=payload())
    rows, diagnostics = extract(handler)
    assert rows[0].timestamp.isoformat() == "2026-10-27T12:00:00+00:00"
    assert rows[0].close_timestamp.isoformat() == "2026-10-27T13:00:00+00:00"
    assert rows[0].close == 105 and rows[0].volume == 1200
    assert diagnostics.returned == 1


@pytest.mark.parametrize("asset,symbol,ticker", [("stock", "MSFT", "MSFT"), ("etf", "SPY", "SPY"), ("crypto", "BTC/USD", "BTC"), ("crypto", "ETH/USD", "ETH"), ("index", "SPX", "SPX")])
def test_asset_classes(asset, symbol, ticker):
    instrument = Instrument(ticker=ticker, provider_symbol=symbol, asset_type=asset)
    rows, _ = normalize_candles([ROW], instrument, "1h", NOW, 10)
    assert rows[0].ticker == ticker and rows[0].provider_symbol == symbol and rows[0].asset_type == asset


@pytest.mark.parametrize("status,code", [(429, "rate_limited"), (500, "http_error"), (401, "authentication_failed"), (403, "authentication_failed")])
def test_http_failures(status, code):
    with pytest.raises(SourceError, match=code):
        extract(lambda _: httpx.Response(status, json={"message": "secret-market"}))


@pytest.mark.parametrize("code,error", [(429, "rate_limited"), (400, "invalid_symbol"), (403, "access_denied"), (500, "provider_error")])
def test_provider_errors(code, error):
    with pytest.raises(SourceError, match=error):
        extract(lambda _: httpx.Response(200, json={"status": "error", "code": code, "message": "secret-market"}))


def test_timeout():
    def handler(request):
        raise httpx.ReadTimeout("secret-market", request=request)
    with pytest.raises(SourceError, match="timeout"):
        extract(handler)


@pytest.mark.parametrize("body,error", [(b"broken", "invalid_json"), (b'[]', "invalid_response"), (b'{}', "invalid_response"),
                                         (b'{"status":"ok","values":{}}', "invalid_response")])
def test_invalid_responses(body, error):
    with pytest.raises(SourceError, match=error):
        extract(lambda _: httpx.Response(200, content=body))


def test_symbol_response_mismatch():
    with pytest.raises(SourceError, match="invalid_response"):
        extract(lambda _: httpx.Response(200, json=payload(meta={"symbol": "OTHER", "interval": "1h"})))


def test_empty():
    rows, diagnostics = extract(lambda _: httpx.Response(200, json=payload([])))
    assert rows == [] and diagnostics.collected == 0


@pytest.mark.parametrize("changes", [{"close": "NaN"}, {"open": "Infinity"}, {"close": 0}, {"low": -1},
                                    {"close": True}, {"volume": True}, {"volume": -1}, {"high": 95},
                                    {"low": 106}, {"close": None}, {"datetime": "invalid"},
                                    {"datetime": "2026-10-27"}, {"datetime": "9999-12-31T23:59:59Z"}])
def test_reject_bad_candles(changes):
    rows, diagnostics = normalize_candles([{**ROW, **changes}], INSTRUMENT, "1h", NOW, 10)
    assert rows == [] and diagnostics.rejected == 1


def test_missing_volume_and_gaps():
    rows, diagnostics = normalize_candles([{**ROW, "datetime": "2026-10-27 14:00:00", "volume": None}, ROW], INSTRUMENT, "1h", NOW, 10)
    assert [row.timestamp.hour for row in rows] == [12, 14]
    assert diagnostics.gaps == 1 and diagnostics.missing_volume == 1
    assert rows[1].volume is None


def test_duplicates_and_conflicts():
    rows, diagnostics = normalize_candles([ROW, ROW, {**ROW, "close": 106}, ROW], INSTRUMENT, "1h", NOW, 10)
    assert rows == []
    assert diagnostics.deduplicated == 1 and diagnostics.conflicting_timestamps == 1 and diagnostics.rejected == 3


def test_incomplete_and_limits():
    rows, diagnostics = normalize_candles([{**ROW, "datetime": "2026-10-27 16:00:00"}, ROW], INSTRUMENT, "1h", NOW, 1)
    assert not rows and diagnostics.incomplete == 1 and diagnostics.limited == 1


def test_timezone_normalization():
    rows, _ = normalize_candles([{**ROW, "datetime": "2026-10-27T14:00:00+02:00"}], INSTRUMENT, "1h", NOW, 10)
    assert rows[0].timestamp.hour == 12


@pytest.mark.parametrize("config", [{"interval": "1day"}, {"outputsize": 1001}, {"timeout_seconds": 0},
                                   {"instruments": [{"ticker": "BAD SYMBOL", "provider_symbol": "BAD"}]},
                                   {"instruments": [INSTRUMENT, INSTRUMENT]}])
def test_invalid_configuration(config):
    with pytest.raises(ValidationError):
        market_settings(**config)


def test_credentials():
    with pytest.raises(SourceError, match="missing_credentials"):
        extract(lambda _: pytest.fail("network"), api_key=None)


def test_api_partial_results():
    async def run():
        settings = market_settings(instruments=[INSTRUMENT, {"ticker": "BTC", "provider_symbol": "BTC/USD", "asset_type": "crypto"}])
        def handler(request):
            return httpx.Response(200, json=payload()) if request.url.params["symbol"] == "AAPL" else httpx.Response(429)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(settings, http)), base_url="http://test") as api:
                result = await api.post("/ingest/market", json={})
                assert result.status_code == 200
                body = result.json()
                assert body["diagnostics"]["returned"] == 1
                assert [s["status"] for s in body["sources"]] == ["success", "error"]
                assert "secret-market" not in result.text
    asyncio.run(run())


@pytest.mark.parametrize("body", [{"symbols": ["UNKNOWN"]}, {"symbols": []}, {"max_candles": 0}, {"max_candles": True}, {"url": "http://internal"}])
def test_api_validation(body):
    client = TestClient(create_app(market_settings(), httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("network")))))
    assert client.post("/ingest/market", json=body).status_code == 422


def test_api_disabled():
    response = TestClient(create_app(Settings())).post("/ingest/market", json={})
    assert response.status_code == 503 and response.json()["detail"]["code"] == "provider_disabled"


def test_api_total_failure_and_selection():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(429))) as http:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(market_settings(), http)), base_url="http://test") as api:
                response = await api.post("/ingest/market", json={"symbols": ["aapl"]})
                assert response.status_code == 503
                assert response.json()["detail"]["sources"][0]["error_code"] == "rate_limited"
    asyncio.run(run())


def test_environment(monkeypatch):
    monkeypatch.setenv("INGESTION_MARKET_API_KEY", "secret-market")
    monkeypatch.setenv("INGESTION_MARKET_INSTRUMENTS", '[{"ticker":"btc","provider_symbol":"BTC/USD","asset_type":"crypto"}]')
    settings = Settings.from_env()
    assert settings.market.instruments[0].ticker == "BTC"
    assert "secret-market" not in repr(settings)


def test_insight_correlation_and_granger_contracts():
    candles, _ = normalize_candles([ROW], INSTRUMENT, "1h", NOW, 10)
    data = [c.insight_price() for c in candles]
    code = """
import json, sys
from app.models.correlation import PriceObservation, CorrelationRequest
from app.models.granger import GrangerRequest
prices = json.load(sys.stdin)
assert PriceObservation.model_validate(prices[0]).close_price == 105.0
for contract in (CorrelationRequest, GrangerRequest):
    result = contract.model_validate({'sentiments': [], 'prices': prices, 'time_bucket': 'hourly'})
    assert result.prices[0].timestamp.hour == 13
"""
    service = Path(__file__).resolve().parents[2] / "insight-service"
    result = subprocess.run([sys.executable, "-c", code], input=json.dumps(data), cwd=service,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr

"""Twelve Data intraday OHLCV; no interpolation or calendar assumptions."""
from datetime import datetime, timedelta, timezone
import re

from pydantic import ValidationError

from app.helper.datetime_utils import parse_utc
from app.helper.provider_http import SourceError, fetch_json, secret_value
from app.models.market import Candle, MarketDiagnostics, MarketResponse, MarketSourceResult

MARKET_URL = "https://api.twelvedata.com/time_series"
INTERVALS = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800, "1h": 3600}


class MarketExtractor:
    def __init__(self, client, settings):
        self.client, self.settings = client, settings

    async def fetch(self, instrument, now, limit):
        config = self.settings.market
        key = secret_value(config.api_key)
        payload = await fetch_json(self.client, "GET", MARKET_URL, timeout=config.timeout_seconds,
                                   max_bytes=self.settings.max_response_bytes,
                                   headers={"Authorization": "apikey " + key},
                                   params={"symbol": instrument.provider_symbol, "interval": config.interval, "timezone": "UTC",
                                           "outputsize": limit, "order": "asc", "format": "JSON"})
        if payload.get("status") == "error":
            code = str(payload.get("code"))
            raise SourceError({"429": "rate_limited", "400": "invalid_symbol", "401": "authentication_failed", "403": "access_denied"}.get(code, "provider_error"))
        if payload.get("status") != "ok" or not isinstance(payload.get("values"), list):
            raise SourceError("invalid_response")
        meta = payload.get("meta")
        if (not isinstance(meta, dict) or meta.get("symbol") != instrument.provider_symbol
                or meta.get("interval") != config.interval):
            raise SourceError("invalid_response")
        return normalize_candles(payload["values"], instrument, config.interval, now, limit)


def normalize_candles(values, instrument, interval, now, limit):
    diagnostics = MarketDiagnostics(collected=min(len(values), limit), limited=max(0, len(values) - limit))
    selected, conflicts = {}, set()
    delta = timedelta(seconds=INTERVALS[interval])
    for raw in values[:limit]:
        try:
            if not isinstance(raw, dict) or not isinstance(raw.get("datetime"), str):
                raise ValueError("missing datetime")
            value = raw["datetime"]
            # UTC was requested explicitly for intraday data. Only interpret this
            # provider's documented naive format as UTC; general dates stay strict.
            if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
                value = value.replace(" ", "T") + "+00:00"
            timestamp = parse_utc(value)
            if timestamp is None:
                raise ValueError("unknown timestamp")
            close_timestamp = timestamp + delta
            candle = Candle(ticker=instrument.ticker, provider_symbol=instrument.provider_symbol,
                            asset_type=instrument.asset_type, timestamp=timestamp, close_timestamp=close_timestamp,
                            interval=interval, **{field: raw.get(field) for field in ("open", "high", "low", "close", "volume")})
        except (ValueError, TypeError, OverflowError, ValidationError):
            diagnostics.rejected += 1
            continue
        if candle.close_timestamp > now:
            diagnostics.incomplete += 1
            continue
        if candle.timestamp in conflicts:
            diagnostics.rejected += 1
        elif candle.timestamp in selected:
            if selected[candle.timestamp] == candle:
                diagnostics.deduplicated += 1
            else:
                del selected[candle.timestamp]
                conflicts.add(candle.timestamp)
                diagnostics.conflicting_timestamps += 1
                diagnostics.rejected += 2
        else:
            selected[candle.timestamp] = candle
    candles = sorted(selected.values(), key=lambda c: c.timestamp)
    diagnostics.returned = len(candles)
    diagnostics.missing_volume = sum(c.volume is None for c in candles)
    diagnostics.gaps = sum(max(0, int((b.timestamp - a.timestamp) / delta) - 1) for a, b in zip(candles, candles[1:]))
    return candles, diagnostics


async def ingest_market(settings, request, client):
    config = settings.market
    if not config.enabled:
        raise ValueError("provider_disabled")
    instruments = config.instruments
    if request.symbols is not None:
        names = set(request.symbols)
        if names - {instrument.ticker for instrument in instruments}:
            raise ValueError("unknown_symbol")
        instruments = [instrument for instrument in instruments if instrument.ticker in names]
    if not instruments:
        raise ValueError("no_symbols_configured")
    now = datetime.now(timezone.utc)
    limit = min(request.max_candles or config.outputsize, config.outputsize)
    extractor = MarketExtractor(client, settings)
    candles, sources = [], []
    totals = MarketDiagnostics()
    for instrument in instruments:
        try:
            rows, diagnostics = await extractor.fetch(instrument, now, limit)
        except SourceError as exc:
            sources.append(MarketSourceResult(ticker=instrument.ticker, provider_symbol=instrument.provider_symbol,
                                             status="error", error_code=exc.code))
            continue
        candles.extend(rows)
        sources.append(MarketSourceResult(ticker=instrument.ticker, provider_symbol=instrument.provider_symbol,
                                         status="success", diagnostics=diagnostics))
        for field in MarketDiagnostics.model_fields:
            setattr(totals, field, getattr(totals, field) + getattr(diagnostics, field))
    return MarketResponse(collected_at=now, candles=candles, sources=sources, diagnostics=totals)

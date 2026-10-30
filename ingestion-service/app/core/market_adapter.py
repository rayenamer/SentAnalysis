"""Explicit projection only when actual closed UTC bucket observations exist."""
from datetime import datetime, timezone


def price_observations(candles, time_bucket="hourly", *, available_at=None):
    if time_bucket not in {"hourly", "daily"}:
        raise ValueError("unsupported_time_bucket")
    now = available_at or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("availability_requires_timezone")
    prices = {}
    for candle in candles:
        timestamp = candle.close_timestamp.astimezone(timezone.utc)
        if timestamp > now:
            raise ValueError("incomplete_candle")
        if timestamp.minute or timestamp.second or timestamp.microsecond or (time_bucket == "daily" and timestamp.hour):
            raise ValueError("unaligned_bucket_close")
        row = candle.insight_price()
        key = (candle.ticker, timestamp)
        if key in prices and prices[key] != row:
            raise ValueError("conflicting_price_observations")
        prices[key] = row
    return [prices[key] for key in sorted(prices)]

"""UTC normalization suitable for Pydantic field validators."""

from datetime import datetime, timezone


def normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    try:
        return value.astimezone(timezone.utc)
    except OverflowError as exc:
        # ValueError becomes a structured Pydantic validation error; a raw
        # OverflowError escapes validation and otherwise causes HTTP 500.
        raise ValueError("timestamp cannot be represented within UTC datetime bounds") from exc

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = parsedate_to_datetime(value)
        # Missing timezone is unknown, never silently interpreted as local time.
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None

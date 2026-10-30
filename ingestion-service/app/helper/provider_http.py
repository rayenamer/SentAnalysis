"""Bounded provider HTTP; never return remote bodies, exception strings or URLs."""
import asyncio
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx


class SourceError(Exception):
    def __init__(self, code, *, status_code=None, retry_after_seconds=None, retryable=False):
        self.code = code
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.retryable = retryable
        super().__init__(code)


def retry_after(value):
    if not value:
        return None
    if value.isdecimal():
        return 86400 * 365 if len(value) > 8 else min(int(value), 86400 * 365)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        return max(0, (date - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None


async def fetch_bytes(client, method, url, *, timeout, max_bytes, **kwargs):
    try:
        async with asyncio.timeout(timeout):
            headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
            async with client.stream(method, url, timeout=timeout, follow_redirects=False, headers=headers, **kwargs) as response:
                if response.status_code == 429:
                    raise SourceError("rate_limited", status_code=429, retry_after_seconds=retry_after(response.headers.get("retry-after")))
                if response.status_code in {401, 403}:
                    raise SourceError("authentication_failed", status_code=response.status_code)
                if not 200 <= response.status_code < 300:
                    raise SourceError("http_error", status_code=response.status_code,
                                      retry_after_seconds=retry_after(response.headers.get("retry-after")))
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise SourceError("unsupported_encoding")
                size = response.headers.get("content-length", "")
                if size.isdecimal() and int(size) > max_bytes:
                    raise SourceError("response_too_large")
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if len(data) + len(chunk) > max_bytes:
                        raise SourceError("response_too_large")
                    data.extend(chunk)
                return bytes(data)
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise SourceError("timeout", retryable=True) from exc
    except httpx.HTTPError as exc:
        raise SourceError("http_error", retryable=isinstance(exc, httpx.NetworkError)) from exc


async def fetch_json(client, method, url, *, timeout, max_bytes, **kwargs):
    data = await fetch_bytes(client, method, url, timeout=timeout, max_bytes=max_bytes, **kwargs)
    try:
        result = json.loads(data, parse_constant=lambda _: None)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SourceError("invalid_json") from exc
    if not isinstance(result, dict):
        raise SourceError("invalid_response")
    return result


def secret_value(secret):
    if secret is None or not secret.get_secret_value().strip():
        raise SourceError("missing_credentials")
    value = secret.get_secret_value()
    if any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise SourceError("invalid_credentials")
    return value


def string(value):
    return value if isinstance(value, str) else None

"""Bounded provider HTTP; never return remote bodies, exception strings or URLs."""
import asyncio
import json

import httpx


class SourceError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


async def fetch_bytes(client, method, url, *, timeout, max_bytes, **kwargs):
    try:
        async with asyncio.timeout(timeout):
            headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
            async with client.stream(method, url, timeout=timeout, follow_redirects=False, headers=headers, **kwargs) as response:
                if response.status_code == 429:
                    raise SourceError("rate_limited")
                if response.status_code in {401, 403}:
                    raise SourceError("authentication_failed")
                if not 200 <= response.status_code < 300:
                    raise SourceError("http_error")
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
        raise SourceError("timeout") from exc
    except httpx.HTTPError as exc:
        raise SourceError("http_error") from exc


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

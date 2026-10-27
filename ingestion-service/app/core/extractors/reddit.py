"""Official Reddit OAuth application-only read access; request-local token."""
from datetime import datetime, timezone
import math
import re

import httpx

from app.core.extractors.news import RawDocument
from app.helper.provider_http import SourceError, fetch_json, secret_value, string

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_URL = "https://oauth.reddit.com"


class RedditExtractor:
    def __init__(self, client, settings):
        self.client, self.settings = client, settings
        self.token = None

    async def request(self, method, url, **kwargs):
        config = self.settings.reddit
        return await fetch_json(self.client, method, url, timeout=config.timeout_seconds,
                                max_bytes=self.settings.max_response_bytes, **kwargs)

    async def authenticate(self):
        config = self.settings.reddit
        if self.token is None:
            payload = await self.request("POST", TOKEN_URL,
                                         auth=httpx.BasicAuth(secret_value(config.client_id), secret_value(config.client_secret)),
                                         data={"grant_type": "client_credentials"}, headers={"User-Agent": config.user_agent})
            token = payload.get("access_token")
            if (not isinstance(token, str) or not token or len(token) > 4096
                    or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token)
                    or str(payload.get("token_type", "")).lower() != "bearer"):
                raise SourceError("authentication_failed")
            self.token = token
        return self.token

    async def fetch(self, source):
        config = self.settings.reddit
        if not config.enabled:
            raise SourceError("provider_disabled")
        token = await self.authenticate()
        subreddit = source.name.split(":", 1)[1]
        records, cursor, seen = [], None, set()
        for _ in range(config.max_pages):
            params = {"limit": config.page_size, "raw_json": 1}
            if cursor:
                params["after"] = cursor
            payload = await self.request("GET", f"{OAUTH_URL}/r/{subreddit}/new", params=params,
                                         headers={"Authorization": f"Bearer {token}", "User-Agent": config.user_agent})
            data = payload.get("data")
            if payload.get("kind") != "Listing" or not isinstance(data, dict) or not isinstance(data.get("children"), list):
                raise SourceError("invalid_response")
            children = data["children"]
            available = min(config.page_size, self.settings.max_entries_per_source - len(records))
            for child in children[:available]:
                post = child.get("data") if isinstance(child, dict) and child.get("kind") == "t3" else None
                records.append(normalize_post(post, subreddit))
            cursor = data.get("after")
            if cursor is not None and (not isinstance(cursor, str) or not re.fullmatch(r"t3_[a-z0-9]+", cursor) or cursor in seen):
                raise SourceError("invalid_pagination")
            if cursor:
                seen.add(cursor)
            if len(children) > available:
                return records, True
            if not cursor:
                return records, False
            if len(records) >= self.settings.max_entries_per_source:
                return records, True
        return records, bool(cursor)


def normalize_post(post, subreddit):
    rejected = RawDocument(None, None, None, None)
    if not isinstance(post, dict) or post.get("removed_by_category") or post.get("selftext") == "[deleted]" or post.get("selftext") == "[removed]":
        return rejected
    post_id, permalink = post.get("id"), post.get("permalink")
    if (not isinstance(post_id, str) or not re.fullmatch(r"[a-z0-9]+", post_id)
            or not isinstance(permalink, str) or not re.fullmatch(r"/r/[A-Za-z0-9_]+/comments/" + post_id + r"/[^?#\s]*", permalink)
            or str(post.get("subreddit", "")).lower() != subreddit):
        return rejected
    created = post.get("created_utc")
    try:
        if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
            return rejected
        published = datetime.fromtimestamp(created, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return rejected
    title, text = string(post.get("title")), string(post.get("selftext"))
    metadata = {"provider": "reddit", "post_id": post_id, "subreddit": subreddit}
    for field in ("score", "num_comments"):
        value = post.get(field)
        metadata[field] = value if type(value) is int and (field == "score" or value >= 0) else None
    if not text:
        text = title
        metadata["content_kind"] = "title_only"
    return RawDocument(title, text, "https://www.reddit.com" + permalink, published, metadata)

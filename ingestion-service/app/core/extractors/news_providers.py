"""NewsAPI Everything and GDELT DOC 2 adapters; no article-page scraping."""
from app.core.extractors.news import RawDocument, SourceError
from app.helper.provider_http import fetch_json, secret_value, string

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class NewsExtractor:
    def __init__(self, client, settings, provider):
        self.client, self.settings, self.provider = client, settings, provider

    async def fetch(self, source):
        config = getattr(self.settings, self.provider)
        if not config.enabled:
            raise SourceError("provider_disabled")
        limit = min(config.max_records, self.settings.max_entries_per_source)
        try:
            if self.provider == "newsapi":
                headers = {"X-Api-Key": secret_value(config.api_key)}
                params = {"q": config.query, "pageSize": limit, "page": 1, "sortBy": "publishedAt"}
                if config.language:
                    params["language"] = config.language
                payload = await fetch_json(self.client, "GET", NEWSAPI_URL, timeout=config.timeout_seconds,
                                           max_bytes=self.settings.max_response_bytes, headers=headers, params=params)
                if payload.get("status") != "ok":
                    raise SourceError("rate_limited" if payload.get("code") == "rateLimited" else "provider_error")
            else:
                payload = await fetch_json(self.client, "GET", GDELT_URL, timeout=config.timeout_seconds,
                                           max_bytes=self.settings.max_response_bytes,
                                           params={"query": "(" + config.query + ")", "mode": "artlist", "format": "json",
                                                   "maxrecords": limit, "timespan": config.timespan, "sort": "datedesc"})
        except SourceError:
            raise
        articles = payload.get("articles")
        if not isinstance(articles, list):
            raise SourceError("invalid_response")
        records = []
        for article in articles[:limit]:
            if not isinstance(article, dict):
                records.append(RawDocument(None, None, None, None))
                continue
            metadata = {"provider": self.provider}
            if self.provider == "newsapi":
                publisher = article.get("source")
                metadata["publisher"] = string(publisher.get("name")) if isinstance(publisher, dict) else None
                metadata["source_language"] = config.language
                text = string(article.get("content")) or string(article.get("description"))
                published = string(article.get("publishedAt"))
            else:
                metadata.update(publisher=string(article.get("domain")), source_language=string(article.get("language")),
                                seen_at=string(article.get("seendate")), content_kind="title_only")
                # DOC artlist provides titles, not article bodies or publication dates.
                text = string(article.get("title"))
                published = None
            records.append(RawDocument(string(article.get("title")), text, string(article.get("url")), published, metadata))
        total = payload.get("totalResults")
        truncated = len(articles) > limit or (self.provider == "newsapi" and isinstance(total, int) and total > len(articles))
        # GDELT has no pagination/total count: reaching its cap may omit results.
        truncated |= self.provider == "gdelt" and len(articles) >= limit
        return records, truncated

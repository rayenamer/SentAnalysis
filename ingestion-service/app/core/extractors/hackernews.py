"""Public HN Algolia stories, without API keys or article-page scraping."""
from datetime import datetime, timezone
import re

from app.core.extractors.news import RawDocument
from app.core.transform import canonical_url, clean_text
from app.helper.datetime_utils import parse_utc
from app.helper.provider_http import SourceError, fetch_json, string

HACKERNEWS_URL = "https://hn.algolia.com/api/v1/search_by_date"


def normalize_story(hit, keyword):
    rejected = RawDocument(None, None, None, None)
    if not isinstance(hit, dict) or hit.get("deleted") or hit.get("dead"):
        return rejected
    hn_id = hit.get("objectID")
    if not isinstance(hn_id, str) or not re.fullmatch(r"[0-9]{1,20}", hn_id):
        return rejected
    # The request filters stories; reject explicit contradictory item tags too.
    tags = hit.get("_tags")
    if tags is not None and (not isinstance(tags, list) or "story" not in tags):
        return rejected
    title, story_text = string(hit.get("title")), string(hit.get("story_text"))
    hn_url = "https://news.ycombinator.com/item?id=" + hn_id
    article_url = string(hit.get("url"))
    url = article_url if canonical_url(article_url) else hn_url
    published = parse_utc(string(hit.get("created_at")))
    if published is None and type(hit.get("created_at_i")) is int:
        try:
            published = datetime.fromtimestamp(hit["created_at_i"], timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
    metadata = {"provider": "hackernews", "hn_id": hn_id, "hn_url": hn_url,
                "article_url": article_url, "author": string(hit.get("author")), "search_keyword": keyword}
    for field in ("points", "num_comments"):
        value = hit.get(field)
        metadata[field] = value if type(value) is int and value >= 0 else None
    if not clean_text(story_text):
        story_text = title
        metadata["content_kind"] = "title_only"
    return RawDocument(title, story_text, url, published.isoformat() if published else None, metadata)


class HackerNewsExtractor:
    def __init__(self, client, settings):
        self.client, self.settings = client, settings

    async def fetch(self, source):
        config = self.settings.hackernews
        if not config.enabled:
            raise SourceError("provider_disabled")
        limit = min(config.max_records, self.settings.max_entries_per_source)
        # Keep page sizes constant within each query to avoid pagination skips.
        page_size = min(config.hits_per_page, limit)
        records, truncated = [], False
        for keyword_index, keyword in enumerate(config.keywords):
            for page in range(config.max_pages):
                payload = await fetch_json(self.client, "GET", HACKERNEWS_URL,
                                           timeout=config.timeout_seconds, max_bytes=self.settings.max_response_bytes,
                                           params={"query": keyword, "tags": "story", "hitsPerPage": page_size, "page": page,
                                                   "typoTolerance": "false", "queryType": "prefixNone"})
                hits, pages, returned_page = payload.get("hits"), payload.get("nbPages"), payload.get("page")
                if (not isinstance(hits, list) or type(pages) is not int or pages < 0
                        or type(returned_page) is not int or returned_page != page
                        or (hits and pages <= page)):
                    raise SourceError("invalid_response")
                remaining = limit - len(records)
                accepted = hits[:min(page_size, remaining)]
                records.extend(normalize_story(hit, keyword) for hit in accepted)
                more_pages = page + 1 < pages
                truncated |= len(hits) > len(accepted)
                if len(records) >= limit:
                    return records, truncated or more_pages or keyword_index + 1 < len(config.keywords)
                if not hits or not more_pages:
                    break
                if page + 1 == config.max_pages:
                    truncated = True
        return records, truncated

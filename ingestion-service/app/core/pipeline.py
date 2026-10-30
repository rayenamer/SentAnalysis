from datetime import datetime, timezone
from app.helper.metrics import record

from app.core.deduplication import Deduplicator, content_hash
from app.core.extractors.news import RSSExtractor, SourceError
from app.core.ticker_tagging import tag_tickers
from app.core.transform import canonical_url, clean_text, detect_language
from app.helper.datetime_utils import parse_utc
from app.models.ingestion import Diagnostics, Document, IngestResponse, SourceResult
from app.config import FeedSource
from app.core.extractors.news_providers import NewsExtractor, NEWSAPI_URL, GDELT_URL
from app.core.extractors.hackernews import HackerNewsExtractor, HACKERNEWS_URL


async def ingest(settings, request, client):
    now = datetime.now(timezone.utc)
    rss = RSSExtractor(client, settings)
    jobs = [(s, rss) for s in settings.sources] if "rss" in request.providers else []
    for provider, endpoint in [("newsapi", NEWSAPI_URL), ("gdelt", GDELT_URL)]:
        if provider in request.providers:
            jobs.append((FeedSource(name=provider, url=endpoint), NewsExtractor(client, settings, provider)))
    if "hackernews" in request.providers:
        jobs.append((FeedSource(name="hackernews", url=HACKERNEWS_URL), HackerNewsExtractor(client, settings)))
    sources = [s for s, _ in jobs]
    if request.source_names is not None:
        names = set(request.source_names)
        if names - {s.name for s in sources}:
            raise ValueError("unknown_source")
        jobs = [(s, extractor) for s, extractor in jobs if s.name in names]
    if not jobs:
        raise ValueError("no_sources_configured")
    limit = min(request.max_documents or settings.max_documents, settings.max_documents)
    diagnostics = Diagnostics()
    results, documents = [], []
    dedupe = Deduplicator()
    for source, extractor in jobs:
        try:
            records, truncated = await extractor.fetch(source)
        except SourceError as exc:
            provider = "rss" if isinstance(extractor, RSSExtractor) else source.name
            record("provider_failures_total", provider)
            if "timeout" in exc.code:
                record("provider_timeouts_total", provider)
            results.append(SourceResult(name=source.name, url=source.url, status="error", error_code=exc.code,
                                        retry_after_seconds=exc.retry_after_seconds))
            continue
        results.append(SourceResult(name=source.name, url=source.url, status="success", collected=len(records), entries_truncated=truncated))
        diagnostics.collected += len(records)
        record("documents_collected_total", "rss" if isinstance(extractor, RSSExtractor) else source.name, len(records))
        for raw in records:
            title, text = clean_text(raw.title), clean_text(raw.text)
            if not title or not text:
                diagnostics.rejected += 1
                continue
            url = canonical_url(raw.url)
            if dedupe.duplicate(url, text):
                diagnostics.deduplicated += 1
                continue
            if len(documents) >= limit:
                diagnostics.limited += 1
                continue
            published = parse_utc(raw.published_at)
            if published is None:
                diagnostics.published_at_missing_or_invalid += 1
            documents.append(Document(document_id=content_hash(url or text), title=title, text=text,
                                      language=detect_language(title + " " + text), tickers=tag_tickers(title + " " + text, settings.company_tickers),
                                      source=source.name, original_url=raw.url, canonical_url=url, collected_at=now, published_at=published,
                                      metadata=raw.metadata))
    diagnostics.returned = len(documents)
    record("documents_accepted_total", "news", len(documents))
    return IngestResponse(collected_at=now, documents=documents, sources=results, diagnostics=diagnostics)

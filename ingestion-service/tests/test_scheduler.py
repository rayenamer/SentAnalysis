import asyncio
from datetime import datetime, timezone
import json
import logging

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.scheduler import IngestionScheduler
from app.helper.scheduler_lock import SchedulerLease
from app.integration_config import SchedulerConfig
from app.main import create_app
from app.models.ingestion import Diagnostics, IngestResponse, SourceResult
from app.models.market import MarketDiagnostics, MarketResponse, MarketSourceResult
from fastapi.testclient import TestClient


def news_result(errors=()):
    return IngestResponse(collected_at=datetime.now(timezone.utc), documents=[],
                          diagnostics=Diagnostics(collected=2), sources=[SourceResult(name="feed", url="https://feed.example", status="success")] +
                          [SourceResult(name=str(index), url="https://feed.example", status="error", error_code=code,
                                        retry_after_seconds=600 if code == "rate_limited" else None) for index, code in enumerate(errors)])


def settings(**overrides):
    return Settings(sources=[{"name": "feed", "url": "https://feed.example"}], scheduler={"enabled": True, **overrides})


def test_disabled_and_duplicate_start(tmp_path):
    async def run():
        disabled = IngestionScheduler(Settings(), lease=SchedulerLease(tmp_path / "disabled.lock"))
        assert not await disabled.start() and disabled.task is None
        assert not (tmp_path / "disabled.lock").exists()
        scheduler = IngestionScheduler(settings(), lease=SchedulerLease(tmp_path / "enabled.lock"))
        assert await scheduler.start()
        task = scheduler.task
        assert not await scheduler.start() and task is scheduler.task
        await scheduler.stop()
        assert task.done() and scheduler.client is None
    asyncio.run(run())


def test_lease_excludes_other_instances_and_releases(tmp_path):
    first, second = SchedulerLease(tmp_path / "lease.lock"), SchedulerLease(tmp_path / "lease.lock")
    assert first.acquire()
    try:
        assert not second.acquire()
    finally:
        first.release()
    assert second.acquire()
    second.release()


def test_multiple_workers_only_owner_starts(tmp_path):
    async def run():
        owner = IngestionScheduler(settings(), lease=SchedulerLease(tmp_path / "same.lock"))
        standby = IngestionScheduler(settings(), lease=SchedulerLease(tmp_path / "same.lock"))
        assert await owner.start()
        assert not await standby.start()
        assert standby.task is None
        await standby.stop()
        await owner.stop()
    asyncio.run(run())


def test_repeated_execution_without_overlap(monkeypatch, tmp_path):
    calls, running = [], []
    completed = asyncio.Event()
    async def ingest(*args):
        assert not running
        running.append(True)
        await asyncio.sleep(.01)
        calls.append(True)
        if len(calls) >= 2:
            completed.set()
        running.clear()
        return news_result()
    monkeypatch.setattr("app.core.scheduler.ingest", ingest)
    async def run():
        scheduler = IngestionScheduler(settings(), lease=SchedulerLease(tmp_path / "loop.lock"))
        scheduler.interval = lambda provider: .01
        assert await scheduler.start()
        assert "rss" in scheduler.next_due
        await asyncio.wait_for(completed.wait(), timeout=3)
        await scheduler.stop()
        assert len(calls) >= 2
    asyncio.run(run())


def test_overlapping_jobs_skipped(monkeypatch):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def ingest(*args):
            entered.set()
            await release.wait()
            return news_result()
        monkeypatch.setattr("app.core.scheduler.ingest", ingest)
        scheduler = IngestionScheduler(settings())
        first = asyncio.create_task(scheduler.run_provider("rss"))
        await entered.wait()
        assert await scheduler.run_provider("rss") == "overlap_skipped"
        release.set()
        assert await first == "success"
    asyncio.run(run())


@pytest.mark.parametrize("provider", ["rss", "newsapi", "gdelt", "hackernews", "market"])
def test_each_provider_dispatch(monkeypatch, provider):
    calls = []
    async def ingest(config, request, client):
        calls.append(request.providers)
        return news_result()
    async def market(*args):
        calls.append("market")
        return MarketResponse(collected_at=datetime.now(timezone.utc), candles=[],
                              sources=[MarketSourceResult(ticker="AAPL", provider_symbol="AAPL", status="success")],
                              diagnostics=MarketDiagnostics(collected=1))
    monkeypatch.setattr("app.core.scheduler.ingest", ingest)
    monkeypatch.setattr("app.core.scheduler.ingest_market", market)
    async def run():
        scheduler = IngestionScheduler(settings())
        assert await scheduler.run_provider(provider) == "success"
        assert calls == (["market"] if provider == "market" else [[provider]])
    asyncio.run(run())


def test_partial_failure_and_rate_limit_backoff(monkeypatch):
    async def ingest(*args):
        return news_result(["rate_limited"])
    monkeypatch.setattr("app.core.scheduler.ingest", ingest)
    async def run():
        scheduler = IngestionScheduler(settings())
        assert await scheduler.run_provider("rss") == "partial"
        assert scheduler.cooldown["rss"] == 600
        await scheduler.run_provider("rss")
        assert scheduler.cooldown["rss"] == 1200
        assert scheduler.last_status["rss"]["error_category"] == "rate_limited"
    asyncio.run(run())


def test_exception_is_sanitized_and_future_jobs_continue(monkeypatch, caplog):
    calls = []
    async def ingest(*args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("secret-token internal URL")
        return news_result()
    monkeypatch.setattr("app.core.scheduler.ingest", ingest)
    async def run():
        scheduler = IngestionScheduler(settings())
        assert await scheduler.run_provider("rss") == "error"
        assert await scheduler.run_provider("rss") == "success"
    with caplog.at_level(logging.INFO, logger="sentanalysis.scheduler"):
        asyncio.run(run())
    assert "secret-token" not in caplog.text and "internal URL" not in caplog.text
    records = [json.loads(record.message) for record in caplog.records if record.name == "sentanalysis.scheduler"]
    ended = [record for record in records if record["event"] == "job_finished"]
    assert ended[0]["started_at"] <= ended[0]["ended_at"]
    assert ended[0]["status"] == "error" and ended[0]["error_category"] == "job_failure"
    assert ended[1]["collected"] == 2


def test_shutdown_cancels_hung_job(monkeypatch, tmp_path):
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def ingest(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        monkeypatch.setattr("app.core.scheduler.ingest", ingest)
        scheduler = IngestionScheduler(settings(shutdown_timeout_seconds=.02), lease=SchedulerLease(tmp_path / "cancel.lock"))
        scheduler.interval = lambda provider: .01
        await scheduler.start()
        await asyncio.wait_for(entered.wait(), 2)
        await scheduler.stop()
        assert cancelled.is_set()
        assert scheduler.lease.handle is None
        assert scheduler.last_status["rss"]["status"] == "cancelled"
    asyncio.run(run())


def test_lifespan_no_scheduler_registration_when_disabled():
    app = create_app(Settings())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state.scheduler.task is None


def test_health_remains_available_during_job(monkeypatch):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def ingest(*args):
            entered.set()
            await release.wait()
            return news_result()
        monkeypatch.setattr("app.core.scheduler.ingest", ingest)
        application = create_app(Settings())
        scheduler = IngestionScheduler(settings())
        job = asyncio.create_task(scheduler.run_provider("rss"))
        await entered.wait()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(application), base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
        release.set()
        await job
    asyncio.run(run())


def test_optional_analysis_and_market_never_sent(monkeypatch):
    from test_insight_client import document
    calls = []
    async def ingest(*args):
        result = news_result()
        result.documents = [document()]
        return result
    async def market(*args):
        return MarketResponse(collected_at=datetime.now(timezone.utc), candles=[], sources=[], diagnostics=MarketDiagnostics())
    async def analyze(self, documents, model, chunk_size):
        calls.append((documents[0].document_id, model, chunk_size))
        from app.models.insight import AnalysisReport
        return AnalysisReport(status="error", model=model, outcomes=[], success_count=0, error_count=1)
    monkeypatch.setattr("app.core.scheduler.ingest", ingest)
    monkeypatch.setattr("app.core.scheduler.ingest_market", market)
    monkeypatch.setattr("app.core.scheduler.InsightClient.analyze", analyze)
    async def run():
        scheduler = IngestionScheduler(settings(analyze_documents=True))
        assert await scheduler.run_provider("rss") == "partial"
        await scheduler.run_provider("market")
        assert calls == [("doc-0", "vader", 4)]
    asyncio.run(run())


@pytest.mark.parametrize("values", [{"rss_interval_seconds": 0}, {"market_interval_seconds": 59},
                                   {"providers": ["unknown"]}, {"shutdown_timeout_seconds": 0}, {"newsapi_interval_seconds": float("inf")}])
def test_invalid_scheduler_config(values):
    with pytest.raises(ValidationError):
        SchedulerConfig(**values)


def test_env_scheduler(monkeypatch):
    monkeypatch.setenv("INGESTION_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("INGESTION_SCHEDULER_PROVIDERS", '["rss","market"]')
    monkeypatch.setenv("INGESTION_SCHEDULER_MARKET_INTERVAL_SECONDS", "600")
    config = Settings.from_env().scheduler
    assert config.enabled and config.providers == ["rss", "market"]
    assert config.market_interval_seconds == 600

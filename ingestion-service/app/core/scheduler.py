"""One asynchronous dispatcher per OS lease, with no catch-up or overlapping jobs."""
import asyncio
from datetime import datetime, timezone
import json
import logging
import time

import httpx
from app.helper.metrics import record as record_metric

from app.core.pipeline import ingest
from app.core.extractors.market import ingest_market
from app.core.insight_client import InsightClient
from app.helper.scheduler_lock import SchedulerLease
from app.models.ingestion import IngestRequest
from app.models.market import MarketRequest

logger = logging.getLogger("sentanalysis.scheduler")


class IngestionScheduler:
    def __init__(self, settings, http_client=None, *, lease=None):
        self.settings = settings
        self.config = settings.scheduler
        self.client = http_client
        self.owns_client = http_client is None
        self.lease = lease or SchedulerLease()
        self.task = None
        self.stop_event = asyncio.Event()
        self.execution_lock = asyncio.Lock()
        self.next_due = {}
        self.cooldown = {}
        self.last_status = {}

    def interval(self, provider):
        return getattr(self.config, provider + "_interval_seconds")

    def active_providers(self):
        return [provider for provider in self.config.providers if
                (bool(self.settings.sources) if provider == "rss" else getattr(self.settings, provider).enabled)]

    async def start(self):
        if not self.config.enabled or self.task is not None:
            return False
        if not self.lease.acquire():
            logger.info(json.dumps({"event": "scheduler_standby", "reason": "another_worker_owns_lease"}))
            return False
        try:
            if self.client is None:
                self.client = httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_connections=2))
            self.stop_event.clear()
            self.next_due = {provider: time.monotonic() + self.interval(provider) for provider in self.active_providers()}
            self.task = asyncio.create_task(self._loop(), name="sentanalysis-ingestion-scheduler")
        except BaseException:
            self.lease.release()
            raise
        return True

    async def _loop(self):
        while not self.stop_event.is_set():
            now = time.monotonic()
            for provider in self.next_due:
                if self.stop_event.is_set():
                    break
                if self.next_due[provider] <= now:
                    await self.run_provider(provider)
                    # Completion-based scheduling avoids bursts after slow jobs.
                    self.next_due[provider] = time.monotonic() + self.cooldown.get(provider, self.interval(provider))
            delay = max(0.01, min(self.next_due.values(), default=time.monotonic() + 60) - time.monotonic())
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def run_provider(self, provider):
        if self.execution_lock.locked():
            record_metric("scheduler_jobs_total", "overlap_skipped")
            return "overlap_skipped"
        async with self.execution_lock:
            started = datetime.now(timezone.utc).isoformat()
            logger.info(json.dumps({"event": "job_started", "provider": provider, "started_at": started}))
            status, error, collected = "success", None, 0
            try:
                if provider == "market":
                    result = await ingest_market(self.settings, MarketRequest(), self.client)
                else:
                    result = await ingest(self.settings, IngestRequest(providers=[provider]), self.client)
                collected = result.diagnostics.collected
                failures = [source for source in result.sources if source.status == "error"]
                if failures:
                    status = "error" if len(failures) == len(result.sources) else "partial"
                    error = "rate_limited" if any(source.error_code == "rate_limited" for source in failures) else "source_failure"
                if error == "rate_limited":
                    cooldown = min(86400, max(self.interval(provider) * 2, self.cooldown.get(provider, 0) * 2))
                    self.cooldown[provider] = max(cooldown, max((source.retry_after_seconds or 0 for source in failures), default=0))
                else:
                    self.cooldown[provider] = self.interval(provider)
                if provider != "market" and self.config.analyze_documents and result.documents:
                    report = await InsightClient(self.settings.insight, self.client).analyze(
                        result.documents, self.config.analysis_model, self.config.analysis_chunk_size)
                    if report.error_count:
                        status = "partial" if status != "error" else status
                        error = error or "insight_failure"
            except asyncio.CancelledError:
                status, error = "cancelled", "shutdown"
                raise
            except Exception:
                # No traceback/exception text: providers may embed URLs/secrets.
                status, error = "error", "job_failure"
            finally:
                record_metric("scheduler_jobs_total", status)
                record = {"event": "job_finished", "provider": provider, "started_at": started,
                          "ended_at": datetime.now(timezone.utc).isoformat(), "collected": collected,
                          "status": status, "error_category": error}
                self.last_status[provider] = record
                logger.info(json.dumps(record))
            return status

    async def stop(self):
        self.stop_event.set()
        task = self.task
        try:
            if task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=self.config.shutdown_timeout_seconds)
                except (TimeoutError, asyncio.CancelledError):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            self.task = None
            try:
                if self.owns_client and self.client is not None:
                    await self.client.aclose()
                    self.client = None
            finally:
                self.lease.release()

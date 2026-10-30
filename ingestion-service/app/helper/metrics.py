"""Small process-local Prometheus text exposition; no external dependencies.

One registry per app/process. Scrape each worker separately. Only the actual
loopback peer may scrape; forwarded headers are deliberately ignored. Keep the
service bound to loopback or restrict ingress at the firewall: an untrusted
local reverse proxy must block /metrics. Counters reset on restart.
"""
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from ipaddress import ip_address
from threading import Lock
from time import perf_counter

from fastapi import Request
from starlette.responses import PlainTextResponse

_current = ContextVar("metrics", default=None)
# All names and label values are bounded here, never derived from content.
_ALLOWED = {
    "http_requests_total": {"health", "ingest", "analyze", "aggregate", "signals", "other"},
    "http_errors_total": {"4xx", "5xx"},
    "http_request_duration_seconds": {"health", "ingest", "analyze", "aggregate", "signals", "other"},
    "documents_collected_total": {"rss", "newsapi", "gdelt", "hackernews", "market"},
    "documents_accepted_total": {"news", "market"},
    "provider_failures_total": {"rss", "newsapi", "gdelt", "hackernews", "market"},
    "provider_timeouts_total": {"rss", "newsapi", "gdelt", "hackernews", "market"},
    "insight_transfers_total": {"success", "failure"},
    "scheduler_jobs_total": {"success", "partial", "error", "cancelled", "overlap_skipped"},
    "documents_analyzed_total": {"vader", "finbert"},
    "inference_duration_seconds": {"vader", "finbert"},
    "batch_failures_total": {"failed"},
    "aggregation_requests_total": {"aggregate"},
    "signal_requests_total": {"signals"},
}
_DURATIONS = {"http_request_duration_seconds", "inference_duration_seconds"}


class Metrics:
    def __init__(self):
        self.values = defaultdict(float)
        self.lock = Lock()

    def add(self, name, category, value=1):
        if category not in _ALLOWED.get(name, ()) or value < 0:
            return
        with self.lock:
            self.values[(name, category)] += value
            if name in _DURATIONS:
                self.values[(name + "_count", category)] += 1

    def render(self):
        with self.lock:
            values = dict(self.values)
        lines = []
        for name, categories in sorted(_ALLOWED.items()):
            kind = "summary" if name in _DURATIONS else "counter"
            lines.append(f"# TYPE sentanalysis_{name} {kind}")
            for category in sorted(categories):
                label = '{category="' + category + '"}'
                if name in _DURATIONS:
                    lines.append(f"sentanalysis_{name}_sum{label} {values.get((name, category), 0)}")
                    lines.append(f"sentanalysis_{name}_count{label} {values.get((name + '_count', category), 0)}")
                else:
                    lines.append(f"sentanalysis_{name}{label} {values.get((name, category), 0)}")
        return "\n".join(lines) + "\n"


def record(name, category, value=1):
    registry = _current.get()
    if registry is not None:
        registry.add(name, category, value)


def measure_model(model):
    """Record actual scorer invocations, including retries; successful outputs only."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                result = function(*args, **kwargs)
                record("documents_analyzed_total", model, len(result.results))
                return result
            finally:
                record("inference_duration_seconds", model, perf_counter() - started)
        return wrapped
    return decorate


class MetricsMiddleware:
    def __init__(self, app, registry):
        self.app, self.registry = app, registry

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/metrics":
            return await self.app(scope, receive, send)
        token = _current.set(self.registry)
        started, status = perf_counter(), 500
        async def observed(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
        try:
            await self.app(scope, receive, observed)
        finally:
            route = getattr(scope.get("route"), "path", "")
            category = ("health" if route == "/health" else
                        "ingest" if route.startswith("/ingest/") else
                        "analyze" if route in {"/analyze", "/analyze/batch", "/analyze/validate"} else
                        "aggregate" if route in {"/aggregate", "/aggregate/weighted"} else
                        "signals" if route.startswith("/signals/") else "other")
            record("http_requests_total", category)
            record("http_request_duration_seconds", category, perf_counter() - started)
            if status >= 400:
                record("http_errors_total", "5xx" if status >= 500 else "4xx")
            if category == "aggregate":
                record("aggregation_requests_total", category)
            elif category == "signals":
                record("signal_requests_total", category)
            _current.reset(token)


def install_metrics(application):
    if hasattr(application.state, "metrics"):
        return
    registry = Metrics()
    application.state.metrics = registry
    application.add_middleware(MetricsMiddleware, registry=registry)

    @application.get("/metrics", include_in_schema=False)
    async def metrics(request: Request):
        try:
            trusted = request.client is not None and ip_address(request.client.host).is_loopback
        except ValueError:
            trusted = False
        if not trusted:
            return PlainTextResponse("Forbidden", status_code=403)
        return PlainTextResponse(registry.render(), media_type="text/plain; version=0.0.4")

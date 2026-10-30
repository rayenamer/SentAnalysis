import sys
import pytest
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from app.main import create_app
from app.helper.metrics import Metrics, _current, measure_model, record


def test_metrics_local_only_and_no_forwarded_header_trust():
    application = create_app()
    with TestClient(application, client=("127.0.0.1", 12345)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/missing-secret-document").status_code == 404
        result = client.get("/metrics")
        assert result.status_code == 200
        assert 'sentanalysis_http_requests_total{category="health"} 1.0' in result.text
        assert 'sentanalysis_http_errors_total{category="4xx"} 1.0' in result.text
        assert 'sentanalysis_http_request_duration_seconds_count{category="health"} 1.0' in result.text
        assert "missing-secret-document" not in result.text
        assert result.headers["content-type"].startswith("text/plain")
        assert client.get("/metrics").text == result.text  # scrapes excluded
    with TestClient(application, client=("203.0.113.5", 12345)) as client:
        assert client.get("/metrics", headers={"x-forwarded-for": "127.0.0.1"}).status_code == 403


def test_per_application_registry_and_ipv6_loopback():
    first, second = create_app(), create_app()
    with TestClient(first, client=("::1", 1)) as client:
        client.get("/health")
        assert client.get("/metrics").status_code == 200
    assert first.state.metrics is not second.state.metrics
    assert 'sentanalysis_http_requests_total{category="health"} 0' in second.state.metrics.render()


def test_bounded_labels_thread_safety_and_model_measurement():
    registry = Metrics()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: registry.add("provider_failures_total", "rss"), range(100)))
    registry.add("provider_failures_total", "secret-url")
    assert "secret-url" not in registry.render()
    assert 'sentanalysis_provider_failures_total{category="rss"} 100.0' in registry.render()
    token = _current.set(registry)
    try:
        from types import SimpleNamespace
        @measure_model("vader")
        def score():
            return SimpleNamespace(results=[1, 2])
        score()
        record("batch_failures_total", "failed")
    finally:
        _current.reset(token)
    assert 'sentanalysis_documents_analyzed_total{category="vader"} 2.0' in registry.render()
    assert 'sentanalysis_inference_duration_seconds_count{category="vader"} 1.0' in registry.render()


def test_scrape_does_not_load_model_runtime():
    before = {name for name in sys.modules if name.startswith(("onnxruntime", "transformers", "torch"))}
    with TestClient(create_app(), client=("127.0.0.1", 1)) as client:
        client.get("/metrics")
    after = {name for name in sys.modules if name.startswith(("onnxruntime", "transformers", "torch"))}
    assert after == before


def test_real_vader_api_counts_and_failed_batch(monkeypatch):
    import app.core.batch as batch
    application = create_app()
    body = {"model": "vader", "documents": [
        {"document_id": "secret-id", "title": "Results", "text": "Excellent profits!"}]}
    with TestClient(application, client=("127.0.0.1", 1)) as client:
        assert client.post("/analyze", json=body).status_code == 200
        text = client.get("/metrics").text
        assert 'sentanalysis_documents_analyzed_total{category="vader"} 1.0' in text
        assert 'sentanalysis_inference_duration_seconds_count{category="vader"} 1.0' in text
        def fail(_):
            raise ValueError("secret")
        monkeypatch.setattr(batch, "_run_model", fail)
        assert client.post("/analyze/batch", json=body).status_code == 200
        text = client.get("/metrics").text
        assert 'sentanalysis_batch_failures_total{category="failed"} 1.0' in text
        assert "secret" not in text


@pytest.mark.parametrize("module", [
    "base", "sentiment", "vader", "aggregation", "extensions",
    "velocity", "outliers", "correlation", "granger",
])
def test_legacy_apps_have_their_own_metrics(module):
    from importlib import import_module
    application = import_module("app.api." + module).app
    assert sum(getattr(route, "path", "") == "/metrics" for route in application.routes) == 1
    before = application.state.metrics.values.get(("http_requests_total", "other"), 0)
    with TestClient(application, client=("127.0.0.1", 1)) as client:
        assert client.get("/unknown").status_code == 404
        assert f'sentanalysis_http_requests_total{{category="other"}} {float(before + 1)}' in client.get("/metrics").text

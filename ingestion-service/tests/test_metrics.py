import sys
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


def test_ingestion_provider_counts(client_factory, rss):
    import httpx
    client = client_factory(lambda _: httpx.Response(200, content=rss))
    response = client.post("/ingest/run", json={})
    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    text = client.app.state.metrics.render()
    assert f'sentanalysis_documents_collected_total{{category="rss"}} {float(diagnostics["collected"])}' in text
    assert f'sentanalysis_documents_accepted_total{{category="news"}} {float(diagnostics["returned"])}' in text


def test_provider_timeout_count(client_factory):
    import httpx
    def timeout(_):
        raise httpx.ReadTimeout("secret")
    client = client_factory(timeout)
    assert client.post("/ingest/run", json={}).status_code == 503
    text = client.app.state.metrics.render()
    assert 'sentanalysis_provider_failures_total{category="rss"} 1.0' in text
    assert 'sentanalysis_provider_timeouts_total{category="rss"} 1.0' in text
    assert "secret" not in text

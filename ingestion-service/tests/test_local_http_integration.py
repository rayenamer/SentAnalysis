"""Real Uvicorn apps over loopback; only the external RSS provider is mocked."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest


def launch(code, cwd, log, *arguments):
    output = log.open("w", encoding="utf-8")
    environment = {**os.environ, "RUN_FINBERT_INTEGRATION": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    interpreter = cwd / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    process = subprocess.Popen([str(interpreter) if interpreter.exists() else sys.executable, "-B", "-c", code, *arguments], cwd=cwd,
                               stdout=output, stderr=output, env=environment,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            text = log.read_text(encoding="utf-8")
            match = re.search(r"Uvicorn running on http://127\.0\.0\.1:(\d+)", text)
            if match:
                return process, output, "http://127.0.0.1:" + match.group(1)
            if process.poll() is not None:
                break
            time.sleep(.05)
        raise AssertionError(log.read_text(encoding="utf-8"))
    except BaseException:
        terminate(process, output)
        raise


def terminate(process, output):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    output.close()


@pytest.mark.parametrize("model", ["vader", "loughran_mcdonald"])
def test_real_ingestion_to_insight_over_local_http(tmp_path, model):
    root = Path(__file__).resolve().parents[2]
    insight_code = """
import uvicorn
from app.main import create_app
uvicorn.run(create_app(), host='127.0.0.1', port=0, log_level='info')
"""
    ingestion_code = """
import sys
from pathlib import Path
import httpx,uvicorn
from app.main import create_app
from app.config import Settings
class Transport(httpx.AsyncBaseTransport):
    def __init__(self): self.real=httpx.AsyncHTTPTransport()
    async def handle_async_request(self, request):
        if request.url.host == 'fixture.example':
            return httpx.Response(200, content=Path('tests/fixtures/news.xml').read_bytes())
        return await self.real.handle_async_request(request)
    async def aclose(self): await self.real.aclose()
settings=Settings(sources=[{'name':'fixture','url':'https://fixture.example/news'}],
                  insight={'service_url':sys.argv[1],'max_retries':0,'request_timeout_seconds':2})
client=httpx.AsyncClient(transport=Transport(),trust_env=False)
uvicorn.run(create_app(settings,client),host='127.0.0.1',port=0,log_level='info')
"""
    servers = []
    try:
        insight, insight_output, insight_url = launch(insight_code, root / "insight-service", tmp_path / "insight.log")
        servers.append((insight, insight_output))
        ingestion, ingestion_output, ingestion_url = launch(ingestion_code, root / "ingestion-service", tmp_path / "ingestion.log", insight_url)
        servers.append((ingestion, ingestion_output))
        with httpx.Client(timeout=15, trust_env=False) as client:
            assert client.get(insight_url + "/health").status_code == 200
            assert client.post(insight_url + "/analyze/batch", json={"documents": []}).status_code == 422
            boundary = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            history = [{"ticker": "AAPL", "model": model, "sentiment_score": score,
                        "observed_at": (boundary - timedelta(hours=7-index)).isoformat(),
                        "aggregation_window": "7d"} for index, score in enumerate([-.8, -.4, .1, .6, -.1, .3])]
            prices = [{"ticker": "AAPL", "close_price": price,
                       "timestamp": (boundary - timedelta(hours=8-index)).isoformat()}
                      for index, price in enumerate([100., 101., 100.5, 102., 100., 102., 103.])]
            response = client.post(ingestion_url + "/ingest/analyze", json={"model": model,
                "analytics": {"signal_window": "7d", "history": history, "prices": prices}})
            assert response.status_code == 200, response.text
            body = response.json()
            doc = body["ingestion"]["documents"][0]
            outcome = body["analysis"]["outcomes"][0]
            assert body["analysis"]["status"] == "success"
            assert body["analysis"]["model"] == outcome["result"]["model"] == model
            assert body["analysis"]["success_count"] == 1
            assert body["analysis"]["error_count"] == 0
            if model == "loughran_mcdonald":
                assert outcome["result"]["confidence"] is None
            assert doc["document_id"] == outcome["document_id"] == outcome["result"]["document_id"]
            assert doc["tickers"] == outcome["tickers"] == ["AAPL"]
            assert doc["source"] == "fixture" and doc["published_at"] == "2026-10-27T10:00:00Z"
            assert "strong earnings" in doc["text"]
            assert body["aggregation"]["status"] == "success"
            assert body["weighted_aggregation"]["status"] == "success"
            windows = body["aggregation"]["data"]["windows"]
            assert [row["window"] for row in windows] == ["1h", "24h", "7d"]
            aggregate = windows[-1]["aggregates"][0]
            assert aggregate["ticker"] == "AAPL" and aggregate["model"] == model
            assert aggregate["article_count"] == 1
            assert body["signals"]["velocity"]["status"] == "success"
            assert body["signals"]["outliers"]["status"] == "success"
            assert body["signals"]["correlation"]["status"] == "success"
            assert body["signals"]["granger"]["status"] == "unavailable"
            assert body["signals"]["granger"]["data"]["results"]
            assert body["signals"]["correlation"]["data"]["results"][0]["paired_observations"] >= 3
            # Real service unavailability must not remove successfully ingested data.
            terminate(insight, insight_output)
            response = client.post(ingestion_url + "/ingest/analyze", json={})
            assert response.status_code == 200
            assert response.json()["analysis"]["status"] == "error"
            assert response.json()["aggregation"]["reason"] == "no_successful_analysis"
            assert len(response.json()["ingestion"]["documents"]) == 1
            assert client.post(ingestion_url + "/ingest/run", json={}).status_code == 200
    finally:
        for process, output in reversed(servers):
            if not output.closed:
                terminate(process, output)

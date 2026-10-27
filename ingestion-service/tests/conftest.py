from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def rss():
    return (Path(__file__).parent / "fixtures" / "news.xml").read_bytes()


@pytest.fixture
def client_factory():
    clients = []

    def build(handler, **overrides):
        settings = Settings(sources=[{"name": "news", "url": "https://feeds.example/news"}], **overrides)
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = TestClient(create_app(settings, http))
        clients.append((client, http))
        return client

    yield build
    import asyncio
    for client, http in clients:
        client.close()
        asyncio.run(http.aclose())

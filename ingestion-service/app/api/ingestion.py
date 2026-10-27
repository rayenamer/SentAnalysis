from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.config import Settings
from app.core.pipeline import ingest
from app.models.ingestion import IngestRequest, IngestResponse

router = APIRouter()


@asynccontextmanager
async def client_context(request):
    if request.app.state.http_client is not None:
        yield request.app.state.http_client
    else:
        async with httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_connections=2)) as client:
            yield client


@router.post("/ingest/run", response_model=IngestResponse,
             responses={422: {"description": "Invalid request or source selection"},
                        503: {"description": "Invalid configuration or all sources unavailable"}})
async def run(body: IngestRequest, request: Request):
    try:
        settings = request.app.state.settings or Settings.from_env()
    except (ValidationError, ValueError):
        raise HTTPException(503, detail={"code": "invalid_configuration"})
    try:
        async with client_context(request) as client:
            result = await ingest(settings, body, client)
    except ValueError as exc:
        code = str(exc)
        raise HTTPException(422 if code == "unknown_source" else 503, detail={"code": code})
    if all(source.status == "error" for source in result.sources):
        raise HTTPException(503, detail={"code": "all_sources_failed", "sources": [s.model_dump() for s in result.sources]})
    return result

from fastapi import APIRouter, HTTPException, Request

from app.api.ingestion import client_context
from app.config import Settings
from app.core.extractors.market import ingest_market
from app.models.market import MarketRequest, MarketResponse

router = APIRouter()


@router.post("/ingest/market", response_model=MarketResponse,
             responses={503: {"description": "Provider unavailable or not configured"}})
async def run_market(body: MarketRequest, request: Request):
    try:
        settings = request.app.state.settings or Settings.from_env()
    except ValueError:
        raise HTTPException(503, detail={"code": "invalid_configuration"})
    try:
        async with client_context(request) as client:
            result = await ingest_market(settings, body, client)
    except ValueError as exc:
        code = str(exc)
        raise HTTPException(422 if code == "unknown_symbol" else 503, detail={"code": code})
    if all(source.status == "error" for source in result.sources):
        raise HTTPException(503, detail={"code": "all_sources_failed", "sources": [s.model_dump() for s in result.sources]})
    return result

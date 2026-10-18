"""Additive entry point: python -m uvicorn app.extensions_api:app --workers 1.

Original /analyze and /aggregate semantics remain unchanged. Use /analyze/batch
for one outcome per input and /aggregate/weighted for explicit weighting.
Integrating these into an old entry point would require app.include_router(router)
and including aggregation_api.router there; no existing file is edited here.
"""

from fastapi import APIRouter, FastAPI

from app.aggregation_api import router as aggregation_router
from app.finbert_api import app as analysis_app
from app.batch_analysis import BatchAnalyzeRequest, BatchAnalyzeResponse, analyze_batch
from app.weighted_aggregation import (WeightedAggregationRequest,
                                     WeightedAggregationResponse, aggregate_weighted)

router = APIRouter()


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
def batch(request: BatchAnalyzeRequest):
    """200 contains explicit success/error outcomes; invalid input returns 422."""
    return analyze_batch(request)


@router.post("/aggregate/weighted", response_model=WeightedAggregationResponse)
def weighted(request: WeightedAggregationRequest):
    return aggregate_weighted(request)


app = FastAPI(title=analysis_app.title, version=analysis_app.version)
app.include_router(analysis_app.router)
app.include_router(aggregation_router)
app.include_router(router)

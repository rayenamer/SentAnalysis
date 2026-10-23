"""Independent service: python -m uvicorn app.aggregation_api:app.

POST /aggregate accepts existing sentiment result fields plus published_at.
For a combined API, an existing entry point would need to include `router`:
    from app.aggregation_api import router
    app.include_router(router)
That integration change is intentionally not applied to existing files.
This module imports no VADER, FinBERT, model weights, or database clients.
"""

from fastapi import APIRouter, FastAPI
from app.validation_errors import install_validation_error_handler

from app.aggregation import aggregate_sentiments
from app.aggregation_models import AggregationRequest, AggregationResponse

router = APIRouter()


@router.post("/aggregate", response_model=AggregationResponse)
def aggregate(request: AggregationRequest) -> AggregationResponse:
    return aggregate_sentiments(request)


app = FastAPI(title="SentAnalysis - Sentiment Aggregation", version="0.1.0")
install_validation_error_handler(app)
app.include_router(router)

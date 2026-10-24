"""HTTP wiring for aggregation; calculations live in app.core."""

from fastapi import APIRouter, FastAPI
from app.helper.validation_errors import install_validation_error_handler

from app.core.aggregate import aggregate_sentiments
from app.models.aggregation import AggregationRequest, AggregationResponse

from app.models.weighted_aggregation import WeightedAggregationRequest, WeightedAggregationResponse
from app.core.weighted_aggregation import aggregate_weighted

router = APIRouter()


@router.post("/aggregate", response_model=AggregationResponse)
def aggregate(request: AggregationRequest) -> AggregationResponse:
    return aggregate_sentiments(request)


app = FastAPI(title="SentAnalysis - Sentiment Aggregation", version="0.1.0")
install_validation_error_handler(app)
app.include_router(router)


weighted_router = APIRouter()


@weighted_router.post("/aggregate/weighted", response_model=WeightedAggregationResponse)
def weighted(request: WeightedAggregationRequest):
    return aggregate_weighted(request)



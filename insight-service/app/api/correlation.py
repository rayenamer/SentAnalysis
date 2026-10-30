"""HTTP wiring for correlation; calculations live in app.core."""

from fastapi import APIRouter, FastAPI
from app.helper.validation_errors import install_validation_error_handler

from app.core.signals.correlation import calculate_correlation
from app.models.correlation import CorrelationRequest, CorrelationResponse
from app.api.outliers import app as existing_app

router = APIRouter()


@router.post("/signals/correlation", response_model=CorrelationResponse)
def correlation(request: CorrelationRequest) -> CorrelationResponse:
    return calculate_correlation(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

from app.helper.metrics import install_metrics
install_metrics(app)

"""python -m uvicorn app.correlation_api:app --workers 1

Includes all existing outliers_api routes unchanged. The standalone router is
also reusable via app.include_router(router) in a future central entry point.
"""

from fastapi import APIRouter, FastAPI
from app.validation_errors import install_validation_error_handler

from app.correlation import calculate_correlation
from app.correlation_models import CorrelationRequest, CorrelationResponse
from app.outliers_api import app as existing_app

router = APIRouter()


@router.post("/signals/correlation", response_model=CorrelationResponse)
def correlation(request: CorrelationRequest) -> CorrelationResponse:
    return calculate_correlation(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

"""python -m uvicorn app.granger_api:app --workers 1

Additive entry point includes all correlation_api routes unchanged. Missing
statistics dependencies yield explicit null inference/diagnostics, not fake F/p.
"""

from fastapi import APIRouter, FastAPI
from app.validation_errors import install_validation_error_handler

from app.correlation_api import app as existing_app
from app.granger import calculate_granger
from app.granger_models import GrangerRequest, GrangerResponse

router = APIRouter()


@router.post("/signals/granger", response_model=GrangerResponse)
def granger(request: GrangerRequest) -> GrangerResponse:
    return calculate_granger(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

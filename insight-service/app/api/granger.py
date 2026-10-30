"""HTTP wiring for granger; calculations live in app.core."""

from fastapi import APIRouter, FastAPI
from app.helper.validation_errors import install_validation_error_handler

from app.api.correlation import app as existing_app
from app.core.signals.granger import calculate_granger
from app.models.granger import GrangerRequest, GrangerResponse

router = APIRouter()


@router.post("/signals/granger", response_model=GrangerResponse)
def granger(request: GrangerRequest) -> GrangerResponse:
    return calculate_granger(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

from app.helper.metrics import install_metrics
install_metrics(app)

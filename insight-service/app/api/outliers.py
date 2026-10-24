"""HTTP wiring for outliers; calculations live in app.core."""

from fastapi import APIRouter, FastAPI
from app.helper.validation_errors import install_validation_error_handler

from app.core.signals.outliers import calculate_outliers
from app.models.outliers import OutliersRequest, OutliersResponse
from app.api.velocity import app as existing_app

router = APIRouter()


@router.post("/signals/outliers", response_model=OutliersResponse)
def outliers(request: OutliersRequest) -> OutliersResponse:
    return calculate_outliers(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

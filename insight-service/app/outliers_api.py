"""python -m uvicorn app.outliers_api:app --workers 1

Additive entry point preserving all velocity_api routes. No existing API file
is changed. The exported router may also be included by another application.
"""

from fastapi import APIRouter, FastAPI
from app.validation_errors import install_validation_error_handler

from app.outliers import calculate_outliers
from app.outliers_models import OutliersRequest, OutliersResponse
from app.velocity_api import app as existing_app

router = APIRouter()


@router.post("/signals/outliers", response_model=OutliersResponse)
def outliers(request: OutliersRequest) -> OutliersResponse:
    return calculate_outliers(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

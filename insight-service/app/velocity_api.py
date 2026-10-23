"""Additive entry point: python -m uvicorn app.velocity_api:app --workers 1.

Preserves existing endpoints by including extensions_api's router unchanged.
Alternatively, include this module's `router` in a chosen existing application;
that would require editing its entry point and is deliberately not applied.
"""

from fastapi import APIRouter, FastAPI
from app.validation_errors import install_validation_error_handler

from app.extensions_api import app as existing_app
from app.velocity import calculate_velocity
from app.velocity_models import VelocityRequest, VelocityResponse

router = APIRouter()


@router.post("/signals/velocity", response_model=VelocityResponse)
def velocity(request: VelocityRequest) -> VelocityResponse:
    return calculate_velocity(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

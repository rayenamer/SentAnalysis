"""HTTP wiring for velocity; calculations live in app.core."""

from fastapi import APIRouter, FastAPI
from app.helper.validation_errors import install_validation_error_handler

from app.api.extensions import app as existing_app
from app.core.signals.velocity import calculate_velocity
from app.models.velocity import VelocityRequest, VelocityResponse

router = APIRouter()


@router.post("/signals/velocity", response_model=VelocityResponse)
def velocity(request: VelocityRequest) -> VelocityResponse:
    return calculate_velocity(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)
app.include_router(router)

from app.helper.metrics import install_metrics
install_metrics(app)

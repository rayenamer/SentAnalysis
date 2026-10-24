"""Compatibility composition for analysis, batch and aggregation."""

from fastapi import APIRouter, FastAPI
from app.helper.validation_errors import install_validation_error_handler
from app.api.aggregation import router as aggregation_router, weighted_router, weighted
from app.api.batch import router as batch_router, batch
from app.api.sentiment import app as analysis_app

router = APIRouter()
router.include_router(batch_router)
router.include_router(weighted_router)

app = FastAPI(title=analysis_app.title, version=analysis_app.version)
install_validation_error_handler(app)
app.include_router(analysis_app.router)
app.include_router(aggregation_router)
app.include_router(router)

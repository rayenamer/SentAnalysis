"""One router collecting the four existing financial signal endpoints."""

from fastapi import APIRouter
from app.api.velocity import router as velocity_router
from app.api.outliers import router as outliers_router
from app.api.correlation import router as correlation_router
from app.api.granger import router as granger_router

router = APIRouter()
for signal_router in (velocity_router, outliers_router, correlation_router, granger_router):
    router.include_router(signal_router)

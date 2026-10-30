from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.config import Settings
from app.core.scheduler import IngestionScheduler
from app.helper.metrics import install_metrics, _current

from app.api.ingestion import router
from app.api.market import router as market_router


def create_app(settings=None, http_client=None):
    @asynccontextmanager
    async def lifespan(application):
        try:
            current_settings = settings or Settings.from_env()
        except ValueError:
            # Preserve existing lazy configuration errors and health availability.
            current_settings = None
            logging.getLogger("sentanalysis.scheduler").error("scheduler_configuration_invalid")
        scheduler = IngestionScheduler(current_settings, http_client) if current_settings else None
        application.state.scheduler = scheduler
        metrics_token = _current.set(application.state.metrics)
        try:
            if scheduler:
                await scheduler.start()
            yield
        finally:
            if scheduler:
                await scheduler.stop()
            _current.reset(metrics_token)

    application = FastAPI(title="SentAnalysis - Ingestion Service", version="0.1.0", lifespan=lifespan)
    application.state.settings = settings
    application.state.http_client = http_client
    install_metrics(application)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Exclude untrusted inputs and exception contexts, including NaN/Infinity.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]})

    @application.get("/health")
    def health():
        return {"status": "ok", "service": "ingestion-service"}

    application.include_router(router)
    application.include_router(market_router)
    return application


app = create_app()

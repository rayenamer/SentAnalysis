from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.ingestion import router
from app.api.market import router as market_router


def create_app(settings=None, http_client=None):
    application = FastAPI(title="SentAnalysis - Ingestion Service", version="0.1.0")
    application.state.settings = settings
    application.state.http_client = http_client

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

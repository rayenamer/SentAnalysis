"""Additive entry point: python -m uvicorn app.finbert_api:app --workers 1."""

import logging

from fastapi import FastAPI, HTTPException
from app.validation_errors import install_validation_error_handler

from app.main import app as existing_app
from app.models.sentiment import AnalyzeRequest, AnalyzeResponse
from app.finbert import get_finbert_service
from app.finbert_models import FinBertResponse
from app.vader import analyze_documents

logger = logging.getLogger(__name__)
app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)


@app.post("/analyze", response_model=FinBertResponse | AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if request.model == "vader":
        return analyze_documents(request)
    try:
        return get_finbert_service().analyze(request)
    except Exception:
        logger.exception("FinBERT analysis failed")
        raise HTTPException(status_code=503, detail="FinBERT CPU inference is unavailable") from None

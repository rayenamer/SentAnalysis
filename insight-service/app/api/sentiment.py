"""HTTP wiring for sentiment; calculations live in app.core."""

import logging

from fastapi import APIRouter, FastAPI, HTTPException
from app.helper.validation_errors import install_validation_error_handler

from app.api.base import app as existing_app
from app.models.sentiment import AnalyzeRequest, AnalyzeResponse
from app.core.sentiment.finbert import get_finbert_service
from app.models.finbert import FinBertResponse
from app.core.sentiment.lexicon import analyze_documents

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=FinBertResponse | AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if request.model == "vader":
        return analyze_documents(request)
    try:
        return get_finbert_service().analyze(request)
    except Exception:
        logger.exception("FinBERT analysis failed")
        raise HTTPException(status_code=503, detail="FinBERT CPU inference is unavailable") from None


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)


app.post("/analyze", response_model=FinBertResponse | AnalyzeResponse)(analyze)

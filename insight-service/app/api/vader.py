"""HTTP wiring for vader; calculations live in app.core."""

from fastapi import APIRouter, FastAPI, HTTPException
from app.helper.validation_errors import install_validation_error_handler

from app.api.base import app as existing_app
from app.models.sentiment import AnalyzeRequest, AnalyzeResponse
from app.core.sentiment.lexicon import analyze_documents

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if request.model != "vader":
        raise HTTPException(status_code=501, detail="Only the vader model is implemented")
    return analyze_documents(request)


app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)


app.post("/analyze", response_model=AnalyzeResponse)(analyze)

from app.helper.metrics import install_metrics
install_metrics(app)

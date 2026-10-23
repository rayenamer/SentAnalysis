"""Additive entry point: uvicorn app.vader_api:app.

The original app.main application and its source remain unchanged.
"""

from fastapi import FastAPI, HTTPException
from app.validation_errors import install_validation_error_handler

from app.main import app as existing_app
from app.models.sentiment import AnalyzeRequest, AnalyzeResponse
from app.vader import analyze_documents

app = FastAPI(title=existing_app.title, version=existing_app.version)
install_validation_error_handler(app)
app.include_router(existing_app.router)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if request.model != "vader":
        raise HTTPException(status_code=501, detail="Only the vader model is implemented")
    return analyze_documents(request)


from fastapi import FastAPI
from app.models.sentiment import AnalyzeRequest
from app.validation_errors import install_validation_error_handler

app = FastAPI(
    title="SentAnalysis - Insight Service",
    version="0.1.0"
)
install_validation_error_handler(app)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "insight-service"
    }


@app.post("/analyze/validate")
def validate_analysis(request: AnalyzeRequest):
    return {
        "status": "valid",
        "documents_count": len(request.documents),
        "model": request.model
    }

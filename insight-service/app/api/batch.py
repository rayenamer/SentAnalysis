"""Batch HTTP wiring; model failures remain document outcomes."""

from fastapi import APIRouter
from app.models.batch import BatchAnalyzeRequest, BatchAnalyzeResponse
from app.core.batch import analyze_batch

router = APIRouter()


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
def batch(request: BatchAnalyzeRequest):
    """200 contains explicit success/error outcomes; invalid input returns 422."""
    return analyze_batch(request)



"""Explicit opt-in only: may download weights and export the real CPU model."""

import os

import pytest


@pytest.mark.skipif(os.getenv("RUN_FINBERT_INTEGRATION") != "1",
                    reason="Set RUN_FINBERT_INTEGRATION=1 for real CPU model inference")
def test_real_finbert_cpu():
    from app.finbert import CPU_PROVIDER, get_finbert_service
    from app.models.sentiment import AnalyzeRequest

    service = get_finbert_service()
    response = service.analyze(AnalyzeRequest(model="finbert", documents=[
        {"document_id": "profit", "title": "Results", "text":
         "The company reported strong profit growth and record earnings.", "tickers": ["TEST"]},
        {"document_id": "loss", "title": "Results", "text":
         "The company reported a significant loss and a sharp decline in revenue."},
    ]))
    assert service._backend[0].model.get_providers() == [CPU_PROVIDER]
    assert [r.label for r in response.results] == ["positive", "negative"]
    assert response.results[0].sentiment_score > 0
    assert response.results[1].sentiment_score < 0
    assert response.results[0].tickers == ["TEST"]
    for row in response.results:
        assert sum(row.probabilities.model_dump().values()) == pytest.approx(1)
        assert row.confidence == max(row.probabilities.model_dump().values())
    loaded = service._backend
    service.analyze(AnalyzeRequest(model="finbert", documents=[
        {"document_id": "repeat", "title": "Report", "text": "Revenue was unchanged."},
    ]))
    assert service._backend is loaded

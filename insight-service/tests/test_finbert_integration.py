"""Explicit opt-in only: may download weights and export the real CPU model."""

import os
from time import perf_counter

import pytest


@pytest.mark.skipif(os.getenv("RUN_FINBERT_INTEGRATION") != "1",
                    reason="Set RUN_FINBERT_INTEGRATION=1 for real CPU model inference")
def test_real_finbert_cpu():
    from app.finbert import CPU_PROVIDER, get_finbert_service
    from app.models.sentiment import AnalyzeRequest

    service = get_finbert_service()
    started = perf_counter()
    response = service.analyze(AnalyzeRequest(model="finbert", documents=[
        {"document_id": "profit", "title": "Results", "text":
         "The company reported strong profit growth and record earnings.", "tickers": ["TEST"]},
        {"document_id": "loss", "title": "Results", "text":
         "The company reported a significant loss and a sharp decline in revenue."},
    ]))
    cold_seconds = perf_counter() - started
    assert service._backend[0].model.get_providers() == [CPU_PROVIDER]
    assert [r.label for r in response.results] == ["positive", "negative"]
    assert response.results[0].sentiment_score > 0
    assert response.results[1].sentiment_score < 0
    assert response.results[0].tickers == ["TEST"]
    assert [r.document_id for r in response.results] == ["profit", "loss"]
    for row in response.results:
        assert sum(row.probabilities.model_dump().values()) == pytest.approx(1)
        assert row.confidence == max(row.probabilities.model_dump().values())
    loaded = service._backend
    started = perf_counter()
    neutral = service.analyze(AnalyzeRequest(model="finbert", documents=[
        {"document_id": "repeat", "title": "Report", "text": "The company is headquartered in Helsinki.", "tickers": ["TEST"]},
    ]))
    warm_seconds = perf_counter() - started
    assert service._backend is loaded
    assert neutral.results[0].label == "neutral"
    assert neutral.results[0].document_id == "repeat"
    assert neutral.results[0].tickers == ["TEST"]
    assert neutral.results[0].confidence == max(neutral.results[0].probabilities.model_dump().values())
    from app.benchmark_cpu import process_memory
    print({"cpu_provider": CPU_PROVIDER, "cold_two_documents_seconds": cold_seconds,
           "warm_one_document_seconds": warm_seconds,
           "memory_scope": "whole process, cumulative peak", **process_memory(),
           "labels": [r.label for r in response.results] + [neutral.results[0].label]})

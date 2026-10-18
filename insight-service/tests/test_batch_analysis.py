from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.batch_analysis import (BatchAnalyzeRequest, DocumentOutcome, analyze_batch)
from app.models.sentiment import AnalyzeResponse, SentimentResult
from app.finbert_models import FinBertResponse, FinBertResult, SentimentProbabilities


def incoming(count=5, model="vader", **kwargs):
    return BatchAnalyzeRequest(model=model, documents=[dict(document_id=f"doc-{i}",
        title="Financial report", text=f"Document {i}", tickers=["AAPL", f"T{i}"])
        for i in range(count)], **kwargs)


def response(request):
    rows = []
    for doc in request.documents:
        values = dict(document_id=doc.document_id, model=request.model, label="neutral",
                      sentiment_score=0, tickers=doc.tickers)
        if request.model == "finbert":
            rows.append(FinBertResult(**values, confidence=0.8,
                probabilities=SentimentProbabilities(positive=0.1, negative=0.1, neutral=0.8)))
        else:
            rows.append(SentimentResult(**values))
    return FinBertResponse(results=rows) if request.model == "finbert" else AnalyzeResponse(results=rows)


@pytest.mark.parametrize("model", ["vader", "finbert"])
@pytest.mark.parametrize("size", [1, 5, 10, 20, 100])
def test_all_inputs_have_aligned_outcomes_and_chunks_are_bounded(model, size):
    request = incoming(size, model=model, chunk_size=3)
    before = request.model_dump()
    runner = Mock(side_effect=response)
    result = analyze_batch(request, runner=runner)
    assert result.success_count == size and result.error_count == 0
    assert len(result.outcomes) == size
    for index, outcome in enumerate(result.outcomes):
        assert outcome.input_index == index
        assert outcome.document_id == request.documents[index].document_id
        assert outcome.tickers == request.documents[index].tickers
        assert outcome.result.model == model
    assert all(len(call.args[0].documents) <= 3 for call in runner.call_args_list)
    assert request.model_dump() == before


@pytest.mark.parametrize("model", ["vader", "finbert"])
def test_document_failure_isolated_without_losing_following_documents(model):
    def infer(request):
        if any(doc.document_id == "doc-2" for doc in request.documents):
            raise RuntimeError("private backend details")
        return response(request)
    request = incoming(model=model)
    runner = Mock(side_effect=infer)
    result = analyze_batch(request, runner=runner)
    assert result.success_count == 4 and result.error_count == 1
    assert [outcome.input_index for outcome in result.outcomes] == list(range(5))
    assert result.outcomes[2].error.code == "inference_failed"
    assert result.outcomes[2].result is None
    assert "private" not in result.model_dump_json()
    assert all(call.args[0].model == model for call in runner.call_args_list)


def test_complete_outage_still_returns_one_error_per_input():
    result = analyze_batch(incoming(), runner=Mock(side_effect=OSError("offline")))
    assert result.error_count == 5 and result.success_count == 0
    assert [row.document_id for row in result.outcomes] == [f"doc-{i}" for i in range(5)]


@pytest.mark.parametrize("change", ["reverse", "drop", "ticker", "model", "invalid"])
def test_contract_failure_never_associates_results_by_guessing(change):
    def infer(request):
        result = response(request)
        if change == "reverse":
            result.results.reverse()
        elif change == "drop":
            result.results.pop()
        elif change == "ticker":
            result.results[0].tickers = ["WRONG"]
        elif change == "model":
            result.results[0].model = "finbert"
        else:
            result.results[0].sentiment_score = float("nan")
        return result
    result = analyze_batch(incoming(3), runner=infer)
    assert result.error_count == 3
    assert all(row.error.code == "inference_contract_error" for row in result.outcomes)
    assert [row.document_id for row in result.outcomes] == ["doc-0", "doc-1", "doc-2"]


def test_repeated_document_ids_are_disambiguated_by_input_index():
    request = incoming(2)
    request.documents[1].document_id = request.documents[0].document_id
    result = analyze_batch(request, runner=response)
    assert [r.input_index for r in result.outcomes] == [0, 1]
    assert [r.tickers for r in result.outcomes] == [["AAPL", "T0"], ["AAPL", "T1"]]


@pytest.mark.parametrize("count", [0, 101])
def test_existing_request_size_limit_is_preserved(count):
    with pytest.raises(ValidationError):
        incoming(count)


@pytest.mark.parametrize("chunk", [0, 21, 1.5, True])
def test_chunk_limit_is_strict(chunk):
    with pytest.raises(ValidationError):
        incoming(chunk_size=chunk)


def test_outcome_contract_rejects_missing_result():
    with pytest.raises(ValidationError):
        DocumentOutcome(input_index=0, document_id="1", tickers=[], status="success")

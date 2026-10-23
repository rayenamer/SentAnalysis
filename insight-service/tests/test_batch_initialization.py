from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app import finbert
from app.batch_analysis import BatchAnalyzeRequest, analyze_batch
from app.inference_errors import ModelInitializationError


def request(count: int = 100) -> BatchAnalyzeRequest:
    return BatchAnalyzeRequest(model="finbert", documents=[dict(document_id=str(i),
        title="Earnings", text=f"Report {i}", tickers=["AAPL", f"T{i}"]) for i in range(count)])


def backend():
    model = Mock()
    model.config = SimpleNamespace(max_position_embeddings=512)
    model.side_effect = lambda **kw: SimpleNamespace(
        logits=np.tile([0., 0., 1.], (len(kw["input_ids"]), 1)))
    tokenizer = Mock(model_max_length=512)
    tokenizer.side_effect = lambda texts, **kw: {"input_ids": np.zeros((len(texts), 4), dtype=np.int64)}
    return model, tokenizer, {0: "positive", 1: "negative", 2: "neutral"}


def test_global_load_failure_attempts_once_and_preserves_every_document(monkeypatch):
    service = finbert.FinBertService()
    loader = Mock(side_effect=OSError("private model-path details"))
    monkeypatch.setattr(finbert, "get_finbert_service", lambda: service)
    monkeypatch.setattr(finbert, "_load_backend", loader)
    incoming = request()
    result = analyze_batch(incoming)
    loader.assert_called_once()
    assert result.success_count == 0 and result.error_count == 100
    for i, outcome in enumerate(result.outcomes):
        assert outcome.input_index == i
        assert outcome.document_id == str(i)
        assert outcome.tickers == incoming.documents[i].tickers
        assert outcome.error.code == "inference_failed"
        assert outcome.result is None
    assert "private" not in result.model_dump_json()


def test_next_request_retries_once_and_can_recover_then_reuses_model(monkeypatch):
    service = finbert.FinBertService()
    loader = Mock(side_effect=[OSError("offline"), backend()])
    monkeypatch.setattr(finbert, "get_finbert_service", lambda: service)
    monkeypatch.setattr(finbert, "_load_backend", loader)
    assert analyze_batch(request()).error_count == 100
    assert loader.call_count == 1
    recovered = analyze_batch(request())
    assert recovered.success_count == 100 and recovered.error_count == 0
    assert [r.document_id for r in recovered.outcomes] == [str(i) for i in range(100)]
    assert loader.call_count == 2
    assert analyze_batch(request(1)).success_count == 1
    assert loader.call_count == 2


def test_service_configuration_failure_is_global_and_retries_on_next_request(monkeypatch):
    factory = Mock(side_effect=ValueError("bad configuration"))
    monkeypatch.setattr(finbert, "get_finbert_service", factory)
    assert analyze_batch(request()).error_count == 100
    assert factory.call_count == 1
    assert analyze_batch(request(1)).error_count == 1
    assert factory.call_count == 2


def test_document_specific_failure_still_isolates_bad_document(monkeypatch):
    loaded = backend()
    tokenizer = loaded[1]
    def tokenize(texts, **kwargs):
        if "Report 2" in texts:
            raise ValueError("document-specific tokenization error")
        return {"input_ids": np.zeros((len(texts), 4), dtype=np.int64)}
    tokenizer.side_effect = tokenize
    service = finbert.FinBertService()
    loader = Mock(return_value=loaded)
    monkeypatch.setattr(finbert, "get_finbert_service", lambda: service)
    monkeypatch.setattr(finbert, "_load_backend", loader)
    result = analyze_batch(request(5))
    assert [r.status for r in result.outcomes] == ["success", "success", "error", "success", "success"]
    assert result.outcomes[2].document_id == "2"
    loader.assert_called_once()


def test_initialization_exception_remains_compatible_with_existing_error_type(monkeypatch):
    monkeypatch.setattr(finbert, "_load_backend", Mock(side_effect=OSError("offline")))
    with pytest.raises(ModelInitializationError) as error:
        finbert.FinBertService().analyze(request(1))
    assert isinstance(error.value, finbert.FinBertUnavailable)

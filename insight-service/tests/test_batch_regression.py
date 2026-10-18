from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app import finbert, vader
from app.models.sentiment import AnalyzeRequest


def request(count, model):
    return AnalyzeRequest(model=model, documents=[dict(document_id=f"doc-{i}", title="Report",
        text=str(i), tickers=["AAPL", f"T{i}"]) for i in range(count)])


@pytest.mark.parametrize("size", [1, 5, 10, 20, 100])
def test_existing_finbert_batch_keeps_logit_rows_aligned(monkeypatch, size):
    model = Mock()
    model.config.max_position_embeddings = 512
    model.side_effect = lambda **kw: SimpleNamespace(logits=np.array([
        [int(index), 0, 0] for index in kw["input_ids"][:, 0]], dtype=float))
    tokenizer = Mock(model_max_length=512)
    tokenizer.side_effect = lambda texts, **kw: {"input_ids": np.array([[int(text)] for text in texts])}
    loader = Mock(return_value=(model, tokenizer, {0: "positive", 1: "negative", 2: "neutral"}))
    monkeypatch.setattr(finbert, "_load_backend", loader)
    service = finbert.FinBertService(finbert.FinBertSettings(batch_size=4))
    incoming = request(size, "finbert")
    result = service.analyze(incoming)
    assert [r.document_id for r in result.results] == [d.document_id for d in incoming.documents]
    assert [r.tickers for r in result.results] == [d.tickers for d in incoming.documents]
    for i, row in enumerate(result.results):
        positive = 1 / (1 + 2 * np.exp(-i))
        negative = np.exp(-i) / (1 + 2 * np.exp(-i))
        assert row.sentiment_score == pytest.approx(positive - negative)
    assert max(len(c.args[0]) for c in tokenizer.call_args_list) <= 4
    loader.assert_called_once()


@pytest.mark.parametrize("size", [1, 5, 10, 20, 100])
def test_existing_vader_batch_keeps_ids_and_tickers_with_mocked_scores(monkeypatch, size):
    analyzer = Mock()
    analyzer.polarity_scores.side_effect = lambda text: {"compound": int(text) / 100}
    monkeypatch.setattr(vader, "get_analyzer", lambda: analyzer)
    incoming = request(size, "vader")
    result = vader.analyze_documents(incoming)
    assert len(result.results) == size
    for i, row in enumerate(result.results):
        assert row.document_id == incoming.documents[i].document_id
        assert row.tickers == incoming.documents[i].tickers
        assert row.sentiment_score == i / 100

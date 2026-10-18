from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import sys

import numpy as np
import pytest

from app import finbert
from app.models.sentiment import AnalyzeRequest, AnalyzeResponse


def request(count=3, model="finbert"):
    return AnalyzeRequest(model=model, documents=[{
        "document_id": str(i), "title": "Report", "text": f"Financial document {i}",
        "tickers": ["AAPL", "MSFT"],
    } for i in range(count)])


def backend(mapping=None):
    # Deliberately different label order: hardcoded Prosus indices would fail.
    mapping = mapping or {0: "neutral", 1: "negative", 2: "positive"}
    model = Mock()
    model.config = SimpleNamespace(max_position_embeddings=512)
    model.side_effect = lambda **kw: SimpleNamespace(
        logits=np.tile([1000., 1001., 1003.], (len(kw["input_ids"]), 1)))
    tokenizer = Mock(model_max_length=512)
    tokenizer.side_effect = lambda texts, **kw: {"input_ids": np.zeros((len(texts), 8), dtype=np.int64)}
    return model, tokenizer, mapping


@pytest.fixture
def mocked_backend(monkeypatch):
    value = backend()
    loader = Mock(return_value=value)
    monkeypatch.setattr(finbert, "_load_backend", loader)
    return value, loader


def test_lazy_single_load_and_configurable_batches(mocked_backend):
    (model, tokenizer, _), loader = mocked_backend
    service = finbert.FinBertService(finbert.FinBertSettings(batch_size=2, max_length=128))
    loader.assert_not_called()
    incoming = request(5)
    before = incoming.model_dump()
    result = service.analyze(incoming)
    service.analyze(request(1))
    loader.assert_called_once()
    assert [len(c.args[0]) for c in tokenizer.call_args_list] == [2, 2, 1, 1]
    assert all(c.kwargs == dict(padding=True, truncation=True, max_length=128, return_tensors="np")
               for c in tokenizer.call_args_list)
    assert [r.document_id for r in result.results] == [str(i) for i in range(5)]
    assert incoming.model_dump() == before
    assert model.call_count == 4


def test_probabilities_mapping_confidence_and_signed_score(mocked_backend):
    result = finbert.FinBertService().analyze(request(1))
    row = result.results[0]
    expected = np.exp([-3., -2., 0.]); expected /= expected.sum()
    assert row.label == "positive"
    assert row.probabilities.neutral == pytest.approx(expected[0])
    assert row.probabilities.negative == pytest.approx(expected[1])
    assert row.probabilities.positive == pytest.approx(expected[2])
    assert sum(row.probabilities.model_dump().values()) == pytest.approx(1)
    assert row.confidence == pytest.approx(expected[2])
    assert row.sentiment_score == pytest.approx(expected[2] - expected[1])
    assert row.tickers == ["AAPL", "MSFT"]
    assert AnalyzeResponse.model_validate(result.model_dump()).results[0].model == "finbert"


@pytest.mark.parametrize("logits,label,score", [
    ([0., 20., 0.], "negative", -1), ([20., 0., 0.], "neutral", 0),
])
def test_negative_and_neutral(mocked_backend, logits, label, score):
    (model, _, _), _loader = mocked_backend
    model.side_effect = lambda **kw: SimpleNamespace(logits=[logits])
    row = finbert.FinBertService().analyze(request(1)).results[0]
    assert row.label == label
    assert row.sentiment_score == pytest.approx(score, abs=1e-7)


def test_concurrent_first_use_loads_once(mocked_backend):
    _, loader = mocked_backend
    service = finbert.FinBertService()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: service.analyze(request(1)), range(8)))
    assert len(results) == 8
    loader.assert_called_once()


@pytest.mark.parametrize("logits", [[[float("nan"), 0, 1]], [[1, 2]], [[float("inf"), 1, 2]]])
def test_invalid_logits_fail_explicitly(mocked_backend, logits):
    (model, _, _), _loader = mocked_backend
    model.side_effect = lambda **kw: SimpleNamespace(logits=logits)
    with pytest.raises(finbert.FinBertUnavailable):
        finbert.FinBertService().analyze(request(1))


def test_load_failure_is_not_cached_or_replaced(monkeypatch):
    loader = Mock(side_effect=[OSError("missing model"), backend()])
    monkeypatch.setattr(finbert, "_load_backend", loader)
    service = finbert.FinBertService()
    with pytest.raises(finbert.FinBertUnavailable):
        service.analyze(request(1))
    assert service.analyze(request(1)).results[0].model == "finbert"
    assert loader.call_count == 2


@pytest.mark.parametrize("kwargs", [dict(batch_size=0), dict(batch_size=101),
    dict(max_length=513), dict(max_length=1), dict(intra_op_threads=0)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        finbert.FinBertSettings(**kwargs)


def test_wrong_model_does_not_load(mocked_backend):
    _, loader = mocked_backend
    with pytest.raises(ValueError):
        finbert.FinBertService().analyze(request(model="vader"))
    loader.assert_not_called()


def test_runtime_loader_enforces_cpu_and_uses_local_tokenizer(monkeypatch):
    model = Mock()
    model.model.get_providers.return_value = [finbert.CPU_PROVIDER]
    model.config.id2label = {"0": "positive", "1": "negative", "2": "neutral"}
    model_class = Mock(); model_class.from_pretrained.return_value = model
    ort = SimpleNamespace(
        get_available_providers=lambda: ["AzureExecutionProvider", finbert.CPU_PROVIDER],
        SessionOptions=SimpleNamespace,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    deps = SimpleNamespace(ort=ort, model=model_class, tokenizer=Mock())
    monkeypatch.setattr(finbert, "_dependencies", lambda: deps)
    monkeypatch.setattr(finbert, "_export_model", Mock(side_effect=AssertionError("must load local")))
    settings = finbert.FinBertSettings(onnx_dir=finbert.Path("local-onnx"))
    _, _, labels = finbert._load_backend(settings)
    kwargs = model_class.from_pretrained.call_args.kwargs
    assert kwargs["provider"] == finbert.CPU_PROVIDER
    assert kwargs["providers"] == [finbert.CPU_PROVIDER]
    assert kwargs["export"] is False and kwargs["use_io_binding"] is False
    assert kwargs["session_options"].intra_op_num_threads == 2
    deps.tokenizer.from_pretrained.assert_called_once_with(
        "local-onnx", local_files_only=True, trust_remote_code=False)
    assert labels == {0: "positive", 1: "negative", 2: "neutral"}
    model.model.get_providers.return_value = ["AzureExecutionProvider"]
    with pytest.raises(RuntimeError, match="exclusively"):
        finbert._load_backend(settings)
    model.model.get_providers.return_value = [finbert.CPU_PROVIDER]
    model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    with pytest.raises(ValueError, match="map three"):
        finbert._load_backend(settings)


def test_truncation_respects_model_and_tokenizer_limits(mocked_backend):
    (model, tokenizer, _), _loader = mocked_backend
    model.config.max_position_embeddings = 128
    tokenizer.model_max_length = 64
    finbert.FinBertService(finbert.FinBertSettings(max_length=512)).analyze(request(1))
    assert tokenizer.call_args.kwargs["max_length"] == 64


def test_process_singleton_uses_environment_once(monkeypatch):
    monkeypatch.setattr(finbert, "_service", None)
    monkeypatch.setenv("FINBERT_BATCH_SIZE", "2")
    monkeypatch.setenv("FINBERT_MAX_LENGTH", "128")
    monkeypatch.setenv("FINBERT_CPU_THREADS", "1")
    monkeypatch.delenv("FINBERT_ONNX_DIR", raising=False)
    service = finbert.get_finbert_service()
    assert service is finbert.get_finbert_service()
    assert service.settings.batch_size == 2
    assert service.settings.max_length == 128
    assert service.settings.intra_op_threads == 1
    assert service._backend is None


def test_export_is_cpu_only_and_writes_to_fresh_service_cache(monkeypatch):
    settings = finbert.FinBertSettings()
    cache, output = MagicMock(), MagicMock()
    cache.glob.return_value = []
    exporter = Mock()
    download = Mock(return_value="local-prosus-snapshot")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "optimum.exporters.onnx", SimpleNamespace(main_export=exporter))
    monkeypatch.setattr(finbert, "Path", Mock(side_effect=[cache, output]))
    make_directory = Mock(return_value="fresh-export")
    monkeypatch.setattr(finbert.tempfile, "mkdtemp", make_directory)
    assert finbert._export_model(settings) is output
    make_directory.assert_called_once_with(prefix="export-", dir=cache)
    kwargs = exporter.call_args.kwargs
    assert download.call_args.kwargs["repo_id"] == "ProsusAI/finbert"
    assert kwargs["model_name_or_path"] == "local-prosus-snapshot"
    assert kwargs["local_files_only"] is True
    assert kwargs["device"] == "cpu"
    assert kwargs["framework"] == "pt"
    assert kwargs["output"] is output
    assert kwargs["do_validation"] is False
    output.__truediv__.return_value.touch.assert_called_once_with(exist_ok=False)


def test_completed_export_is_reused_without_exporting(monkeypatch):
    settings = finbert.FinBertSettings()
    cache = MagicMock()
    marker = SimpleNamespace(parent="previously-exported-model")
    cache.glob.return_value = [marker]
    monkeypatch.setattr(finbert, "Path", Mock(return_value=cache))
    assert finbert._export_model(settings) == "previously-exported-model"
    cache.mkdir.assert_not_called()

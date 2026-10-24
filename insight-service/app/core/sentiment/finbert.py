"""Lazy Optimum/ONNX CPU inference for ProsusAI/finbert.

Only first use imports the ML stack. Standard tests replace the backend loader.
One process owns one service/model; run a single Uvicorn worker on small PCs.
Long bodies are truncated, not chunked or translated. Title is not scored.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import os
import tempfile

import numpy as np

from app.models.finbert import FinBertResponse, FinBertResult, SentimentProbabilities
from app.helper.inference_errors import FinBertUnavailable, ModelInitializationError
from app.models.sentiment import AnalyzeRequest

MODEL_ID = "ProsusAI/finbert"
CPU_PROVIDER = "CPUExecutionProvider"
LABELS = {"positive", "negative", "neutral"}


@dataclass(frozen=True)
class FinBertSettings:
    batch_size: int = 4
    max_length: int = 256
    intra_op_threads: int = 2
    cache_dir: Path = Path(__file__).resolve().parents[3] / ".finbert-cache"
    onnx_dir: Path | None = None

    def __post_init__(self):
        if not 1 <= self.batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100")
        if not 2 <= self.max_length <= 512:
            raise ValueError("max_length must be between 2 and 512")
        if self.intra_op_threads < 1:
            raise ValueError("intra_op_threads must be positive")


def _dependencies():
    # No Transformers, Torch, Optimum or ONNX imports on application startup.
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    return SimpleNamespace(ort=ort, model=ORTModelForSequenceClassification,
                           tokenizer=AutoTokenizer)


def _export_model(settings: FinBertSettings) -> Path:
    """Reuse completed exports, or export to a fresh directory without overwrites.

    Initial FP32 export requires PyTorch and more RAM than subsequent inference.
    Interrupted exports are retained, never reused or deleted by this code.
    All explicit download/export destinations are inside the service cache.
    """
    cache = Path(settings.cache_dir)
    completed = sorted(cache.glob("export-*/.complete"))
    if completed:
        return completed[0].parent
    from huggingface_hub import snapshot_download
    from optimum.exporters.onnx import main_export

    cache.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="export-", dir=cache))
    # Resolve one snapshot for weights, config, and tokenizer. Pass a local
    # source to Optimum because its preprocessor helpers do not forward cache_dir.
    source = snapshot_download(
        repo_id=MODEL_ID, cache_dir=str(cache / "hub"),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"],
    )
    main_export(
        model_name_or_path=source, output=output, task="text-classification",
        device="cpu", framework="pt", cache_dir=str(cache / "hub"),
        trust_remote_code=False, do_validation=False, local_files_only=True,
        batch_size=1, sequence_length=min(settings.max_length, 128),
    )
    # Optimum saves config and matching tokenizer alongside model.onnx.
    # Validate inference in the explicit integration test instead of creating
    # another model/session during export on memory-constrained machines.
    (output / ".complete").touch(exist_ok=False)
    return output


def _load_backend(settings: FinBertSettings):
    deps = _dependencies()
    if CPU_PROVIDER not in deps.ort.get_available_providers():
        raise RuntimeError("CPUExecutionProvider is unavailable")
    source = Path(settings.onnx_dir) if settings.onnx_dir else _export_model(settings)
    options = deps.ort.SessionOptions()
    options.intra_op_num_threads = settings.intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = deps.ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = deps.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = deps.model.from_pretrained(
        str(source), export=False, provider=CPU_PROVIDER, providers=[CPU_PROVIDER],
        session_options=options, use_io_binding=False, local_files_only=True,
        trust_remote_code=False,
    )
    if model.model.get_providers() != [CPU_PROVIDER]:
        raise RuntimeError("FinBERT must run exclusively on CPUExecutionProvider")
    tokenizer = deps.tokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False,
    )
    mapping = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
    if set(mapping) != {0, 1, 2} or set(mapping.values()) != LABELS:
        raise ValueError("FinBERT config must map three output indices to sentiment labels")
    return model, tokenizer, mapping


class FinBertService:
    def __init__(self, settings: FinBertSettings | None = None):
        self.settings = settings or FinBertSettings()
        self._backend = None
        self._lock = Lock()

    def analyze(self, request: AnalyzeRequest) -> FinBertResponse:
        if request.model != "finbert":
            raise ValueError("FinBERT service requires model='finbert'")
        # Serialize loading and inference to bound peak memory; prevent duplicate
        # first-load work when simultaneous requests arrive.
        with self._lock:
            if self._backend is None:
                try:
                    self._backend = _load_backend(self.settings)
                except Exception as exc:
                    # No failed initialization is cached on the shared service.
                    # A subsequent request may make a fresh initialization attempt.
                    raise ModelInitializationError("FinBERT initialization failed") from exc
            try:
                model, tokenizer, mapping = self._backend
                results = []
                limit = min(self.settings.max_length,
                            int(model.config.max_position_embeddings),
                            int(tokenizer.model_max_length))
                for start in range(0, len(request.documents), self.settings.batch_size):
                    documents = request.documents[start:start + self.settings.batch_size]
                    inputs = tokenizer(
                        [doc.text for doc in documents], padding=True, truncation=True,
                        max_length=limit, return_tensors="np",
                    )
                    logits = np.asarray(model(**inputs).logits, dtype=np.float64)
                    if logits.shape != (len(documents), 3) or not np.isfinite(logits).all():
                        raise ValueError("Invalid FinBERT logits")
                    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
                    probabilities = weights / weights.sum(axis=1, keepdims=True)
                    for document, row in zip(documents, probabilities):
                        scores = {mapping[i]: float(row[i]) for i in range(3)}
                        results.append(FinBertResult(
                            document_id=document.document_id,
                            label=mapping[int(row.argmax())],
                            confidence=float(row.max()),
                            sentiment_score=scores["positive"] - scores["negative"],
                            tickers=list(document.tickers),
                            probabilities=SentimentProbabilities(**scores),
                        ))
                return FinBertResponse(results=results)
            except Exception as exc:
                raise FinBertUnavailable("FinBERT CPU inference is unavailable") from exc


_service = None
_service_lock = Lock()


def get_finbert_service() -> FinBertService:
    """Read optional deployment settings once, then reuse one service per process."""
    global _service
    with _service_lock:
        if _service is None:
            _service = FinBertService(FinBertSettings(
                batch_size=int(os.getenv("FINBERT_BATCH_SIZE", "4")),
                max_length=int(os.getenv("FINBERT_MAX_LENGTH", "256")),
                intra_op_threads=int(os.getenv("FINBERT_CPU_THREADS", "2")),
                onnx_dir=Path(os.environ["FINBERT_ONNX_DIR"])
                if os.getenv("FINBERT_ONNX_DIR") else None,
            ))
        return _service

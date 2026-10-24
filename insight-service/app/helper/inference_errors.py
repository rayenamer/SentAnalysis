"""Inference failures shared without importing the model/runtime stack."""


class FinBertUnavailable(RuntimeError):
    """FinBERT loading or inference failed; never substitute another model."""


class ModelInitializationError(FinBertUnavailable):
    """A global initialization failure, not a document-specific inference error."""

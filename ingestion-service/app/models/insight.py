"""Small HTTP boundary contracts, checked against the actual Insight models in tests.

The services both use an `app` package, so runtime imports across checkouts would
be fragile. Keep only the wire boundary here; no scoring logic is duplicated.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.ingestion import IngestRequest, IngestResponse
from app.models.analytics import AnalysisModel, AnalyticsOptions, AnalyticsStage, SignalReports


class AnalyzeIngestionRequest(IngestRequest):
    model: AnalysisModel = "vader"
    chunk_size: int = Field(default=4, ge=1, le=20, strict=True)
    analytics: AnalyticsOptions = Field(default_factory=AnalyticsOptions)

    @model_validator(mode="after")
    def matching_history_model(self):
        if any(row.model != self.model for row in self.analytics.history):
            raise ValueError("History must use the requested analysis model")
        return self


class Probabilities(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    positive: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)


class InsightResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    document_id: str
    model: AnalysisModel
    label: Literal["positive", "neutral", "negative"]
    sentiment_score: float = Field(ge=-1, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    tickers: list[str]
    probabilities: Probabilities | None = None

    @model_validator(mode="after")
    def finbert_probabilities(self):
        if self.model == "finbert" and self.probabilities is None:
            raise ValueError("FinBERT requires probabilities")
        return self


class WireError(BaseModel):
    code: Literal["inference_failed", "inference_contract_error"]
    message: str


class WireOutcome(BaseModel):
    input_index: int = Field(ge=0, strict=True)
    document_id: str
    tickers: list[str]
    status: Literal["success", "error"]
    result: InsightResult | None = None
    error: WireError | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "success":
            if self.result is None or self.error is not None:
                raise ValueError("success requires only a result")
            if self.result.document_id != self.document_id or self.result.tickers != self.tickers:
                raise ValueError("result identity mismatch")
        elif self.error is None or self.result is not None:
            raise ValueError("error requires only error details")
        return self


class WireBatch(BaseModel):
    schema_version: Literal["1.0"]
    model: AnalysisModel
    outcomes: list[WireOutcome] = Field(max_length=100)
    success_count: int = Field(ge=0, strict=True)
    error_count: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def counts(self):
        if (self.success_count != sum(o.status == "success" for o in self.outcomes)
                or self.error_count != sum(o.status == "error" for o in self.outcomes)):
            raise ValueError("invalid batch counts")
        return self


class AnalysisOutcome(BaseModel):
    input_index: int
    document_id: str
    tickers: list[str]
    status: Literal["success", "error"]
    result: InsightResult | None = None
    error_code: str | None = None


class AnalysisReport(BaseModel):
    status: Literal["success", "partial", "error", "empty"]
    model: AnalysisModel
    outcomes: list[AnalysisOutcome]
    success_count: int
    error_count: int


class AnalyzeIngestionResponse(BaseModel):
    ingestion: IngestResponse
    analysis: AnalysisReport
    aggregation: AnalyticsStage
    weighted_aggregation: AnalyticsStage
    signals: SignalReports

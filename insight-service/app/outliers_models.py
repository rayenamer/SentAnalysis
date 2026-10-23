"""Historical Z-scores for comparable aggregate snapshots, not article counts.

One value per ticker/model/UTC time bucket. Default context is raw 24h means;
set aggregation_window/score_kind correctly when using other aggregates. A
series cannot mix contexts. Baseline is the open interval (t-lookback, t).
Sample standard deviation uses ddof=1. Undefined classifications stay null.
"""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.aggregation_models import ScoringModel, Window
from app.velocity_models import SentimentObservation, VelocityDiagnostics, VelocityRequest


class OutlierObservation(SentimentObservation):
    aggregation_window: Window = "24h"
    score_kind: Literal["raw", "weighted"] = "raw"


class OutliersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[OutlierObservation] = Field(max_length=100_000)
    lookback_hours: float = Field(default=168, ge=1 / 3_600_000_000,
                                  le=876_000, allow_inf_nan=False, strict=True)
    min_history: int = Field(default=5, ge=2, le=100_000, strict=True)
    moderate_threshold: float = Field(default=2, gt=0, allow_inf_nan=False, strict=True)
    strong_threshold: float = Field(default=3, gt=0, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def validate_series(self):
        if self.strong_threshold <= self.moderate_threshold:
            raise ValueError("strong_threshold must exceed moderate_threshold")
        # Reuse the established identical/conflicting UTC duplicate convention.
        VelocityRequest.check_conflicting_timestamps(self)
        contexts = {}
        for row in self.observations:
            key = (row.ticker, row.model)
            context = (row.aggregation_window, row.score_kind)
            if key in contexts and contexts[key] != context:
                raise ValueError("Cannot mix aggregation horizons or raw/weighted scores in one series")
            contexts[key] = context
        return self


class OutlierResult(BaseModel):
    ticker: str
    model: ScoringModel
    timestamp: AwareDatetime | None = None
    aggregation_window: Window
    score_kind: Literal["raw", "weighted"]
    current_sentiment: float | None = Field(default=None, allow_inf_nan=False)
    historical_mean: float | None = Field(default=None, allow_inf_nan=False)
    historical_std: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    historical_count: int | None = Field(default=None, ge=0)
    z_score: float | None = Field(default=None, allow_inf_nan=False)
    anomaly_detected: bool | None = None
    anomaly_direction: Literal["positive", "negative", "none"] | None = None
    severity: Literal["normal", "moderate_outlier", "strong_outlier"] | None = None
    diagnostic_reason: Literal["insufficient_history", "zero_variance", "missing_timestamp",
                               "missing_score", "numerical_error"] | None = None


class OutliersResponse(BaseModel):
    schema_version: str = "1.0"
    lookback_hours: float
    min_history: int
    moderate_threshold: float
    strong_threshold: float
    std_ddof: Literal[1] = 1
    results: list[OutlierResult]
    diagnostics: VelocityDiagnostics

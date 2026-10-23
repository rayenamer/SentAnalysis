"""Regular UTC aggregate snapshots for bivariate, intercept-only Granger tests.

Uses correlation contracts and exact bucket boundaries. Each series is isolated
by ticker/model/aggregation_window/score_kind. No filling or transformations.
For order p: effective_sample_size = N-p, unrestricted parameters = 2p+1,
residual df = N-3p-1. Library requires N > 3p+1; min_observations also applies.
Every order 1..max_lag is reported; orders are never reduced silently.
"""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.aggregation_models import ScoringModel, Window
from app.correlation_models import (CorrelationRequest, CorrelationSentiment,
    PriceObservation, TimeBucket, series_key)

Direction = Literal["sentiment_to_returns", "returns_to_sentiment"]


class GrangerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentiments: list[CorrelationSentiment] = Field(max_length=10_000)
    prices: list[PriceObservation] = Field(max_length=20_000)
    time_bucket: TimeBucket = "daily"
    max_lag: int = Field(default=3, ge=1, le=12, strict=True)
    alpha: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False, strict=True)
    min_observations: int = Field(default=40, ge=8, le=10_000, strict=True)
    directions: list[Direction] = Field(default_factory=lambda: ["sentiment_to_returns"],
                                         min_length=1, max_length=2)
    start_timestamp: AwareDatetime | None = None
    end_timestamp: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_alignment(self):
        validated = CorrelationRequest(sentiments=self.sentiments, prices=self.prices,
            time_bucket=self.time_bucket, start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp)
        self.start_timestamp, self.end_timestamp = validated.start_timestamp, validated.end_timestamp
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("directions must be unique")
        if len({series_key(row) for row in self.sentiments}) > 32:
            raise ValueError("At most 32 ticker/model/horizon/score-kind series per request")
        return self


class StationarityDiagnostic(BaseModel):
    test: Literal["adf"] = "adf"
    regression: Literal["c"] = "c"
    autolag: Literal["AIC"] = "AIC"
    significance_level: float = 0.05
    statistic: float | None = Field(default=None, allow_inf_nan=False)
    p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    used_lag: int | None = None
    observation_count: int | None = None
    reject_unit_root: bool | None = None
    diagnostic_reason: str | None = "not_computed"


class GrangerResult(BaseModel):
    ticker: str
    model: ScoringModel
    aggregation_window: Window
    score_kind: Literal["raw", "weighted"]
    time_bucket: TimeBucket
    direction: Direction
    lag: int
    observation_count: int
    effective_sample_size: int | None = None
    f_statistic: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    significance_level: float
    reject_null_hypothesis: bool | None = None
    stationarity_diagnostics: dict[str, StationarityDiagnostic] = Field(default_factory=dict)
    residual_diagnostics: dict[str, float] = Field(default_factory=dict)
    start_timestamp: AwareDatetime | None = None
    end_timestamp: AwareDatetime | None = None
    statistical_warnings: list[str] = Field(default_factory=list)
    diagnostic_reason: str | None = None


class GrangerResponse(BaseModel):
    schema_version: str = "1.0"
    results: list[GrangerResult]
    diagnostics: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=lambda: [
        "Granger predictability is not proof of real-world causation; non-rejection does not prove absence of predictive information.",
        "P-values are unadjusted across lags, directions and assets. Do not select the minimum as an unqualified discovery.",
        "Assumes stable linear dynamics, stationary series, appropriate lags and well-behaved residuals. ADF rejection is a screening result, not proof of stationarity; finite-sample F inference also relies on distributional assumptions.",
        "Exact UTC calendar buckets only: weekends and closures may break continuity. No filling, differencing, trading calendar, price downloads or model inference. Timestamps must reflect actual information availability.",
    ])

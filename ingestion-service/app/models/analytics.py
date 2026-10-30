"""Bounded HTTP analytics inputs; all calculations remain in Insight."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

AnalysisModel = Literal["vader", "finbert", "loughran_mcdonald"]
Window = Literal["1h", "24h", "7d"]


class HistoryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ticker: str = Field(min_length=1, max_length=32)
    model: AnalysisModel
    sentiment_score: float = Field(ge=-1, le=1, strict=True)
    observed_at: AwareDatetime
    aggregation_window: Window = "24h"
    score_kind: Literal["raw"] = "raw"

    @field_validator("ticker")
    @classmethod
    def ticker_normalized(cls, value):
        if not value.strip():
            raise ValueError("ticker must not be blank")
        return value.strip().upper()

    @field_validator("observed_at")
    @classmethod
    def utc_time(cls, value):
        return value.astimezone(timezone.utc)


class AnalyticsPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ticker: str = Field(min_length=1, max_length=32)
    timestamp: AwareDatetime
    close_price: float = Field(gt=0, strict=True)

    @field_validator("ticker")
    @classmethod
    def ticker_normalized(cls, value):
        return HistoryObservation.ticker_normalized(value)

    @field_validator("timestamp")
    @classmethod
    def utc_time(cls, value):
        return value.astimezone(timezone.utc)


class AnalyticsOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    signal_window: Window = "24h"
    history: list[HistoryObservation] = Field(default_factory=list, max_length=1000)
    prices: list[AnalyticsPrice] = Field(default_factory=list, max_length=2000)
    time_bucket: Literal["hourly", "daily"] = "hourly"

    @model_validator(mode="after")
    def comparable_observations(self):
        now = datetime.now(timezone.utc)
        seen = {}
        for row in self.history:
            if row.aggregation_window != self.signal_window:
                raise ValueError("History must use the selected aggregation window")
            if row.observed_at > now:
                raise ValueError("Future history is not allowed")
            key = row.ticker, row.model, row.observed_at
            if key in seen and seen[key] != row.sentiment_score:
                raise ValueError("Conflicting history observations")
            seen[key] = row.sentiment_score
        seen_prices = {}
        for row in self.prices:
            if row.timestamp > now:
                raise ValueError("Future prices are not allowed")
            key = row.ticker, row.timestamp
            if key in seen_prices and seen_prices[key] != row.close_price:
                raise ValueError("Conflicting prices")
            seen_prices[key] = row.close_price
            if not bucket_aligned(row.timestamp, self.time_bucket):
                raise ValueError("Prices must use exact UTC bucket-close timestamps")
        if self.prices and any(not bucket_aligned(row.observed_at, self.time_bucket) for row in self.history):
            raise ValueError("Market signals require bucket-aligned sentiment history")
        return self


def bucket_aligned(timestamp, bucket):
    return not (timestamp.minute or timestamp.second or timestamp.microsecond or
                (bucket == "daily" and timestamp.hour))


class AnalyticsStage(BaseModel):
    status: Literal["success", "unavailable", "error"]
    reason: str | None = None
    data: dict[str, JsonValue] | None = None


class SignalReports(BaseModel):
    velocity: AnalyticsStage
    outliers: AnalyticsStage
    correlation: AnalyticsStage
    granger: AnalyticsStage


class WireTickerAggregate(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    ticker: str = Field(min_length=1)
    model: AnalysisModel
    mean_sentiment: float = Field(ge=-1, le=1)
    article_count: int = Field(ge=1, strict=True)
    label_counts: dict[Literal["positive", "neutral", "negative"], int]
    distribution: dict[Literal["positive", "neutral", "negative"], float]


class WireWindow(BaseModel):
    window: Window
    start: AwareDatetime
    end: AwareDatetime
    aggregates: list[WireTickerAggregate]


class WireAggregation(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    schema_version: Literal["1.0"]
    as_of: AwareDatetime
    windows: list[WireWindow]
    diagnostics: dict[str, int]


class WireSignalRow(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    ticker: str
    model: AnalysisModel


class WireSignal(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    schema_version: Literal["1.0"]
    results: list[WireSignalRow]
    diagnostics: dict[str, int]


class VelocityRow(WireSignalRow):
    velocity_per_hour: float | None = Field(allow_inf_nan=False)
    current_observed_at: AwareDatetime | None
    current_sentiment: float | None = Field(ge=-1, le=1, allow_inf_nan=False)


class OutlierRow(WireSignalRow):
    z_score: float | None = Field(allow_inf_nan=False)
    diagnostic_reason: str | None
    historical_count: int | None = Field(ge=0)


class CorrelationRow(WireSignalRow):
    pearson_r: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    paired_observations: int = Field(ge=0, strict=True)
    diagnostic_reason: str | None


class GrangerRow(WireSignalRow):
    f_statistic: float | None = Field(ge=0, allow_inf_nan=False)
    p_value: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    observation_count: int = Field(ge=0, strict=True)
    diagnostic_reason: str | None


class WireVelocity(WireSignal):
    results: list[VelocityRow]


class WireOutliers(WireSignal):
    results: list[OutlierRow]


class WireCorrelation(WireSignal):
    results: list[CorrelationRow]


class WireGranger(WireSignal):
    results: list[GrangerRow]

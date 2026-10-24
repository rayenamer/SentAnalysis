"""Exact UTC bucket-close alignment; no resampling or backward/as-of matching.

hourly: closes on HH:00:00Z; daily: closes on 00:00:00Z. Bucket t denotes
(t-delta, t]. Sentiment must be available by its timestamp, not backdated.
Return t uses only close(t) and close(t-delta); missing/weekend buckets are not
compressed, so e.g. Monday returns need an actual Sunday close for daily data.
Lag k pairs S(t) with R(t+k*delta). Lag 0 is descriptive, not a predictive
feature; negative lags explicitly examine returns preceding the sentiment.
Defaults for aggregate metadata match the outlier engine: raw, 24h horizon.
"""

from datetime import timedelta
from typing import Annotated, Literal

from pydantic import AliasChoices, AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.aggregation import ScoringModel, Window
from app.models.outliers import OutlierObservation
from app.models.velocity import SentimentObservation

TimeBucket = Literal["hourly", "daily"]
BUCKET_DELTAS = {"hourly": timedelta(hours=1), "daily": timedelta(days=1)}


class CorrelationSentiment(OutlierObservation):
    sentiment_score: float | None = Field(default=None, ge=-1, le=1, strict=True,
        allow_inf_nan=False, validation_alias=AliasChoices("aggregated_sentiment", "sentiment_score"))
    observed_at: AwareDatetime | None = Field(default=None,
        validation_alias=AliasChoices("timestamp", "observed_at"))


class PriceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    timestamp: AwareDatetime | None = None
    close_price: float | None = Field(default=None, gt=0, strict=True, allow_inf_nan=False)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value):
        return SentimentObservation.normalize_ticker(value)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value):
        return SentimentObservation.normalize_time(value)


Lag = Annotated[int, Field(ge=-10_000, le=10_000, strict=True)]


def series_key(row):
    return row.ticker, row.model, row.aggregation_window, row.score_kind


class CorrelationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentiments: list[CorrelationSentiment] = Field(max_length=100_000)
    prices: list[PriceObservation] = Field(max_length=100_000)
    time_bucket: TimeBucket = "daily"
    lags: list[Lag] = Field(default_factory=lambda: [0], min_length=1, max_length=101)
    min_pairs: int = Field(default=3, ge=2, le=100_000, strict=True)
    start_timestamp: AwareDatetime | None = None
    end_timestamp: AwareDatetime | None = None

    @field_validator("start_timestamp", "end_timestamp")
    @classmethod
    def normalize_bounds(cls, value):
        return SentimentObservation.normalize_time(value)

    @model_validator(mode="after")
    def check_alignment_and_duplicates(self):
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be unique")
        if self.start_timestamp and self.end_timestamp and self.start_timestamp > self.end_timestamp:
            raise ValueError("start_timestamp must not exceed end_timestamp")
        if len({series_key(row) for row in self.sentiments}) * len(self.lags) > 10_000:
            raise ValueError("At most 10000 series/lag combinations are allowed")
        sentiment_seen, price_seen = {}, {}
        for rows, seen, sentiment in [(self.sentiments, sentiment_seen, True),
                                      (self.prices, price_seen, False)]:
            for row in rows:
                timestamp = row.observed_at if sentiment else row.timestamp
                if timestamp is None:
                    continue
                if timestamp.minute or timestamp.second or timestamp.microsecond or (
                    self.time_bucket == "daily" and timestamp.hour
                ):
                    raise ValueError("Observations must lie exactly on UTC bucket-close boundaries")
                value = row.sentiment_score if sentiment else row.close_price
                if value is None:
                    continue
                key = (series_key(row), timestamp) if sentiment else (row.ticker, timestamp)
                if key in seen and seen[key] != value:
                    raise ValueError("Conflicting duplicate sentiment or price at the same timestamp")
                seen[key] = value
        return self


Reason = Literal["insufficient_pairs", "constant_sentiment", "constant_returns",
                 "constant_sentiment_and_returns", "numerical_error"]


class CorrelationResult(BaseModel):
    ticker: str
    model: ScoringModel
    aggregation_window: Window
    score_kind: Literal["raw", "weighted"]
    time_bucket: TimeBucket
    lag: int
    correlation_method: Literal["pearson"] = "pearson"
    pearson_r: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    paired_observations: int = Field(ge=0)
    # Envelope of actual paired sentiment and return-end timestamps, not the
    # preceding close used to construct a return. None when no pairs exist.
    start_timestamp: AwareDatetime | None = None
    end_timestamp: AwareDatetime | None = None
    diagnostic_reason: Reason | None = None


class CorrelationDiagnostics(BaseModel):
    sentiment_input_count: int = 0
    price_input_count: int = 0
    missing_sentiment_timestamps: int = 0
    missing_sentiment_scores: int = 0
    missing_price_timestamps: int = 0
    missing_prices: int = 0
    sentiment_duplicates_removed: int = 0
    price_duplicates_removed: int = 0
    valid_returns: int = 0
    price_gaps: int = 0
    nonfinite_returns: int = 0


class CorrelationResponse(BaseModel):
    schema_version: str = "1.0"
    results: list[CorrelationResult]
    diagnostics: CorrelationDiagnostics
    warnings: list[str] = Field(default_factory=lambda: [
        "Pearson correlations are descriptive, not evidence of causality. Autocorrelated financial time series can make ordinary correlations and naive p-values misleading; no p-values or significance claims are provided.",
        "No filling or trading calendar: missing buckets, weekends and market closures break consecutive-bucket returns. Input timestamps must reflect actual information availability.",
    ])

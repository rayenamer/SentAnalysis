"""Contracts for observed sentiment, independent of model inference.

Supply one comparable series per ticker/model (e.g. repeated aggregate snapshots
using window.end as observed_at). Do not mix aggregation horizons or raw and
weighted means within a series. No missing observation is synthesized.
Stability is abs(velocity_per_hour) <= stability_threshold_per_hour, inclusive;
the default is 1e-6 sentiment units/hour, independent of the display time unit.
"""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.aggregation import AggregationRequest, ScoringModel, TimestampedSentimentResult

TimeUnit = Literal["hours", "minutes", "days"]
HOURS_PER_UNIT = {"hours": 1.0, "minutes": 1 / 60, "days": 24.0}


class SentimentObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    model: ScoringModel
    sentiment_score: float | None = Field(default=None, ge=-1, le=1,
                                         allow_inf_nan=False, strict=True)
    observed_at: AwareDatetime | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value):
        return TimestampedSentimentResult.normalize_tickers([value])[0]

    @field_validator("observed_at")
    @classmethod
    def normalize_time(cls, value):
        if value is None:
            return None
        try:
            return AggregationRequest.normalize_reference_time(value)
        except OverflowError as exc:
            raise ValueError("timestamp cannot be represented in UTC") from exc


class VelocityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[SentimentObservation] = Field(max_length=100_000)
    time_unit: TimeUnit = "hours"
    stability_threshold_per_hour: float = Field(default=1e-6, ge=0, allow_inf_nan=False,
        strict=True, description="Inclusive absolute rate threshold in sentiment units/hour")

    @model_validator(mode="after")
    def check_conflicting_timestamps(self):
        seen = {}
        for row in self.observations:
            if row.observed_at is None or row.sentiment_score is None:
                continue
            key = (row.ticker, row.model, row.observed_at)
            if key in seen and seen[key] != row.sentiment_score:
                raise ValueError("Conflicting scores at the same ticker/model/timestamp")
            seen[key] = row.sentiment_score
        return self


class VelocityResult(BaseModel):
    ticker: str
    model: ScoringModel
    previous_observed_at: AwareDatetime | None = None
    current_observed_at: AwareDatetime | None = None
    previous_sentiment: float | None = None
    current_sentiment: float | None = None
    sentiment_change: float | None = None
    elapsed_hours: float | None = None
    velocity_per_hour: float | None = None
    velocity_per_unit: float | None = None
    direction: Literal["increasing", "decreasing", "stable"] | None = None


class VelocityDiagnostics(BaseModel):
    input_count: int
    valid_observation_count: int
    duplicates_removed: int
    missing_timestamp_count: int
    missing_score_count: int


class VelocityResponse(BaseModel):
    schema_version: str = "1.0"
    time_unit: TimeUnit
    stability_threshold_per_hour: float
    results: list[VelocityResult]
    diagnostics: VelocityDiagnostics

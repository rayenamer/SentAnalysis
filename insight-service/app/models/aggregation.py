"""Aggregation contracts independent of inference and the original models.

Join existing result.model_dump() values with source publication timestamps.
FinBERT probabilities are accepted as extra data but are not used: aggregation
uses the supplied sentiment_score and label without rescoring or relabeling.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

from app.models.sentiment import SentimentResult
from app.helper.datetime_utils import normalize_utc

Window = Literal["1h", "24h", "7d"]
ScoringModel = Literal["vader", "finbert", "loughran_mcdonald"]
WINDOW_DURATIONS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24),
                    "7d": timedelta(days=7)}
NonEmptyString = Annotated[str, Field(min_length=1)]


class TimestampedSentimentResult(SentimentResult):
    document_id: NonEmptyString
    model: ScoringModel
    sentiment_score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    published_at: AwareDatetime | None = None

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")
        return value

    @field_validator("published_at")
    @classmethod
    def normalize_publication_time(cls, value):
        return normalize_utc(value) if value is not None else None

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, value: list[str]) -> list[str]:
        if any(not ticker.strip() for ticker in value):
            raise ValueError("tickers must not contain blank symbols")
        return sorted({ticker.strip().upper() for ticker in value})

    def aggregation_identity(self) -> tuple:
        """Only fields affecting aggregation participate in duplicate conflicts."""
        return (self.label, self.sentiment_score, self.published_at, tuple(self.tickers))


class AggregationRequest(BaseModel):
    results: list[TimestampedSentimentResult] = Field(max_length=100_000)
    as_of: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    windows: list[Window] = Field(default_factory=lambda: ["1h", "24h", "7d"],
                                 min_length=1, max_length=3)

    @field_validator("as_of")
    @classmethod
    def normalize_reference_time(cls, value):
        return normalize_utc(value)

    @field_validator("windows")
    @classmethod
    def unique_windows(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("windows must be unique")
        return value

    @model_validator(mode="after")
    def validate_time_range_and_duplicates(self):
        try:
            self.as_of - max(WINDOW_DURATIONS[window] for window in self.windows)
        except OverflowError as exc:
            raise ValueError("as_of is too early for the requested windows") from exc
        seen = {}
        for result in self.results:
            key = (result.document_id, result.model)
            identity = result.aggregation_identity()
            if key in seen and seen[key] != identity:
                raise ValueError(f"Conflicting duplicate for document {result.document_id!r} ({result.model})")
            seen[key] = identity
        return self


class LabelCounts(BaseModel):
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)


class LabelDistribution(BaseModel):
    """Fractions in [0, 1], calculated from supplied labels, not score thresholds."""

    positive: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)


class TickerAggregate(BaseModel):
    ticker: str
    model: ScoringModel
    mean_sentiment: float = Field(ge=-1, le=1)
    article_count: int = Field(ge=1)
    label_counts: LabelCounts
    distribution: LabelDistribution


class WindowAggregate(BaseModel):
    window: Window
    start: AwareDatetime
    end: AwareDatetime
    aggregates: list[TickerAggregate]


class AggregationDiagnostics(BaseModel):
    """Counts concern document/model records, not ticker contributions.

    Exclusion reasons are disjoint, evaluated in the listed order. Too-old
    counts refer to the longest requested window; duplicates count extra rows.
    """

    input_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    duplicates_removed: int = Field(ge=0)
    missing_timestamp_count: int = Field(ge=0)
    future_timestamp_count: int = Field(ge=0)
    missing_tickers_count: int = Field(ge=0)
    outside_window_count: int = Field(ge=0)


class AggregationResponse(BaseModel):
    schema_version: str = "1.0"
    as_of: AwareDatetime
    windows: list[WindowAggregate]
    diagnostics: AggregationDiagnostics

"""Source/engagement weighting layered over the existing aggregation engine.

weight = source_weight * (1 + engagement_factor * min(engagement, engagement_cap))
Missing engagement means zero; absent/unknown source uses default_source_weight.
Defaults give every document weight 1. Raw means, counts and distributions keep
their existing meaning. Weighted distributions use label weight / total weight.
Zero-weight articles remain in raw counts but not effective_document_count.
When total weight is zero, weighted mean/distribution are None, never fabricated.
"""

from collections import defaultdict
from math import fsum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.aggregation import aggregate_sentiments
from app.aggregation_models import (AggregationRequest, AggregationResponse,
    TimestampedSentimentResult, TickerAggregate, WindowAggregate, LabelDistribution)

SourceWeight = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]


class WeightingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_weights: dict[str, SourceWeight] = Field(default_factory=dict, max_length=1000)
    default_source_weight: SourceWeight = 1
    engagement_factor: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    engagement_cap: float = Field(default=100, ge=0, le=1_000_000, allow_inf_nan=False)

    @field_validator("source_weights")
    @classmethod
    def normalize_sources(cls, values):
        normalized = {}
        for source, weight in values.items():
            key = source.strip().casefold()
            if not key or key in normalized:
                raise ValueError("source weight names must be nonblank and unique after normalization")
            normalized[key] = weight
        return normalized


class WeightedSentimentResult(TimestampedSentimentResult):
    source: str | None = Field(default=None, max_length=256)
    engagement: float | None = Field(default=None, ge=0, le=1e12, allow_inf_nan=False)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value):
        if value is None:
            return None
        return value.strip().casefold() or None

    def aggregation_identity(self):
        return super().aggregation_identity() + (self.source, self.engagement)


class WeightedAggregationRequest(AggregationRequest):
    model_config = ConfigDict(extra="forbid")
    results: list[WeightedSentimentResult] = Field(max_length=100_000)
    weighting: WeightingPolicy = Field(default_factory=WeightingPolicy)


class WeightedTickerAggregate(TickerAggregate):
    total_weight: float = Field(ge=0, allow_inf_nan=False)
    effective_document_count: int = Field(ge=0)
    weighted_mean_sentiment: float | None = Field(default=None, ge=-1, le=1)
    weighted_distribution: LabelDistribution | None = None


class WeightedWindowAggregate(WindowAggregate):
    aggregates: list[WeightedTickerAggregate]


class WeightedAggregationResponse(AggregationResponse):
    windows: list[WeightedWindowAggregate]
    weighting: WeightingPolicy


def document_weight(result: WeightedSentimentResult, policy: WeightingPolicy) -> float:
    source = policy.source_weights.get(result.source, policy.default_source_weight)
    engagement = min(result.engagement or 0, policy.engagement_cap)
    return source * (1 + policy.engagement_factor * engagement)


def aggregate_weighted(request: WeightedAggregationRequest) -> WeightedAggregationResponse:
    # Reuse all established validation, deduplication, raw calculations and
    # diagnostics. Only the additional weighted statistics are calculated here.
    base = aggregate_sentiments(request)
    unique = {(r.document_id, r.model): r for r in request.results}
    windows = []
    for window in base.windows:
        groups = defaultdict(list)
        for row in unique.values():
            if row.published_at is not None and window.start < row.published_at <= window.end:
                weight = document_weight(row, request.weighting)
                for ticker in row.tickers:
                    groups[(ticker, row.model)].append((row, weight))
        aggregates = []
        for original in window.aggregates:
            rows = groups[(original.ticker, original.model)]
            total = fsum(weight for _, weight in rows)
            mean, distribution = None, None
            if total > 0:
                # Policy bounds keep products finite. Divide once to preserve
                # the exact +/-1 endpoints when every contributing score agrees.
                mean = fsum(row.sentiment_score * weight for row, weight in rows) / total
                distribution = LabelDistribution(**{
                    label: fsum(weight for row, weight in rows if row.label == label) / total
                    for label in ("positive", "neutral", "negative")})
            aggregates.append(WeightedTickerAggregate(**original.model_dump(),
                total_weight=total, effective_document_count=sum(weight > 0 for _, weight in rows),
                weighted_mean_sentiment=mean, weighted_distribution=distribution))
        windows.append(WeightedWindowAggregate(window=window.window, start=window.start,
                                              end=window.end, aggregates=aggregates))
    return WeightedAggregationResponse(schema_version=base.schema_version, as_of=base.as_of,
        windows=windows, diagnostics=base.diagnostics, weighting=request.weighting)

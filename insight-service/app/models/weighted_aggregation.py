"""Existing contracts relocated without validation changes."""

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.models.aggregation import (AggregationRequest, AggregationResponse, TimestampedSentimentResult, TickerAggregate, WindowAggregate, LabelDistribution)


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


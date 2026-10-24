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

from app.core.aggregate import aggregate_sentiments
from app.models.aggregation import (AggregationRequest, AggregationResponse,
    TimestampedSentimentResult, TickerAggregate, WindowAggregate, LabelDistribution)
from app.models.weighted_aggregation import (SourceWeight, WeightingPolicy,
    WeightedSentimentResult, WeightedAggregationRequest, WeightedTickerAggregate,
    WeightedWindowAggregate, WeightedAggregationResponse)


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

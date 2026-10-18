"""Stateless aggregation of existing sentiment scores; no inference or storage.

Rolling UTC windows are (as_of - duration, as_of]. A document contributes its
full score once to each distinct associated ticker. Article counts therefore
must not be summed across tickers to obtain a unique-document total.
"""

from collections import defaultdict
from math import fsum

from app.aggregation_models import (
    AggregationDiagnostics, AggregationRequest, AggregationResponse,
    LabelCounts, LabelDistribution, TickerAggregate, WindowAggregate,
    WINDOW_DURATIONS,
)


def aggregate_sentiments(request: AggregationRequest) -> AggregationResponse:
    """Compute equally weighted means separately for every ticker/model pair.

    Empty windows contain no aggregate rows; zero observations do not imply
    neutral sentiment. Missing dates are never replaced with the current time.
    Duplicate conflicts are rejected by AggregationRequest before this function.
    """
    unique = {}
    for result in request.results:
        unique.setdefault((result.document_id, result.model), result)

    diagnostics = AggregationDiagnostics(
        input_count=len(request.results), unique_count=len(unique),
        duplicates_removed=len(request.results) - len(unique),
        missing_timestamp_count=0, future_timestamp_count=0,
        missing_tickers_count=0, outside_window_count=0,
    )
    earliest = request.as_of - max(WINDOW_DURATIONS[w] for w in request.windows)
    eligible = []
    for result in unique.values():
        if result.published_at is None:
            diagnostics.missing_timestamp_count += 1
        elif result.published_at > request.as_of:
            diagnostics.future_timestamp_count += 1
        elif not result.tickers:
            diagnostics.missing_tickers_count += 1
        elif result.published_at <= earliest:
            diagnostics.outside_window_count += 1
        else:
            eligible.append(result)

    windows = []
    for window in request.windows:
        start = request.as_of - WINDOW_DURATIONS[window]
        groups = defaultdict(list)
        for result in eligible:
            if result.published_at > start:
                for ticker in result.tickers:
                    groups[(ticker, result.model)].append(result)
        aggregates = []
        for (ticker, model), results in sorted(groups.items()):
            counts = {label: 0 for label in ("positive", "neutral", "negative")}
            for result in results:
                counts[result.label] += 1
            count = len(results)
            aggregates.append(TickerAggregate(
                ticker=ticker, model=model, article_count=count,
                mean_sentiment=fsum(result.sentiment_score for result in results) / count,
                label_counts=LabelCounts(**counts),
                distribution=LabelDistribution(**{label: n / count for label, n in counts.items()}),
            ))
        windows.append(WindowAggregate(window=window, start=start, end=request.as_of,
                                       aggregates=aggregates))
    return AggregationResponse(as_of=request.as_of, windows=windows, diagnostics=diagnostics)

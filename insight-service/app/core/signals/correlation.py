"""Standard-library Pearson and exact bucket joins. No inference or network.

For each context/lag, pairs are assembled using actual timestamp arithmetic,
not array offsets. Price predecessors can be outside the analysis interval,
but both the sentiment time and return end must be within inclusive bounds.
Runtime O(p log p + s log s + s*lags), memory O(p+s), excluding result envelope.
"""

from math import fsum, isfinite, sqrt

from app.models.correlation import (BUCKET_DELTAS, CorrelationDiagnostics, CorrelationRequest,
    CorrelationResponse, CorrelationResult, series_key)


def pearson(pairs, min_pairs=3):
    if len(pairs) < min_pairs:
        return None, "insufficient_pairs"
    if any(not isfinite(value) for pair in pairs for value in pair):
        return None, "numerical_error"
    # Subtract anchors before scaling to retain tiny representable differences
    # around a large constant. For this domain x is [-1,1] and returns >= -1,
    # so anchor subtraction cannot overflow on validated observations.
    x = [pair[0] - pairs[0][0] for pair in pairs]
    y = [pair[1] - pairs[0][1] for pair in pairs]
    sx, sy = max(map(abs, x)), max(map(abs, y))
    if sx == 0 and sy == 0:
        return None, "constant_sentiment_and_returns"
    if sx == 0:
        return None, "constant_sentiment"
    if sy == 0:
        return None, "constant_returns"
    if not isfinite(sx) or not isfinite(sy):
        return None, "numerical_error"
    x = [value / sx for value in x]
    y = [value / sy for value in y]
    mx, my = fsum(x) / len(x), fsum(y) / len(y)
    x = [value - mx for value in x]
    y = [value - my for value in y]
    xx, yy = fsum(value * value for value in x), fsum(value * value for value in y)
    if xx == 0 or yy == 0:
        return None, "numerical_error"
    value = fsum(a * b for a, b in zip(x, y)) / (sqrt(xx) * sqrt(yy))
    if not isfinite(value):
        return None, "numerical_error"
    # Only floating round-off beyond a mathematical correlation bound is clipped.
    return max(-1.0, min(1.0, value)), None


def calculate_correlation(request: CorrelationRequest) -> CorrelationResponse:
    diagnostics = CorrelationDiagnostics(sentiment_input_count=len(request.sentiments),
                                         price_input_count=len(request.prices))
    groups, prices = {}, {}
    for row in request.sentiments:
        values = groups.setdefault(series_key(row), {})
        if row.observed_at is None:
            diagnostics.missing_sentiment_timestamps += 1
        elif row.sentiment_score is None:
            diagnostics.missing_sentiment_scores += 1
        elif row.observed_at in values:
            diagnostics.sentiment_duplicates_removed += 1
        else:
            values[row.observed_at] = row.sentiment_score
    for row in request.prices:
        values = prices.setdefault(row.ticker, {})
        if row.timestamp is None:
            diagnostics.missing_price_timestamps += 1
        elif row.close_price is None:
            diagnostics.missing_prices += 1
        elif row.timestamp in values:
            diagnostics.price_duplicates_removed += 1
        else:
            values[row.timestamp] = row.close_price
    delta = BUCKET_DELTAS[request.time_bucket]
    returns = {}
    for ticker, values in prices.items():
        outcomes = returns.setdefault(ticker, {})
        previous_time = previous_price = None
        for timestamp, price in sorted(values.items()):
            if previous_time is not None:
                if timestamp - previous_time != delta:
                    diagnostics.price_gaps += 1
                else:
                    result = price / previous_price - 1
                    if isfinite(result):
                        outcomes[timestamp] = result
                        diagnostics.valid_returns += 1
                    else:
                        diagnostics.nonfinite_returns += 1
            previous_time, previous_price = timestamp, price

    def within_bounds(timestamp):
        return ((request.start_timestamp is None or timestamp >= request.start_timestamp)
                and (request.end_timestamp is None or timestamp <= request.end_timestamp))

    results = []
    for key, sentiments in sorted(groups.items()):
        ticker, model, aggregation_window, score_kind = key
        outcomes = returns.get(ticker, {})
        ordered = sorted(sentiments.items())
        for lag in sorted(request.lags):
            pairs = []
            start = end = None
            for timestamp, score in ordered:
                if not within_bounds(timestamp):
                    continue
                try:
                    target = timestamp + lag * delta
                except OverflowError:
                    continue
                if target not in outcomes or not within_bounds(target):
                    continue
                pairs.append((score, outcomes[target]))
                lower, upper = min(timestamp, target), max(timestamp, target)
                start = lower if start is None else min(start, lower)
                end = upper if end is None else max(end, upper)
            value, reason = pearson(pairs, request.min_pairs)
            results.append(CorrelationResult(ticker=ticker, model=model,
                aggregation_window=aggregation_window, score_kind=score_kind,
                time_bucket=request.time_bucket, lag=lag, pearson_r=value,
                paired_observations=len(pairs), start_timestamp=start, end_timestamp=end,
                diagnostic_reason=reason))
    return CorrelationResponse(results=results, diagnostics=diagnostics)

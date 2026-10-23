"""CPU-only finite differences: O(n log n) time, O(n) memory, no inference.

Return a null-velocity row for the first valid observation of every series;
subsequent rows compare consecutive valid, unique timestamps. A series with
only missing data returns one wholly insufficient row. Missing data is skipped,
so elapsed time spans the actual gap; no interpolation or resampling is done.
Empty input yields no rows. Output order is ticker, model, then UTC timestamp.
"""

from app.velocity_models import (HOURS_PER_UNIT, VelocityDiagnostics, VelocityRequest,
                                 VelocityResponse, VelocityResult)


def calculate_velocity(request: VelocityRequest) -> VelocityResponse:
    groups = {}
    missing_time = missing_score = duplicates = 0
    for row in request.observations:
        series = groups.setdefault((row.ticker, row.model), {})
        # Exclusion counts are disjoint: missing timestamp takes precedence.
        if row.observed_at is None:
            missing_time += 1
        elif row.sentiment_score is None:
            missing_score += 1
        elif row.observed_at in series:
            duplicates += 1
        else:
            series[row.observed_at] = row.sentiment_score

    results = []
    for (ticker, model), series in sorted(groups.items()):
        if not series:
            results.append(VelocityResult(ticker=ticker, model=model))
            continue
        previous_time = previous_score = None
        for observed_at, score in sorted(series.items()):
            values = dict(ticker=ticker, model=model, current_observed_at=observed_at,
                          current_sentiment=score)
            if previous_time is not None:
                elapsed = (observed_at - previous_time).total_seconds() / 3600
                # Sorted unique timestamps guarantee elapsed > 0, even at
                # datetime's minimum positive separation of one microsecond.
                change = score - previous_score
                rate = change / elapsed
                direction = ("stable" if abs(rate) <= request.stability_threshold_per_hour
                             else "increasing" if rate > 0 else "decreasing")
                values.update(previous_observed_at=previous_time, previous_sentiment=previous_score,
                    sentiment_change=change, elapsed_hours=elapsed, velocity_per_hour=rate,
                    velocity_per_unit=rate * HOURS_PER_UNIT[request.time_unit], direction=direction)
            results.append(VelocityResult(**values))
            previous_time, previous_score = observed_at, score
    return VelocityResponse(time_unit=request.time_unit,
        stability_threshold_per_hour=request.stability_threshold_per_hour, results=results,
        diagnostics=VelocityDiagnostics(input_count=len(request.observations),
            valid_observation_count=sum(len(series) for series in groups.values()),
            duplicates_removed=duplicates, missing_timestamp_count=missing_time,
            missing_score_count=missing_score))

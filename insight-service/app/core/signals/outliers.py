"""Rolling sample Z-scores without model inference or third-party statistics.

Sorting is O(n log n). A two-stack aggregate queue maintains historical moments
in amortized O(1) per update without subtracting nearly equal running variances.
Each stack stores mergeable centered moments (count, mean, sum of squared
deviations); combining them gives sample variance M2/(n-1). Memory is O(n).
No Gaussian distribution, independence, significance or trading claim is made.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite, sqrt
from statistics import stdev

from app.models.outliers import OutlierResult, OutliersRequest, OutliersResponse
from app.models.velocity import VelocityDiagnostics


@dataclass(frozen=True)
class _Moments:
    count: int
    mean: float
    m2: float
    minimum: float
    maximum: float


def _merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    count = a.count + b.count
    delta = b.mean - a.mean
    return _Moments(count, a.mean + delta * b.count / count,
        a.m2 + b.m2 + delta * delta * (a.count * b.count / count),
        min(a.minimum, b.minimum), max(a.maximum, b.maximum))


class _History:
    def __init__(self):
        self.incoming = []
        self.outgoing = []

    @staticmethod
    def _push(stack, timestamp, value):
        previous = stack[-1][2] if stack else None
        stack.append((timestamp, value, _merge(previous, _Moments(1, value, 0, value, value))))

    def add(self, timestamp, value):
        self._push(self.incoming, timestamp, value)

    def expire(self, cutoff):
        while self.incoming or self.outgoing:
            if not self.outgoing:
                while self.incoming:
                    timestamp, value, _ = self.incoming.pop()
                    self._push(self.outgoing, timestamp, value)
            if self.outgoing[-1][0] > cutoff:
                break
            self.outgoing.pop()

    def statistics(self):
        summary = _merge(self.outgoing[-1][2] if self.outgoing else None,
                         self.incoming[-1][2] if self.incoming else None)
        if summary is None:
            return 0, None, None
        deviation = sqrt(summary.m2 / (summary.count - 1)) if summary.count > 1 else None
        if deviation == 0 and summary.minimum != summary.maximum:
            # Rare underflow for subnormal scores: statistics.stdev uses exact
            # rational accumulation internally. Never mistake it for flat data.
            deviation = stdev(item[1] for stack in (self.incoming, self.outgoing) for item in stack)
            if deviation == 0:
                deviation = None
        return summary.count, summary.mean, deviation


def calculate_outliers(request: OutliersRequest) -> OutliersResponse:
    groups = {}
    missing_time = missing_score = duplicates = 0
    for row in request.observations:
        valid, missing = groups.setdefault((row.ticker, row.model), ({}, []))
        if row.observed_at is None:
            missing_time += 1
            missing.append(row)
        elif row.sentiment_score is None:
            missing_score += 1
            missing.append(row)
        elif row.observed_at in valid:
            duplicates += 1
        else:
            valid[row.observed_at] = row

    results = []
    maximum_time = datetime.max.replace(tzinfo=timezone.utc)
    lookback = timedelta(hours=request.lookback_hours)
    for _, (valid, missing) in sorted(groups.items()):
        history = _History()
        series_results = []
        for timestamp, row in sorted(valid.items()):
            try:
                cutoff = timestamp - lookback
            except OverflowError:
                cutoff = None
            if cutoff is not None:
                history.expire(cutoff)
            count, mean, deviation = history.statistics()
            result = OutlierResult(ticker=row.ticker, model=row.model, timestamp=timestamp,
                aggregation_window=row.aggregation_window, score_kind=row.score_kind,
                current_sentiment=row.sentiment_score, historical_count=count,
                historical_mean=mean, historical_std=deviation)
            if count < request.min_history:
                result.diagnostic_reason = "insufficient_history"
            elif deviation == 0:
                result.diagnostic_reason = "zero_variance"
            elif deviation is None:
                result.diagnostic_reason = "numerical_error"
            else:
                z_score = (row.sentiment_score - mean) / deviation
                if not isfinite(z_score):
                    result.diagnostic_reason = "numerical_error"
                else:
                    detected = abs(z_score) >= request.moderate_threshold
                    result.z_score = z_score
                    result.anomaly_detected = detected
                    result.anomaly_direction = ("positive" if z_score > 0 else "negative") if detected else "none"
                    result.severity = ("strong_outlier" if abs(z_score) >= request.strong_threshold
                                       else "moderate_outlier" if detected else "normal")
            series_results.append(result)
            # Critical: add current only AFTER its own historical test.
            history.add(timestamp, row.sentiment_score)
        for row in missing:
            series_results.append(OutlierResult(ticker=row.ticker, model=row.model,
                timestamp=row.observed_at, current_sentiment=row.sentiment_score,
                aggregation_window=row.aggregation_window, score_kind=row.score_kind,
                diagnostic_reason="missing_timestamp" if row.observed_at is None else "missing_score"))
        series_results.sort(key=lambda r: (r.timestamp is None, r.timestamp or maximum_time,
            r.current_sentiment is None, r.current_sentiment if r.current_sentiment is not None else 0))
        results.extend(series_results)
    return OutliersResponse(lookback_hours=request.lookback_hours, min_history=request.min_history,
        moderate_threshold=request.moderate_threshold, strong_threshold=request.strong_threshold,
        results=results, diagnostics=VelocityDiagnostics(input_count=len(request.observations),
            valid_observation_count=sum(len(valid) for valid, _ in groups.values()),
            duplicates_removed=duplicates, missing_timestamp_count=missing_time,
            missing_score_count=missing_score))

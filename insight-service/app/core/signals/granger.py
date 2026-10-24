"""Stateless Granger adapter; statsmodels is imported only for eligible data.

No custom F-test. statsmodels tests whether column 2 predicts column 1, so
primary columns are [returns, sentiment]; reverse swaps them. Only lagged
regressors are used. No rows can bridge missing calendar buckets.

ADF with constant and AIC (at most 12 lags) must reject a unit root in BOTH
series at 5%; failure to reject is called 'stationarity_not_confirmed', not
proof of non-stationarity. No automatic differencing or significance hunting.
Standardized lag-design rank/condition checks reject singular/ill-conditioned
regressions (condition > 1e10). Breusch-Godfrey and Breusch-Pagan residual checks
gate inference at 5%; a failure or undefined residual diagnostic withholds F/p.
These checks are screens, not guarantees of all classical F-test assumptions.
"""

import inspect
from functools import lru_cache
from math import isfinite
from types import SimpleNamespace

from app.models.correlation import BUCKET_DELTAS, series_key
from app.models.granger import (GrangerRequest, GrangerResponse, GrangerResult,
                                StationarityDiagnostic)


class StatisticsUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _backend():
    try:
        import numpy as np
        from statsmodels.tsa.stattools import adfuller, grangercausalitytests
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
    except (ImportError, OSError, ValueError) as exc:
        raise StatisticsUnavailable("statsmodels and its dependencies are required") from exc
    return SimpleNamespace(np=np, adf=adfuller, granger=grangercausalitytests,
                           bg=acorr_breusch_godfrey, bp=het_breuschpagan)


def _aligned(request):
    """Reuse 7.3 contracts, keys and intervals; its return map is not public.

    Keep its minimal close_t/close_(t-delta)-1 preparation here rather than
    changing the working correlation implementation or trying to infer prices
    from correlation summary outputs.
    """
    groups, prices = {}, {}
    diagnostics = dict(missing_sentiments=0, missing_prices=0, sentiment_duplicates_removed=0,
                       price_duplicates_removed=0, nonfinite_returns=0)
    for row in request.sentiments:
        values = groups.setdefault(series_key(row), {})
        if row.observed_at is None or row.sentiment_score is None:
            diagnostics["missing_sentiments"] += 1
        elif row.observed_at in values:
            diagnostics["sentiment_duplicates_removed"] += 1
        else:
            values[row.observed_at] = row.sentiment_score
    for row in request.prices:
        values = prices.setdefault(row.ticker, {})
        if row.timestamp is None or row.close_price is None:
            diagnostics["missing_prices"] += 1
        elif row.timestamp in values:
            diagnostics["price_duplicates_removed"] += 1
        else:
            values[row.timestamp] = row.close_price
    delta = BUCKET_DELTAS[request.time_bucket]
    returns = {}
    for ticker, values in prices.items():
        outcomes = returns.setdefault(ticker, {})
        previous_time = previous_price = None
        for timestamp, price in sorted(values.items()):
            if previous_time is not None and timestamp - previous_time == delta:
                value = price / previous_price - 1
                if isfinite(value):
                    outcomes[timestamp] = value
                else:
                    diagnostics["nonfinite_returns"] += 1
            previous_time, previous_price = timestamp, price
    aligned = {}
    for key, sentiments in sorted(groups.items()):
        outcomes = returns.get(key[0], {})
        aligned[key] = [(t, sentiments[t], outcomes[t]) for t in sorted(sentiments)
            if t in outcomes and (request.start_timestamp is None or t >= request.start_timestamp)
            and (request.end_timestamp is None or t <= request.end_timestamp)]
    return aligned, diagnostics


def _adf(values, backend):
    try:
        options = dict(maxlag=min(12, len(values) // 2 - 2), regression="c", autolag="AIC")
        # statsmodels 0.15 adds an explicit tuple/result-object switch.
        if "result_object" in inspect.signature(backend.adf).parameters:
            options["result_object"] = False
        output = backend.adf(values, **options)
        statistic, p_value = float(output[0]), float(output[1])
        if not isfinite(statistic) or not isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError("invalid ADF result")
        return StationarityDiagnostic(statistic=statistic, p_value=p_value,
            used_lag=int(output[2]), observation_count=int(output[3]),
            reject_unit_root=p_value < 0.05, diagnostic_reason=None)
    except Exception:
        return StationarityDiagnostic(diagnostic_reason="adf_failed")


def _fit(values, lag, backend):
    """Return only vetted library results; raise a diagnostic on unsafe designs."""
    np = backend.np
    # Scale for numerical conditioning only; no temporal transformations. The
    # intercept-model F-test is invariant to nonzero affine rescaling.
    scale = np.max(np.abs(values), axis=0)
    normalized = values / scale
    normalized -= normalized.mean(axis=0)
    deviations = normalized.std(axis=0)
    if not np.isfinite(deviations).all() or (deviations == 0).any():
        raise ValueError("numerical_error")
    normalized /= deviations
    size = len(normalized)
    design = np.column_stack([np.ones(size-lag)] + [normalized[lag-i:size-i, j]
        for j in range(2) for i in range(1, lag+1)])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("singular_design")
    if np.linalg.cond(design) > 1e10:
        raise ValueError("ill_conditioned_design")
    kwargs = dict(maxlag=[lag], addconst=True)
    if "verbose" in inspect.signature(backend.granger).parameters:
        kwargs["verbose"] = False
    output = backend.granger(normalized, **kwargs)[lag]
    statistic, p_value = float(output[0]["ssr_ftest"][0]), float(output[0]["ssr_ftest"][1])
    if not isfinite(statistic) or statistic < 0 or not isfinite(p_value) or not 0 <= p_value <= 1:
        raise ValueError("numerical_error")
    unrestricted = output[1][1]
    if not np.isfinite(unrestricted.resid).all() or unrestricted.ssr <= np.finfo(float).eps * unrestricted.centered_tss:
        raise ValueError("perfect_fit")
    bg_p = float(backend.bg(unrestricted, nlags=lag)[1])
    bp_p = float(backend.bp(unrestricted.resid, unrestricted.model.exog)[1])
    if not all(isfinite(p) and 0 <= p <= 1 for p in (bg_p, bp_p)):
        raise ValueError("residual_diagnostic_failed")
    residuals = dict(breusch_godfrey_p_value=bg_p, breusch_pagan_p_value=bp_p)
    reason = "residual_autocorrelation" if bg_p < 0.05 else "residual_heteroskedasticity" if bp_p < 0.05 else None
    return statistic, p_value, residuals, reason


def calculate_granger(request: GrangerRequest) -> GrangerResponse:
    aligned, diagnostics = _aligned(request)
    results = []
    delta = BUCKET_DELTAS[request.time_bucket]
    for key, rows in aligned.items():
        count = len(rows)
        reason = None
        if any(rows[i][0] - rows[i-1][0] != delta for i in range(1, count)):
            reason = "insufficient_continuity"
        elif count < request.min_observations:
            reason = "insufficient_observations"
        elif len({row[1] for row in rows}) == 1:
            reason = "constant_sentiment"
        elif len({row[2] for row in rows}) == 1:
            reason = "constant_returns"
        backend, stationarity = None, {}
        if reason is None:
            try:
                backend = _backend()
            except StatisticsUnavailable:
                reason = "dependency_unavailable"
            if backend is not None:
                stationarity = {"sentiment": _adf([r[1] for r in rows], backend),
                                "returns": _adf([r[2] for r in rows], backend)}
                if any(s.reject_unit_root is None for s in stationarity.values()):
                    reason = "stationarity_test_failed"
                elif not all(s.reject_unit_root for s in stationarity.values()):
                    reason = "stationarity_not_confirmed"
        for direction in sorted(request.directions):
            for lag in range(1, request.max_lag+1):
                lag_reason = reason
                if reason != "insufficient_continuity" and (
                    count < request.min_observations or count <= 3*lag+1
                ):
                    lag_reason = "insufficient_observations"
                result = GrangerResult(ticker=key[0], model=key[1], aggregation_window=key[2],
                    score_kind=key[3], time_bucket=request.time_bucket, direction=direction,
                    lag=lag, observation_count=count,
                    effective_sample_size=None if reason == "insufficient_continuity" else max(0, count-lag),
                    significance_level=request.alpha, stationarity_diagnostics=stationarity,
                    start_timestamp=rows[0][0] if rows else None, end_timestamp=rows[-1][0] if rows else None,
                    diagnostic_reason=lag_reason)
                if lag_reason is None:
                    values = backend.np.asarray([[r[2], r[1]] if direction == "sentiment_to_returns"
                                                else [r[1], r[2]] for r in rows], dtype=float)
                    try:
                        statistic, p_value, residuals, failure = _fit(values, lag, backend)
                        result.residual_diagnostics = residuals
                        if failure:
                            result.diagnostic_reason = failure
                        else:
                            result.f_statistic, result.p_value = statistic, p_value
                            result.reject_null_hypothesis = p_value < request.alpha
                    except ValueError as exc:
                        allowed = {"singular_design", "ill_conditioned_design", "numerical_error",
                                   "perfect_fit", "residual_diagnostic_failed"}
                        result.diagnostic_reason = str(exc) if str(exc) in allowed else "statistical_test_failed"
                    except Exception:
                        result.diagnostic_reason = "statistical_test_failed"
                if result.diagnostic_reason:
                    result.statistical_warnings.append("No inferential conclusion: " + result.diagnostic_reason)
                if request.max_lag > 1 or len(request.directions) > 1:
                    result.statistical_warnings.append("Multiple tests; p-values are not adjusted.")
                results.append(result)
    return GrangerResponse(results=results, diagnostics=diagnostics)

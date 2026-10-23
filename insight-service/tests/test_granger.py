from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import json

import numpy as np
import pytest
from pydantic import ValidationError

from app import granger
from app.granger_models import GrangerRequest

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def dataset(count: int = 80, seed: int = 74, predictive: bool = True) -> dict[str, Any]:
    """Synthetic test inputs only; no prices are fabricated by production code."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-0.5, 0.5, count)
    returns = rng.normal(0, 0.002, count)
    if predictive:
        returns[1:] += 0.04 * scores[:-1]
    close = 100.0
    prices = [dict(ticker="AAPL", timestamp=NOW, close_price=close)]
    sentiments = []
    for i, (score, change) in enumerate(zip(scores, returns), 1):
        timestamp = NOW + timedelta(hours=i)
        close *= 1 + float(change)
        prices.append(dict(ticker="AAPL", timestamp=timestamp, close_price=close))
        sentiments.append(dict(ticker="AAPL", model="vader", timestamp=timestamp,
                                aggregated_sentiment=float(score)))
    return dict(time_bucket="hourly", sentiments=sentiments, prices=prices)


def compute(data: dict[str, Any] | None = None, **changes: Any):
    data = dataset() if data is None else data
    return granger.calculate_granger(GrangerRequest(**dict(data, **changes)))


@pytest.fixture
def mocked_statistics(monkeypatch):
    backend = SimpleNamespace(np=np, adf=Mock(side_effect=lambda values, **kw:
        (-5.0, 0.001, 1, len(values)-2, {})))
    monkeypatch.setattr(granger, "_backend", lambda: backend)
    fit = Mock(return_value=(4.0, 0.01, dict(breusch_godfrey_p_value=0.5,
                                            breusch_pagan_p_value=0.5), None))
    monkeypatch.setattr(granger, "_fit", fit)
    return backend, fit


def test_reports_every_order_and_both_directions_without_reversing_columns(mocked_statistics):
    _, fit = mocked_statistics
    data = dataset()
    result = compute(data, directions=["sentiment_to_returns", "returns_to_sentiment"])
    assert len(result.results) == 6
    aligned, _ = granger._aligned(GrangerRequest(**data))
    rows = next(iter(aligned.values()))
    for output, call in zip(result.results, fit.call_args_list):
        values, order, _ = call.args
        expected = [[r[2], r[1]] if output.direction == "sentiment_to_returns" else [r[1], r[2]] for r in rows]
        np.testing.assert_allclose(values, expected)
        assert order == output.lag
        assert output.effective_sample_size == 80 - order
        assert output.observation_count == 80
        assert output.p_value == 0.01 and output.reject_null_hypothesis is True
        assert output.stationarity_diagnostics["returns"].reject_unit_root is True


def test_alpha_boundary_fails_to_reject(mocked_statistics):
    _, fit = mocked_statistics
    fit.return_value = (1, 0.05, {}, None)
    assert compute(max_lag=1).results[0].reject_null_hypothesis is False


def test_each_lag_sample_size_is_independent(mocked_statistics):
    _, fit = mocked_statistics
    result = compute(dataset(10), min_observations=8)
    assert [r.diagnostic_reason for r in result.results] == [None, None, "insufficient_observations"]
    assert fit.call_count == 2


@pytest.mark.parametrize("kind,reason", [("sentiment", "constant_sentiment"), ("returns", "constant_returns")])
def test_constant_series_never_calls_statistics(monkeypatch, kind, reason):
    data = dataset()
    for row in data["sentiments"] if kind == "sentiment" else data["prices"]:
        row["aggregated_sentiment" if kind == "sentiment" else "close_price"] = 0.1 if kind == "sentiment" else 100
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(granger, "_backend", loader)
    assert all(r.diagnostic_reason == reason for r in compute(data).results)
    loader.assert_not_called()


def test_stationarity_failure_withholds_f_p_and_does_not_transform(mocked_statistics):
    backend, fit = mocked_statistics
    backend.adf.side_effect = lambda values, **kw: (-1.0, 0.9, 1, len(values)-2, {})
    result = compute()
    assert all(r.diagnostic_reason == "stationarity_not_confirmed" for r in result.results)
    assert all(r.p_value is r.f_statistic is r.reject_null_hypothesis is None for r in result.results)
    fit.assert_not_called()


@pytest.mark.parametrize("output", [(float("nan"), 0.01, 1, 20), (-5.0, 2, 1, 20)])
def test_invalid_adf_results_withhold_inference(mocked_statistics, output):
    backend, fit = mocked_statistics
    backend.adf.side_effect = None; backend.adf.return_value = output
    assert compute().results[0].diagnostic_reason == "stationarity_test_failed"
    fit.assert_not_called()


def test_dependency_unavailable_is_explicit_and_has_no_fabricated_statistics(monkeypatch):
    monkeypatch.setattr(granger, "_backend", Mock(side_effect=granger.StatisticsUnavailable("missing")))
    result = compute()
    assert all(r.diagnostic_reason == "dependency_unavailable" for r in result.results)
    assert all(r.p_value is r.f_statistic is r.reject_null_hypothesis is None for r in result.results)
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("kind", ["sentiment", "price"])
def test_missing_interior_period_never_compresses_lags(mocked_statistics, kind):
    _, fit = mocked_statistics
    data = dataset()
    del data["sentiments" if kind == "sentiment" else "prices"][20]
    output = compute(data)
    assert all(r.diagnostic_reason == "insufficient_continuity" for r in output.results)
    assert output.results[0].effective_sample_size is None
    fit.assert_not_called()


def test_explicit_missing_values_do_not_become_zero(mocked_statistics):
    _, fit = mocked_statistics
    data = dataset(); data["sentiments"][20]["aggregated_sentiment"] = None
    data["prices"][30]["close_price"] = None
    data["sentiments"].append(dict(ticker="OTHER", model="vader", timestamp=None))
    result = compute(data)
    assert result.diagnostics["missing_sentiments"] == 2
    assert result.diagnostics["missing_prices"] == 1
    assert any(r.diagnostic_reason == "insufficient_continuity" for r in result.results)
    fit.assert_not_called()


def test_unsorted_utc_duplicates_are_deterministic_and_input_is_preserved(mocked_statistics):
    data = dataset()
    duplicate = dict(data["sentiments"][0], ticker=" aapl ")
    duplicate["timestamp"] = duplicate["timestamp"].astimezone(timezone(timedelta(hours=3)))
    data["sentiments"].append(duplicate)
    data["prices"].append(dict(data["prices"][0]))
    request = GrangerRequest(**data); before = request.model_dump()
    output = granger.calculate_granger(request)
    assert request.model_dump() == before
    reverse = compute(dict(data, sentiments=list(reversed(data["sentiments"])), prices=list(reversed(data["prices"]))))
    assert output == reverse
    assert output.diagnostics["sentiment_duplicates_removed"] == 1
    assert output.diagnostics["price_duplicates_removed"] == 1


def test_bounds_exclude_future_rows_and_do_not_construct_future_features(mocked_statistics):
    _, fit = mocked_statistics
    data = dataset(50)
    end = NOW+timedelta(hours=40)
    original = compute(data, end_timestamp=end)
    captured = fit.call_args_list[0].args[0].copy()
    for row in data["sentiments"][40:]:
        row["aggregated_sentiment"] = 1
    for row in data["prices"][41:]:
        row["close_price"] = 1000
    fit.reset_mock()
    result = compute(data, end_timestamp=end)
    assert result.results == original.results
    np.testing.assert_array_equal(fit.call_args_list[0].args[0], captured)


def test_contexts_assets_models_and_weights_are_isolated(mocked_statistics):
    data = dataset()
    original = list(data["sentiments"])
    for changes in [dict(model="finbert"), dict(score_kind="weighted"), dict(aggregation_window="7d")]:
        data["sentiments"] += [dict(row, **changes) for row in original]
    data["sentiments"] += [dict(row, ticker="MSFT") for row in original]
    data["prices"] += [dict(row, ticker="MSFT") for row in list(data["prices"])]
    result = compute(data, max_lag=1)
    assert len(result.results) == 5
    assert all(r.observation_count == 80 for r in result.results)


def test_empty_and_insufficient_input_do_not_require_statsmodels(monkeypatch):
    loader = Mock(side_effect=AssertionError("must not import"))
    monkeypatch.setattr(granger, "_backend", loader)
    assert compute(dict(sentiments=[], prices=[])).results == []
    assert all(r.diagnostic_reason == "insufficient_observations" for r in compute(dataset(5)).results)
    loader.assert_not_called()


@pytest.mark.parametrize("changes", [dict(max_lag=0), dict(max_lag=13), dict(max_lag=True),
    dict(alpha=0), dict(alpha=1), dict(alpha=float("nan")), dict(min_observations=7),
    dict(directions=["unknown"]), dict(directions=["sentiment_to_returns"]*2)])
def test_invalid_config(changes):
    with pytest.raises(ValidationError):
        compute(**changes)


@pytest.mark.parametrize("kind,changes", [("prices", dict(close_price=0)),
    ("prices", dict(close_price=-1)), ("prices", dict(close_price=float("inf"))),
    ("sentiments", dict(aggregated_sentiment=float("nan"))),
    ("sentiments", dict(timestamp="2026-10-23T01:00:00")),
    ("sentiments", dict(timestamp=NOW+timedelta(minutes=1)))])
def test_existing_validation_is_reused(kind, changes):
    data = dataset(); data[kind][0].update(changes)
    with pytest.raises(ValidationError):
        compute(data)


@pytest.mark.parametrize("kind", ["sentiments", "prices"])
def test_conflicting_duplicates_are_rejected(kind):
    data = dataset()
    changes = dict(aggregated_sentiment=1) if kind == "sentiments" else dict(close_price=999)
    data[kind].append(dict(data[kind][0], **changes))
    with pytest.raises(ValidationError, match="Conflicting"):
        compute(data)


@pytest.mark.parametrize("reason", ["singular_design", "ill_conditioned_design", "perfect_fit", "numerical_error"])
def test_numerical_errors_are_null_results(mocked_statistics, reason):
    _, fit = mocked_statistics; fit.side_effect = ValueError(reason)
    result = compute().results[0]
    assert result.diagnostic_reason == reason and result.p_value is None


@pytest.mark.parametrize("reason", ["residual_autocorrelation", "residual_heteroskedasticity"])
def test_residual_failures_withhold_nominal_p_value(mocked_statistics, reason):
    _, fit = mocked_statistics
    fit.return_value = (4, 0.01, dict(breusch_godfrey_p_value=0.001), reason)
    output = compute().results[0]
    assert output.diagnostic_reason == reason and output.p_value is None
    assert output.residual_diagnostics["breusch_godfrey_p_value"] == 0.001


def test_actual_design_rank_guard_rejects_perfect_collinearity_before_library_call():
    values = np.random.default_rng(74).normal(size=80)
    backend = SimpleNamespace(np=np, granger=Mock())
    with pytest.raises(ValueError, match="singular_design"):
        granger._fit(np.column_stack([values, values]), 2, backend)
    backend.granger.assert_not_called()


def test_near_collinearity_is_rejected_before_library_inference():
    rng = np.random.default_rng(76)
    values = rng.normal(size=80)
    near_copy = values + 1e-11 * rng.normal(size=80)
    backend = SimpleNamespace(np=np, granger=Mock())
    with pytest.raises(ValueError, match="ill_conditioned_design"):
        granger._fit(np.column_stack([values, near_copy]), 2, backend)
    backend.granger.assert_not_called()


def test_actual_fit_adapter_uses_ssr_ftest_and_validates_residuals():
    values = np.random.default_rng(75).normal(size=(80, 2))
    unrestricted = SimpleNamespace(resid=np.ones(79), ssr=20, centered_tss=30,
                                   model=SimpleNamespace(exog=np.ones((79, 3))))
    library = Mock(return_value={1: ({"ssr_ftest": (4, 0.01, 76, 1)}, [None, unrestricted, None])})
    backend = SimpleNamespace(np=np, granger=library, bg=Mock(return_value=(1, .5)),
                              bp=Mock(return_value=(1, .5)))
    f, p, _, reason = granger._fit(values, 1, backend)
    assert f == 4 and p == .01 and reason is None
    assert library.call_args.kwargs == dict(maxlag=[1], addconst=True)


REAL_STATS = find_spec("statsmodels") is not None


@pytest.mark.skipif(not REAL_STATS, reason="statsmodels not installed; installation requires approval")
@pytest.mark.parametrize("predictive", [True, False])
def test_real_granger_matches_statsmodels_reference_for_seeded_series(predictive):
    """No assertion that every random sample must reject or fail to reject."""
    from statsmodels.tsa.stattools import grangercausalitytests
    import inspect
    data = dataset(400, predictive=predictive)
    result = compute(data, directions=["sentiment_to_returns", "returns_to_sentiment"])
    aligned, _ = granger._aligned(GrangerRequest(**data))
    rows = next(iter(aligned.values()))
    for output in result.results:
        values = np.array([[r[2], r[1]] if output.direction == "sentiment_to_returns" else [r[1], r[2]] for r in rows])
        kwargs = dict(maxlag=[output.lag], addconst=True)
        if "verbose" in inspect.signature(grangercausalitytests).parameters:
            kwargs["verbose"] = False
        reference = grangercausalitytests(values, **kwargs)[output.lag][0]["ssr_ftest"]
        if output.diagnostic_reason is None:
            assert output.f_statistic == pytest.approx(reference[0], rel=1e-9)
            assert output.p_value == pytest.approx(reference[1], rel=1e-8, abs=1e-14)
        else:
            assert output.diagnostic_reason in {"stationarity_not_confirmed", "residual_autocorrelation", "residual_heteroskedasticity"}


@pytest.mark.skipif(not REAL_STATS, reason="statsmodels not installed; installation requires approval")
def test_real_adf_nonstationary_trend_withholds_granger_results():
    data = dataset(400)
    for i, row in enumerate(data["sentiments"]):
        row["aggregated_sentiment"] = -0.5 + i / 400
    result = compute(data)
    assert all(r.p_value is None for r in result.results)
    assert all(r.diagnostic_reason in {"stationarity_not_confirmed", "stationarity_test_failed"} for r in result.results)

from datetime import datetime, timedelta, timezone
import math
import random
import statistics
import sys

import pytest
from pydantic import ValidationError

from app.outliers import calculate_outliers
from app.outliers_models import OutliersRequest

NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)


def row(score=0, hour=0, **changes):
    value = dict(ticker="AAPL", model="vader", sentiment_score=score,
                 observed_at=NOW + timedelta(hours=hour))
    value.update(changes)
    return value


def compute(rows, **config):
    return calculate_outliers(OutliersRequest(observations=rows, **config))


def sample(current):
    return [row(-0.125, 0), row(0, 1), row(0.125, 2), row(current, 3)]


@pytest.mark.parametrize("score,severity,direction,z", [
    (0, "normal", "none", 0), (0.125, "normal", "none", 1),
    (0.25, "moderate_outlier", "positive", 2), (-0.25, "moderate_outlier", "negative", -2),
    (0.375, "strong_outlier", "positive", 3), (-0.375, "strong_outlier", "negative", -3),
    (1, "strong_outlier", "positive", 8)])
def test_sample_statistics_exact_thresholds_and_signed_outliers(score, severity, direction, z):
    result = compute(sample(score), min_history=3).results[-1]
    assert result.historical_count == 3
    assert result.historical_mean == 0
    assert result.historical_std == 0.125
    assert result.z_score == z
    assert result.severity == severity and result.anomaly_direction == direction
    assert result.anomaly_detected == (severity != "normal")
    assert result.diagnostic_reason is None


def test_configurable_thresholds_and_minimum_history():
    result = compute(sample(0.125), min_history=3, moderate_threshold=0.5,
                     strong_threshold=1).results[-1]
    assert result.severity == "strong_outlier"
    missing = compute(sample(0.375), min_history=4).results[-1]
    assert missing.z_score is None and missing.historical_count == 3
    assert missing.diagnostic_reason == "insufficient_history"


def test_window_lower_boundary_is_excluded_and_current_never_enters_baseline():
    rows = [row(0.8, 0), row(-0.125, 1), row(0.125, 2), row(1, 3)]
    result = compute(rows, lookback_hours=3, min_history=2).results[-1]
    assert result.historical_count == 2
    assert result.historical_mean == 0
    assert result.historical_std == pytest.approx(math.sqrt(0.03125))
    short = compute(rows, lookback_hours=1, min_history=2).results[-1]
    assert short.historical_count == 0 and short.z_score is None


def test_unsorted_input_and_future_values_do_not_leak():
    original = compute(sample(0.25), min_history=3)
    rows = sample(0.25) + [row(-1, 10), row(1, 20)]
    shuffled = list(reversed(rows))
    result = compute(shuffled, min_history=3)
    assert result.results[:4] == original.results
    assert result == compute(rows, min_history=3)


def test_multiple_models_and_tickers_never_share_history():
    rows = sample(0.25) + [row(0.5, model="finbert"), row(-0.5, ticker="MSFT")]
    result = compute(rows, min_history=3)
    assert len(result.results) == 6
    for value in result.results:
        if value.ticker != "AAPL" or value.model != "vader":
            assert value.historical_count == 0 and value.z_score is None


def test_identical_duplicates_offsets_and_ticker_normalization():
    rows = sample(0.25)
    duplicate = row(-0.125, ticker=" aapl ")
    duplicate["observed_at"] = NOW.astimezone(timezone(timedelta(hours=3)))
    result = compute(rows + [duplicate], min_history=3)
    assert result.results == compute(rows, min_history=3).results
    assert result.diagnostics.duplicates_removed == 1
    assert all(value.timestamp.utcoffset() == timedelta(0) for value in result.results)


@pytest.mark.parametrize("rows", [[row(), row(1)], [row(1), row()]])
def test_conflicting_duplicates_are_rejected(rows):
    with pytest.raises(ValidationError, match="Conflicting"):
        compute(rows)


@pytest.mark.parametrize("changes", [dict(aggregation_window="1h"), dict(score_kind="weighted")])
def test_incompatible_scales_and_horizons_rejected(changes):
    with pytest.raises(ValidationError, match="Cannot mix"):
        compute([row(), row(0.2, 1, **changes)])


def test_zero_variance_never_becomes_infinite_z_score():
    result = compute([row(0.1, i) for i in range(3)] + [row(1, 3)], min_history=3).results[-1]
    assert result.historical_std == 0
    assert result.z_score is result.anomaly_detected is result.severity is result.anomaly_direction is None
    assert result.diagnostic_reason == "zero_variance"


def test_empty_one_history_and_insufficient_cases():
    assert compute([]).results == []
    result = compute([row(), row(0.25, 1)], min_history=2).results
    assert result[0].historical_mean is None and result[0].historical_std is None
    assert result[1].historical_count == 1 and result[1].historical_mean == 0
    assert result[1].historical_std is None
    assert all(r.diagnostic_reason == "insufficient_history" for r in result)


def test_missing_values_have_explicit_reasons_and_do_not_become_zero():
    rows = sample(0.25) + [row(None, 1.5), row(0.8, observed_at=None)]
    result = compute(rows, min_history=3)
    missing = [r for r in result.results if r.diagnostic_reason in ("missing_timestamp", "missing_score")]
    assert len(missing) == 2
    assert all(r.historical_count is None for r in missing)
    final = next(r for r in result.results if r.timestamp == NOW + timedelta(hours=3))
    assert final.historical_count == 3 and final.z_score == 2
    assert result.diagnostics.missing_score_count == result.diagnostics.missing_timestamp_count == 1
    assert result == compute(list(reversed(rows)), min_history=3)


@pytest.mark.parametrize("changes", [dict(sentiment_score=1.1), dict(sentiment_score=-1.1),
    dict(sentiment_score=float("nan")), dict(sentiment_score=float("inf")),
    dict(sentiment_score=True), dict(observed_at="invalid"),
    dict(observed_at="2026-10-23T12:00:00"), dict(model="other"), dict(ticker=" ")])
def test_invalid_observations(changes):
    with pytest.raises(ValidationError):
        compute([row(**changes)])


@pytest.mark.parametrize("config", [dict(min_history=1), dict(min_history=True),
    dict(lookback_hours=0), dict(lookback_hours=float("inf")), dict(moderate_threshold=0),
    dict(strong_threshold=2), dict(strong_threshold=float("nan"))])
def test_invalid_config(config):
    with pytest.raises(ValidationError):
        compute([], **config)


def test_rolling_moments_match_independent_standard_library_calculation():
    rng = random.Random(42)
    rows = [row(rng.uniform(-1, 1), i) for i in range(300)]
    result = compute(rows, lookback_hours=17.5, min_history=3)
    for index, value in enumerate(result.results):
        historical = [r["sentiment_score"] for r in rows[:index]
                      if value.timestamp - timedelta(hours=17.5) < r["observed_at"]]
        assert value.historical_count == len(historical)
        if historical:
            assert value.historical_mean == pytest.approx(statistics.mean(historical), abs=1e-14)
        if len(historical) >= 2:
            assert value.historical_std == pytest.approx(statistics.stdev(historical), abs=1e-14)


def test_minimum_datetime_does_not_discard_valid_history():
    start = datetime.min.replace(tzinfo=timezone.utc)
    rows = [row(-0.125, observed_at=start), row(0.125, observed_at=start+timedelta(hours=1)),
            row(0.5, observed_at=start+timedelta(hours=2))]
    assert compute(rows, min_history=2).results[-1].historical_count == 2


def test_subnormal_variance_and_overflow_never_emit_nonfinite_json():
    result = compute([row(-1e-310), row(1e-310, 1), row(1, 2)], min_history=2)
    assert result.results[-1].diagnostic_reason == "numerical_error"
    assert result.results[-1].z_score is None
    assert "Infinity" not in result.model_dump_json()


def test_statistical_engine_does_not_import_model_stack_in_new_process():
    import subprocess
    code = "from app.outliers import calculate_outliers; import sys; assert not ({'torch','transformers','onnxruntime','app.finbert'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

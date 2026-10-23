from datetime import datetime, timedelta, timezone
import json
import random
import statistics
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from app.correlation import calculate_correlation, pearson
from app.correlation_models import CorrelationRequest, CorrelationResponse

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def sentiment(score: float | None = 0, hour: float = 1, **changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = dict(ticker="AAPL", model="vader", aggregated_sentiment=score,
                 timestamp=NOW + timedelta(hours=hour))
    value.update(changes)
    return value


def price(close: float | None = 100, hour: float = 0, **changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = dict(ticker="AAPL", close_price=close, timestamp=NOW + timedelta(hours=hour))
    value.update(changes)
    return value


def dataset(lag: int = 0) -> dict[str, Any]:
    # Exact binary-friendly returns at times 1,2,3: -.5, 0, +.5.
    return dict(time_bucket="hourly", sentiments=[sentiment(value, i-lag)
        for i, value in enumerate([-0.5, 0, 0.5], 1)],
        prices=[price(value, i) for i, value in enumerate([100, 50, 50, 75])])


def compute(data: dict[str, Any] | None = None, **changes: Any) -> CorrelationResponse:
    data = dataset() if data is None else data
    return calculate_correlation(CorrelationRequest(**dict(data, **changes)))


@pytest.mark.parametrize("sign,expected", [(1, 1), (-1, -1)])
def test_perfect_correlations_from_actual_simple_price_returns(sign, expected):
    data = dataset()
    for row in data["sentiments"]:
        row["aggregated_sentiment"] *= sign
    value = compute(data).results[0]
    assert value.pearson_r == pytest.approx(expected)
    assert value.paired_observations == 3 and value.diagnostic_reason is None
    assert value.start_timestamp == NOW + timedelta(hours=1)
    assert value.end_timestamp == NOW + timedelta(hours=3)


def test_zero_correlation_is_defined_and_not_a_missing_value():
    assert pearson([(-1, 1), (0, -1), (1, 1)])[0] == pytest.approx(0)


@pytest.mark.parametrize("lag", [-2, -1, 0, 1, 2])
def test_signed_lag_uses_exact_bucket_shift(lag):
    result = compute(dataset(lag), lags=[lag]).results[0]
    assert result.lag == lag and result.paired_observations == 3
    assert result.pearson_r == pytest.approx(1)


def test_sparse_series_does_not_shift_by_array_position_or_fill_missing_prices():
    data: dict[str, Any] = dict(time_bucket="hourly", lags=[1],
        sentiments=[sentiment(-0.5, 0), sentiment(1, 2), sentiment(0.5, 4), sentiment(0, 6)],
        prices=[price(p, t) for t, p in [(0, 100), (1, 50), (2, 50),
                (4, 100), (5, 150), (6, 150), (7, 150), (8, 225)]])
    result = compute(data)
    # t=2 has no return ending t=3 and must not pair with the next available
    # return at t=5. Likewise no return spans the price gap t=2 -> t=4.
    assert result.results[0].paired_observations == 3
    assert result.results[0].pearson_r == pytest.approx(1)
    assert result.diagnostics.price_gaps == 1
    assert result.diagnostics.valid_returns == 6


def test_contexts_tickers_and_models_remain_separate():
    data = dataset()
    original = list(data["sentiments"])
    data["sentiments"] += [dict(r, score_kind="weighted") for r in original]
    data["sentiments"] += [dict(r, aggregation_window="1h") for r in original]
    data["sentiments"] += [dict(r, model="finbert") for r in original]
    data["sentiments"] += [dict(r, ticker="MSFT") for r in original]
    data["prices"] += [dict(r, ticker="MSFT") for r in list(data["prices"])]
    result = compute(data)
    assert len(result.results) == 5
    assert all(r.paired_observations == 3 and r.pearson_r == pytest.approx(1) for r in result.results)
    assert {(r.score_kind, r.aggregation_window) for r in result.results} == {
        ("raw", "24h"), ("weighted", "24h"), ("raw", "1h")}


def test_one_assets_prices_cannot_construct_anothers_returns():
    data = dataset(); data["prices"][1]["ticker"] = "OTHER"
    assert compute(data).results[0].paired_observations == 1


def test_input_order_duplicate_normalization_and_utc_offsets_are_deterministic():
    data = dataset()
    data["sentiments"].append(dict(data["sentiments"][0], ticker=" aapl ",
        timestamp=data["sentiments"][0]["timestamp"].astimezone(timezone(timedelta(hours=2)))))
    data["prices"].append(dict(data["prices"][0], ticker="aapl"))
    request = CorrelationRequest(**data); before = request.model_dump()
    result = calculate_correlation(request)
    assert request.model_dump() == before
    assert result.diagnostics.sentiment_duplicates_removed == 1
    assert result.diagnostics.price_duplicates_removed == 1
    assert result == compute(dict(data, sentiments=list(reversed(data["sentiments"])),
                                 prices=list(reversed(data["prices"]))))


@pytest.mark.parametrize("kind", ["sentiments", "prices"])
def test_conflicting_duplicates_are_rejected(kind):
    data = dataset()
    changes = dict(aggregated_sentiment=1) if kind == "sentiments" else dict(close_price=101)
    data[kind].append(dict(data[kind][0], **changes))
    with pytest.raises(ValidationError, match="Conflicting"):
        compute(data)


def test_undefined_statistics_have_specific_reasons():
    assert pearson([(0, 0)])[1] == "insufficient_pairs"
    assert pearson([(0, 1), (0, 2), (0, 3)])[1] == "constant_sentiment"
    assert pearson([(0, 1), (1, 1), (-1, 1)])[1] == "constant_returns"
    assert pearson([(0, 1)] * 3)[1] == "constant_sentiment_and_returns"
    assert all(pearson(pairs)[0] is None for pairs in [[(0, 0)], [(0, 1)] * 3])


def test_constant_prices_produce_real_zero_returns_not_undefined_price_data():
    data = dataset()
    for row in data["prices"]:
        row["close_price"] = 100
    result = compute(data).results[0]
    assert result.paired_observations == 3
    assert result.pearson_r is None and result.diagnostic_reason == "constant_returns"


def test_missing_data_and_empty_inputs_are_not_filled():
    data = dataset()
    data["sentiments"] += [sentiment(None, 4), sentiment(0, timestamp=None)]
    data["prices"] += [price(None, 4), price(100, timestamp=None)]
    result = compute(data)
    assert result.results[0].paired_observations == 3
    assert result.diagnostics.missing_sentiment_scores == result.diagnostics.missing_sentiment_timestamps == 1
    assert result.diagnostics.missing_prices == result.diagnostics.missing_price_timestamps == 1
    assert compute(dict(sentiments=[], prices=[])).results == []
    empty = compute(dict(time_bucket="hourly", sentiments=[sentiment(None)], prices=[])).results[0]
    assert empty.pearson_r is empty.start_timestamp is empty.end_timestamp is None
    assert empty.diagnostic_reason == "insufficient_pairs"


def test_each_lag_has_its_own_pair_count_and_diagnostic():
    values = {r.lag: r for r in compute(lags=[1, -1, 0]).results}
    assert values[0].paired_observations == 3 and values[0].pearson_r == pytest.approx(1)
    for lag in [-1, 1]:
        assert values[lag].paired_observations == 2
        assert values[lag].pearson_r is None
        assert values[lag].diagnostic_reason == "insufficient_pairs"
    explicit_small_sample = compute(lags=[1], min_pairs=2).results[0]
    assert explicit_small_sample.paired_observations == 2
    assert explicit_small_sample.pearson_r == pytest.approx(1)


def test_analysis_bounds_are_inclusive_and_predecessor_close_can_precede_start():
    data = dataset()
    result = compute(data, start_timestamp=NOW+timedelta(hours=1),
                     end_timestamp=NOW+timedelta(hours=3)).results[0]
    assert result.paired_observations == 3
    assert result.pearson_r == pytest.approx(1)


def test_future_observations_do_not_change_earlier_window_or_enter_features():
    data = dataset()
    bounds = dict(start_timestamp=NOW, end_timestamp=NOW+timedelta(hours=3), lags=[-1, 0, 1])
    original = compute(data, **bounds)
    data["prices"].append(price(150, 4))
    data["sentiments"].append(sentiment(1, 4))
    result = compute(data, **bounds)
    assert result.results == original.results
    assert next(r for r in result.results if r.lag == 1).paired_observations == 2


def test_daily_calendar_gaps_are_not_trading_day_array_shifts():
    data: dict[str, Any] = dict(time_bucket="daily", sentiments=[sentiment(0.1, 24*3), sentiment(0.2, 24*4)],
        prices=[price(100, 0), price(110, 24*3), price(120, 24*4)], min_pairs=2)
    result = compute(data)
    assert result.results[0].paired_observations == 1
    assert result.diagnostics.price_gaps == 1


@pytest.mark.parametrize("bucket,timestamp", [("hourly", NOW+timedelta(minutes=1)),
    ("hourly", NOW+timedelta(microseconds=1)), ("daily", NOW+timedelta(hours=1))])
@pytest.mark.parametrize("kind", ["sentiments", "prices"])
def test_off_boundary_observations_rejected_instead_of_rounding_into_past(bucket, timestamp, kind):
    data: dict[str, Any] = dict(time_bucket=bucket, sentiments=[sentiment(0, 0)], prices=[price()])
    data[kind][0]["timestamp"] = timestamp
    with pytest.raises(ValidationError, match="boundaries"):
        compute(data)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_prices(value):
    with pytest.raises(ValidationError):
        compute(dict(sentiments=[], prices=[price(value)]))


@pytest.mark.parametrize("value", [-1.1, 1.1, float("inf"), float("nan"), True])
def test_invalid_sentiment(value):
    with pytest.raises(ValidationError):
        compute(dict(sentiments=[sentiment(value)], prices=[]))


@pytest.mark.parametrize("changes", [dict(lags=[0, 0]), dict(lags=[True]), dict(lags=[1.5]),
    dict(lags=[]), dict(min_pairs=1), dict(time_bucket="weekly"),
    dict(start_timestamp=NOW+timedelta(days=1), end_timestamp=NOW),
    dict(start_timestamp="2026-10-23T00:00:00")])
def test_invalid_configuration(changes):
    with pytest.raises(ValidationError):
        compute(**changes)


def test_numerical_extremes_remain_finite_and_match_standard_library():
    assert pearson([(1e-310, 1e307), (2e-310, 2e307), (3e-310, 3e307)])[0] == pytest.approx(1)
    rng = random.Random(73)
    pairs = [(rng.uniform(-1, 1), rng.uniform(-0.5, 10)) for _ in range(100)]
    x_values, y_values = zip(*pairs)
    assert pearson(pairs)[0] == pytest.approx(statistics.correlation(x_values, y_values), abs=1e-14)
    assert pearson([(0, 0), (1, float("inf")), (0, 1)])[1] == "numerical_error"


def test_nonfinite_price_ratio_is_excluded_with_diagnostic_and_valid_json():
    result = compute(dict(time_bucket="hourly", sentiments=[sentiment(0, 1), sentiment(1, 2)],
        prices=[price(1e-310, 0), price(1e308, 1), price(1e308, 2)]))
    assert result.diagnostics.nonfinite_returns == 1
    assert result.results[0].paired_observations == 1
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_lag_beyond_datetime_range_yields_no_pairs():
    end = datetime(9999, 12, 31, tzinfo=timezone.utc)
    result = compute(dict(sentiments=[sentiment(0, timestamp=end)], prices=[], lags=[1]))
    assert result.results[0].paired_observations == 0


def test_existing_observation_names_remain_accepted_and_context_is_preserved():
    data = dataset()
    for row in data["sentiments"]:
        row["observed_at"] = row.pop("timestamp")
        row["sentiment_score"] = row.pop("aggregated_sentiment")
        row.update(score_kind="weighted", aggregation_window="7d")
    assert compute(data).results[0].score_kind == "weighted"
    assert compute(data).results[0].aggregation_window == "7d"


def test_statistical_engine_has_no_model_or_heavy_statistical_imports():
    code = "from app.correlation import calculate_correlation; import sys; assert not ({'torch','transformers','onnxruntime','app.finbert','numpy','pandas','scipy'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

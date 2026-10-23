from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.velocity import calculate_velocity
from app.velocity_models import SentimentObservation, VelocityRequest

NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)


def row(score=0, at=NOW, **changes):
    data = dict(ticker="AAPL", model="vader", sentiment_score=score, observed_at=at)
    data.update(changes)
    return data


def compute(rows, **settings):
    return calculate_velocity(VelocityRequest(observations=rows, **settings))


@pytest.mark.parametrize("previous,current,change,direction", [
    (-0.5, 0.5, 1, "increasing"), (0.5, -0.5, -1, "decreasing"),
    (0, 0, 0, "stable"), (-1, 0, 1, "increasing"), (0, -1, -1, "decreasing"),
    (-1, 1, 2, "increasing"), (-0.8, -0.2, 0.6, "increasing")])
def test_formula_and_signed_scores(previous, current, change, direction):
    result = compute([row(previous), row(current, NOW + timedelta(hours=2))])
    first, second = result.results
    assert first.velocity_per_hour is None and first.direction is None
    assert second.previous_sentiment == previous and second.current_sentiment == current
    assert second.sentiment_change == pytest.approx(change)
    assert second.elapsed_hours == 2
    assert second.velocity_per_hour == pytest.approx(change / 2)
    assert second.direction == direction


@pytest.mark.parametrize("unit,multiplier", [("hours", 1), ("minutes", 1 / 60), ("days", 24)])
def test_unit_conversion(unit, multiplier):
    value = compute([row(0), row(1, NOW + timedelta(hours=2))], time_unit=unit).results[-1]
    assert value.velocity_per_hour == 0.5
    assert value.velocity_per_unit == pytest.approx(0.5 * multiplier)


@pytest.mark.parametrize("change,direction", [(0.125, "stable"), (-0.125, "stable"),
    (0.126, "increasing"), (-0.126, "decreasing")])
def test_inclusive_rate_threshold_is_independent_of_display_unit(change, direction):
    value = compute([row(), row(change, NOW + timedelta(hours=1))],
        stability_threshold_per_hour=0.125, time_unit="minutes").results[-1]
    assert value.direction == direction


def test_unsorted_multiple_tickers_models_and_consecutive_pairs():
    rows = [row(1, NOW + timedelta(hours=2)), row(0.25, NOW + timedelta(hours=1)), row(),
            row(0.6, NOW + timedelta(hours=1), ticker="MSFT"), row(0, ticker="MSFT"),
            row(-0.5, model="finbert"), row(0, NOW + timedelta(hours=1), model="finbert")]
    incoming = VelocityRequest(observations=rows)
    before = incoming.model_dump()
    result = calculate_velocity(incoming)
    assert incoming.model_dump() == before
    aapl = [r for r in result.results if r.ticker == "AAPL" and r.model == "vader"]
    assert [r.velocity_per_hour for r in aapl] == [None, 0.25, 0.75]
    assert next(r for r in result.results if r.ticker == "MSFT" and r.velocity_per_hour is not None).velocity_per_hour == 0.6
    assert compute(list(reversed(rows))) == result


def test_equal_times_and_equivalent_offsets_collapse_without_zero_division():
    offset = NOW.astimezone(timezone(timedelta(hours=3)))
    result = compute([row(0.5), row(0.5, offset), row(0.5)])
    assert len(result.results) == 1 and result.results[0].velocity_per_hour is None
    assert result.diagnostics.duplicates_removed == 2
    assert result.results[0].current_observed_at.utcoffset() == timedelta(0)


def test_conflicting_duplicate_time_rejected_in_both_orders():
    for rows in ([row(), row(1)], [row(1), row()]):
        with pytest.raises(ValidationError, match="Conflicting"):
            compute(rows)


def test_missing_data_skipped_and_actual_gap_used():
    result = compute([row(-0.5), row(None, NOW + timedelta(hours=1)), row(0.1, None),
                      row(0.5, NOW + timedelta(hours=4))])
    assert len(result.results) == 2
    assert result.results[-1].elapsed_hours == 4
    assert result.results[-1].velocity_per_hour == 0.25
    assert result.diagnostics.missing_score_count == result.diagnostics.missing_timestamp_count == 1


def test_empty_single_and_wholly_missing_series():
    assert compute([]).results == []
    singleton = compute([row()]).results[0]
    assert singleton.current_sentiment == 0 and singleton.previous_sentiment is None
    missing = compute([row(None, None)]).results[0]
    assert missing.current_sentiment is None and missing.current_observed_at is None
    assert missing.velocity_per_hour is None and missing.direction is None


def test_microsecond_gap_and_zero_threshold_are_supported():
    value = compute([row(-1), row(1, NOW + timedelta(microseconds=1))],
                    stability_threshold_per_hour=0).results[-1]
    assert value.elapsed_hours == pytest.approx(1 / 3_600_000_000)
    assert value.velocity_per_hour == pytest.approx(7_200_000_000)
    assert value.direction == "increasing"


@pytest.mark.parametrize("changes", [dict(sentiment_score=1.01), dict(sentiment_score=-1.01),
    dict(sentiment_score=float("nan")), dict(sentiment_score=float("inf")),
    dict(sentiment_score=True), dict(sentiment_score="0.2"), dict(ticker=" "),
    dict(model="unknown"), dict(observed_at="2026-10-23T12:00:00"), dict(observed_at="bad")])
def test_invalid_observations_rejected(changes):
    data = row(); data.update(changes)
    with pytest.raises(ValidationError):
        SentimentObservation(**data)


@pytest.mark.parametrize("settings", [dict(stability_threshold_per_hour=-1),
    dict(stability_threshold_per_hour=float("nan")), dict(time_unit="weeks"),
    dict(stability_threshold_per_hour=True)])
def test_invalid_settings(settings):
    with pytest.raises(ValidationError):
        compute([], **settings)


def test_ticker_normalization_and_missing_optional_fields():
    result = compute([dict(ticker=" aapl ", model="finbert")])
    assert result.results[0].ticker == "AAPL"
    assert result.diagnostics.missing_timestamp_count == 1

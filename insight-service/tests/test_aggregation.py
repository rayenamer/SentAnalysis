from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.aggregation import aggregate_sentiments
from app.aggregation_models import AggregationRequest, TimestampedSentimentResult
from app.finbert_models import FinBertResult, SentimentProbabilities
from app.models.sentiment import SentimentResult

NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)


def row(document_id="1", **changes):
    data = dict(document_id=document_id, model="vader", label="positive",
                sentiment_score=0.6, tickers=["AAPL"], published_at=NOW)
    data.update(changes)
    return data


def calculate(results, **changes):
    return aggregate_sentiments(AggregationRequest(results=results, as_of=NOW, **changes))


def test_mean_counts_and_distribution():
    result = calculate([row(), row("2", sentiment_score=-0.3, label="negative"),
                        row("3", sentiment_score=0, label="neutral")])
    for window in result.windows:
        aggregate = window.aggregates[0]
        assert aggregate.article_count == 3
        assert aggregate.mean_sentiment == pytest.approx(0.1)
        assert aggregate.label_counts.model_dump() == dict(positive=1, neutral=1, negative=1)
        assert all(n == pytest.approx(1 / 3) for n in aggregate.distribution.model_dump().values())
        assert sum(aggregate.distribution.model_dump().values()) == pytest.approx(1)


@pytest.mark.parametrize("window,delta", [("1h", timedelta(hours=1)),
    ("24h", timedelta(hours=24)), ("7d", timedelta(days=7))])
def test_exact_time_boundaries(window, delta):
    result = calculate([
        row("start", published_at=NOW - delta),
        row("inside", published_at=NOW - delta + timedelta(microseconds=1)),
        row("end"), row("future", published_at=NOW + timedelta(microseconds=1)),
    ], windows=[window])
    output = result.windows[0]
    assert output.start == NOW - delta
    assert output.end == NOW
    assert output.aggregates[0].article_count == 2
    assert result.diagnostics.future_timestamp_count == 1
    assert result.diagnostics.outside_window_count == 1


def test_window_membership_with_older_articles():
    result = calculate([row(), row("2", published_at=NOW - timedelta(hours=2)),
                        row("3", published_at=NOW - timedelta(days=2)),
                        row("4", published_at=NOW - timedelta(days=8))])
    assert [w.aggregates[0].article_count for w in result.windows] == [1, 2, 3]
    assert result.diagnostics.outside_window_count == 1


def test_duplicates_normalization_and_multiple_tickers_do_not_mutate_input():
    incoming = AggregationRequest(as_of=NOW, results=[
        row(tickers=[" aapl ", "AAPL", "msft"]), row(tickers=["MSFT", "AAPL"]),
    ])
    before = incoming.model_dump()
    result = aggregate_sentiments(incoming)
    assert incoming.model_dump() == before
    assert result.diagnostics.duplicates_removed == 1
    assert [(a.ticker, a.article_count, a.mean_sentiment) for a in result.windows[0].aggregates] == [
        ("AAPL", 1, 0.6), ("MSFT", 1, 0.6)]


@pytest.mark.parametrize("changes", [dict(sentiment_score=0.1), dict(label="neutral"),
    dict(published_at=None), dict(tickers=["MSFT"])])
def test_conflicting_duplicates_are_rejected(changes):
    with pytest.raises(ValidationError, match="Conflicting duplicate"):
        calculate([row(), row(**changes)])


def test_model_groups_are_separate_even_for_same_document():
    result = calculate([row(), row(model="finbert", sentiment_score=-0.8, label="negative")])
    assert [(a.model, a.article_count, a.mean_sentiment) for a in result.windows[0].aggregates] == [
        ("finbert", 1, -0.8), ("vader", 1, 0.6)]
    assert result.diagnostics.duplicates_removed == 0


def test_empty_dataset_and_empty_windows_have_no_invented_means():
    result = calculate([])
    assert len(result.windows) == 3
    assert all(w.aggregates == [] for w in result.windows)
    assert all(n == 0 for n in result.diagnostics.model_dump().values())
    older = calculate([row(published_at=NOW - timedelta(days=2))])
    assert older.windows[0].aggregates == older.windows[1].aggregates == []


def test_missing_dates_future_dates_and_missing_tickers_are_reported_once():
    result = calculate([row("missing", published_at=None), row("missing", published_at=None),
                        row("future", published_at=NOW + timedelta(days=1)), row("none", tickers=[])])
    assert all(w.aggregates == [] for w in result.windows)
    assert result.diagnostics.model_dump() == dict(input_count=4, unique_count=3,
        duplicates_removed=1, missing_timestamp_count=1, future_timestamp_count=1,
        missing_tickers_count=1, outside_window_count=0)


def test_timezone_offsets_and_equivalent_duplicate_timestamps():
    offset = NOW.astimezone(timezone(timedelta(hours=3)))
    result = calculate([row(published_at=offset), row()])
    assert result.diagnostics.duplicates_removed == 1
    parsed = AggregationRequest(results=[row(published_at=offset)], as_of=offset)
    assert parsed.as_of.utcoffset() == timedelta(0)
    assert parsed.results[0].published_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("changes", [dict(published_at="2026-10-18T12:00:00"),
    dict(document_id=" "), dict(tickers=[" "]), dict(sentiment_score=1.01),
    dict(sentiment_score=float("nan")), dict(sentiment_score=float("inf")),
    dict(model="other"), dict(label="other")])
def test_invalid_result_contract(changes):
    with pytest.raises(ValidationError):
        TimestampedSentimentResult(**row(**changes))


@pytest.mark.parametrize("changes", [dict(as_of="2026-10-18T12:00:00"),
    dict(as_of="0001-01-01T00:00:00Z"), dict(windows=[]), dict(windows=["1h", "1h"]),
    dict(windows=["30d"])])
def test_invalid_request_contract(changes):
    with pytest.raises(ValidationError):
        AggregationRequest(results=[], **changes)


def test_existing_result_contracts_can_be_enriched_without_inference():
    vader = SentimentResult(document_id="v", model="vader", label="positive",
                            sentiment_score=0.6, tickers=["AAPL"])
    finbert = FinBertResult(document_id="f", label="neutral", sentiment_score=0.1,
        confidence=0.7, tickers=["AAPL"],
        probabilities=SentimentProbabilities(positive=0.2, negative=0.1, neutral=0.7))
    result = calculate([dict(vader.model_dump(), published_at=NOW),
                        dict(finbert.model_dump(), published_at=NOW)])
    by_model = {a.model: a for a in result.windows[0].aggregates}
    assert by_model["finbert"].label_counts.neutral == 1
    assert by_model["finbert"].mean_sentiment == 0.1
    assert by_model["vader"].mean_sentiment == 0.6


def test_order_independent_output_and_requested_window_order():
    rows = [row("3", tickers=["MSFT"]), row("1"), row("2", sentiment_score=-0.1)]
    forward = calculate(rows, windows=["7d", "1h"])
    reverse = calculate(list(reversed(rows)), windows=["7d", "1h"])
    assert forward == reverse
    assert [w.window for w in forward.windows] == ["7d", "1h"]


def test_default_reference_is_aware_utc_and_missing_date_stays_missing():
    parsed = AggregationRequest(results=[{k: v for k, v in row().items() if k != "published_at"}])
    assert parsed.as_of.utcoffset() == timedelta(0)
    assert parsed.results[0].published_at is None

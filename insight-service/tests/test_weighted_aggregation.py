from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.aggregation import aggregate_sentiments
from app.aggregation_models import AggregationRequest
from app.weighted_aggregation import (WeightedAggregationRequest, WeightingPolicy,
                                     aggregate_weighted, document_weight, WeightedSentimentResult)

NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)


def row(document_id="1", **changes):
    value = dict(document_id=document_id, model="finbert", tickers=["AAPL"],
                 sentiment_score=0.8, label="positive", published_at=NOW)
    value.update(changes)
    return value


def test_manually_computed_weighted_mean_and_all_label_distributions():
    # Weights: positive 2*(1+.1*10)=4; negative=1; neutral=1.
    # Weighted mean=(4*.8 + 1*(-.4) + 1*0)/6 = 7/15.
    request = WeightedAggregationRequest(as_of=NOW, results=[
        row(source=" Premium ", engagement=10),
        row("2", source="wire", sentiment_score=-0.4, label="negative"),
        row("3", sentiment_score=0, label="neutral")],
        weighting=dict(source_weights={"PREMIUM": 2}, engagement_factor=0.1))
    before = request.model_dump()
    result = aggregate_weighted(request)
    for window in result.windows:
        value = window.aggregates[0]
        assert value.article_count == value.effective_document_count == 3
        assert value.total_weight == 6
        assert value.mean_sentiment == pytest.approx(2 / 15)
        assert value.weighted_mean_sentiment == pytest.approx(7 / 15)
        assert value.label_counts.model_dump() == dict(positive=1, neutral=1, negative=1)
        assert value.weighted_distribution.positive == pytest.approx(2 / 3)
        assert value.weighted_distribution.negative == pytest.approx(1 / 6)
        assert value.weighted_distribution.neutral == pytest.approx(1 / 6)
    assert request.model_dump() == before


def test_defaults_exactly_preserve_existing_statistics():
    rows = [row(), row("2", sentiment_score=-0.4, label="negative", source="unknown", engagement=100)]
    original = aggregate_sentiments(AggregationRequest(as_of=NOW, results=rows))
    weighted = aggregate_weighted(WeightedAggregationRequest(as_of=NOW, results=rows))
    assert original.diagnostics == weighted.diagnostics
    for old, new in zip(original.windows, weighted.windows):
        assert old.start == new.start and old.end == new.end
        for a, b in zip(old.aggregates, new.aggregates):
            assert a.mean_sentiment == pytest.approx(b.weighted_mean_sentiment)
            assert a.distribution == b.weighted_distribution
            assert b.total_weight == a.article_count


def test_missing_source_engagement_and_capped_large_engagement():
    policy = WeightingPolicy(default_source_weight=2, engagement_factor=0.1, engagement_cap=10)
    assert document_weight(WeightedSentimentResult(**row()), policy) == 2
    assert document_weight(WeightedSentimentResult(**row(source="unknown", engagement=0)), policy) == 2
    assert document_weight(WeightedSentimentResult(**row(engagement=1e12)), policy) == 4


def test_zero_weight_produces_null_weighted_statistics_but_keeps_raw_counts():
    request = WeightedAggregationRequest(as_of=NOW, results=[row()],
        weighting=dict(default_source_weight=0))
    value = aggregate_weighted(request).windows[0].aggregates[0]
    assert value.article_count == 1 and value.effective_document_count == 0
    assert value.total_weight == 0
    assert value.weighted_mean_sentiment is None and value.weighted_distribution is None
    assert value.mean_sentiment == 0.8


def test_zero_weight_article_has_no_weighted_influence():
    result = aggregate_weighted(WeightedAggregationRequest(as_of=NOW, results=[
        row(source="blocked"), row("2", sentiment_score=-0.4, label="negative")],
        weighting=dict(source_weights={"blocked": 0})))
    value = result.windows[0].aggregates[0]
    assert value.article_count == 2 and value.effective_document_count == 1
    assert value.weighted_mean_sentiment == -0.4
    assert value.weighted_distribution.negative == 1


def test_duplicates_multi_ticker_and_models_use_existing_semantics():
    rows = [row(tickers=["AAPL", "MSFT", "aapl"], source="wire"),
            row(tickers=["MSFT", "AAPL"], source=" WIRE "), row(model="vader")]
    result = aggregate_weighted(WeightedAggregationRequest(as_of=NOW, results=rows))
    assert result.diagnostics.duplicates_removed == 1
    assert [(r.ticker, r.model, r.article_count) for r in result.windows[0].aggregates] == [
        ("AAPL", "finbert", 1), ("AAPL", "vader", 1), ("MSFT", "finbert", 1)]


@pytest.mark.parametrize("changes", [dict(source="different"), dict(engagement=1)])
def test_metadata_conflicts_are_rejected(changes):
    with pytest.raises(ValidationError, match="Conflicting duplicate"):
        WeightedAggregationRequest(as_of=NOW, results=[row(), row(**changes)])


@pytest.mark.parametrize("changes", [dict(engagement=-1), dict(engagement=float("inf")),
    dict(engagement=float("nan")), dict(engagement=1e13)])
def test_invalid_engagement(changes):
    with pytest.raises(ValidationError):
        WeightedSentimentResult(**row(**changes))


@pytest.mark.parametrize("changes", [dict(engagement_factor=-1), dict(engagement_cap=-1),
    dict(default_source_weight=-1), dict(source_weights={"wire": float("inf")}),
    dict(source_weights={" ": 1}), dict(source_weights={"wire": 1, " WIRE ": 2})])
def test_invalid_policy(changes):
    with pytest.raises(ValidationError):
        WeightingPolicy(**changes)


def test_exclusion_diagnostics_empty_windows_and_utc_offsets_are_reused():
    rows = [row("missing", published_at=None), row("future", published_at=NOW + timedelta(seconds=1)),
            row("old", published_at=NOW - timedelta(days=8)), row("no-ticker", tickers=[]),
            row("offset", published_at=NOW.astimezone(timezone(timedelta(hours=2))))]
    request = WeightedAggregationRequest(as_of=NOW, results=rows)
    result = aggregate_weighted(request)
    assert result.diagnostics.model_dump() == dict(input_count=5, unique_count=5,
        duplicates_removed=0, missing_timestamp_count=1, future_timestamp_count=1,
        missing_tickers_count=1, outside_window_count=1)
    assert all(w.aggregates[0].article_count == 1 for w in result.windows)
    empty = aggregate_weighted(WeightedAggregationRequest(as_of=NOW, results=[]))
    assert all(w.aggregates == [] for w in empty.windows)


@pytest.mark.parametrize("window,delta", [("1h", timedelta(hours=1)),
    ("24h", timedelta(hours=24)), ("7d", timedelta(days=7))])
def test_weighted_exact_window_boundaries(window, delta):
    result = aggregate_weighted(WeightedAggregationRequest(as_of=NOW, windows=[window],
        results=[row(), row("start", published_at=NOW-delta),
                 row("inside", published_at=NOW-delta+timedelta(microseconds=1))]))
    assert result.windows[0].aggregates[0].article_count == 2
    assert result.windows[0].aggregates[0].total_weight == 2


@pytest.mark.parametrize("score,label", [(1, "positive"), (-1, "negative")])
def test_endpoint_scores_remain_bounded_with_unequal_weights(score, label):
    rows = [row(str(i), sentiment_score=score, label=label, engagement=i / 7)
            for i in range(100)]
    result = aggregate_weighted(WeightedAggregationRequest(as_of=NOW, results=rows,
        weighting=dict(engagement_factor=0.3)))
    assert result.windows[0].aggregates[0].weighted_mean_sentiment == score

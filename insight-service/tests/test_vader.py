import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.models.sentiment import AnalyzeRequest
from app.vader import analyze_documents, classify_compound


@pytest.mark.parametrize(
    "score,label",
    [(-1, "negative"), (-0.05, "negative"), (-0.0499, "neutral"),
     (0, "neutral"), (0.0499, "neutral"), (0.05, "positive"), (1, "positive")],
)
def test_classification_boundaries(score, label):
    assert classify_compound(score) == label


def test_real_vader_batch_scores_and_metadata():
    texts = ["Excellent profits and strong growth!", "The report was filed on Tuesday.",
             "Terrible losses and disastrous results."]
    request = AnalyzeRequest(documents=[
        {"document_id": str(i), "title": "Financial report", "text": text,
         "tickers": ["AAPL", "MSFT"]}
        for i, text in enumerate(texts)
    ])
    before = request.model_dump()
    response = analyze_documents(request)
    oracle = SentimentIntensityAnalyzer()
    assert response.schema_version == "1.0"
    assert [r.label for r in response.results] == ["positive", "neutral", "negative"]
    for i, result in enumerate(response.results):
        assert result.document_id == str(i)
        assert result.sentiment_score == oracle.polarity_scores(texts[i])["compound"]
        assert result.model == "vader"
        assert result.tickers == ["AAPL", "MSFT"]
        assert result.confidence is None
    response.results[0].tickers.append("OTHER")
    assert request.model_dump() == before


def test_body_is_scored_without_title_and_default_tickers():
    request = AnalyzeRequest(documents=[{
        "document_id": "1", "title": "Terrible disastrous losses",
        "text": "The report was filed on Tuesday.",
    }])
    result = analyze_documents(request).results[0]
    assert result.sentiment_score == 0
    assert result.label == "neutral"
    assert result.tickers == []


def test_finbert_is_not_silently_scored_as_vader():
    request = AnalyzeRequest(model="finbert", documents=[{
        "document_id": "1", "title": "Report", "text": "Strong profits",
    }])
    with pytest.raises(NotImplementedError, match="Only the vader"):
        analyze_documents(request)

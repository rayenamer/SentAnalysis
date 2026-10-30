"""VADER scoring using the existing sentiment contracts."""

from functools import lru_cache
from typing import Literal

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.models.sentiment import AnalyzeRequest, AnalyzeResponse, SentimentResult
from app.helper.metrics import measure_model


def classify_compound(score: float) -> Literal["positive", "neutral", "negative"]:
    """Apply VADER's inclusive positive and negative thresholds."""
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Load the bundled lexicon once; no model download is required."""
    return SentimentIntensityAnalyzer()


@measure_model("vader")
def analyze_documents(request: AnalyzeRequest) -> AnalyzeResponse:
    """Score document bodies in input order, preserving IDs and tickers.

    Confidence remains None: VADER does not produce calibrated probabilities.
    Language metadata is preserved by the input contract but no translation is
    performed; this baseline uses VADER's English lexicon.
    """
    if request.model != "vader":
        raise NotImplementedError("Only the vader model is implemented")

    analyzer = get_analyzer()
    results = []
    for document in request.documents:
        compound = analyzer.polarity_scores(document.text)["compound"]
        results.append(
            SentimentResult(
                document_id=document.document_id,
                model="vader",
                label=classify_compound(compound),
                sentiment_score=compound,
                confidence=None,
                tickers=list(document.tickers),
            )
        )
    return AnalyzeResponse(results=results)

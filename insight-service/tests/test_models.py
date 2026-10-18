
import pytest
from pydantic import ValidationError

from app.models.sentiment import (
    AnalyzeRequest,
    SentimentResult,
)


def test_valid_request():
    request = AnalyzeRequest(
        documents=[
            {
                "document_id": "doc-001",
                "title": "Apple earnings",
                "text": "Apple reports strong profits",
                "tickers": ["AAPL"],
            }
        ],
        model="vader",
    )

    assert request.documents[0].tickers == ["AAPL"]
    assert request.model == "vader"


def test_empty_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            documents=[
                {
                    "document_id": "doc-001",
                    "title": "Apple earnings",
                    "text": "",
                }
            ]
        )


def test_invalid_model():
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            documents=[
                {
                    "document_id": "doc-001",
                    "title": "Apple earnings",
                    "text": "Strong profits",
                }
            ],
            model="unknown_model",
        )


def test_invalid_sentiment_score():
    with pytest.raises(ValidationError):
        SentimentResult(
            document_id="doc-001",
            model="vader",
            label="positive",
            sentiment_score=1.5,
            tickers=["AAPL"],
        )


def test_empty_documents():
    with pytest.raises(ValidationError):
        AnalyzeRequest(documents=[])

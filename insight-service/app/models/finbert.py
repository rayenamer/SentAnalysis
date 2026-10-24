"""Additive FinBERT contracts; the original contracts remain valid."""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.sentiment import AnalyzeResponse, SentimentResult


class SentimentProbabilities(BaseModel):
    positive: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)


class FinBertResult(SentimentResult):
    model: Literal["finbert"] = "finbert"
    probabilities: SentimentProbabilities


class FinBertResponse(AnalyzeResponse):
    results: list[FinBertResult]

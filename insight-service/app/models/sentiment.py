
from typing import Literal

from pydantic import BaseModel, Field


class FinancialDocument(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = "en"
    tickers: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    documents: list[FinancialDocument] = Field(
        min_length=1,
        max_length=100
    )
    model: Literal["vader", "finbert"] = "vader"


class SentimentResult(BaseModel):
    document_id: str
    model: str
    label: Literal["positive", "neutral", "negative"]
    sentiment_score: float = Field(ge=-1, le=1)
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1
    )
    tickers: list[str]


class AnalyzeResponse(BaseModel):
    schema_version: str = "1.0"
    results: list[SentimentResult]

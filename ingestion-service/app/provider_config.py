"""Opt-in provider configuration. Credentials never participate in repr/output."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    timeout_seconds: float = Field(default=10, gt=0, le=60)


class SearchConfig(ProviderConfig):
    keywords: list[str] = Field(default_factory=list, max_length=20)
    tickers: list[str] = Field(default_factory=list, max_length=20)
    max_records: int = Field(default=50, ge=1, le=100)

    @field_validator("keywords", "tickers")
    @classmethod
    def validate_terms(cls, values):
        if any(not value.strip() or len(value) > 80 or re.search(r'["\r\n\\]', value) for value in values):
            raise ValueError("search terms must contain 1–80 characters without quotes or newlines")
        return list(dict.fromkeys(value.strip() for value in values))

    @model_validator(mode="after")
    def validate_query(self):
        if self.enabled and (not self.query or len(self.query) > 500):
            raise ValueError("enabled search requires a query of at most 500 characters")
        return self

    @property
    def query(self):
        return " OR ".join('"' + term + '"' for term in self.keywords + self.tickers)


class NewsAPIConfig(SearchConfig):
    api_key: SecretStr | None = Field(default=None, repr=False)
    language: Literal["ar", "de", "en", "es", "fr", "he", "it", "nl", "no", "pt", "ru", "sv", "ud", "zh"] | None = None


class GDELTConfig(SearchConfig):
    timespan: str = "24h"

    @field_validator("timespan")
    @classmethod
    def validate_timespan(cls, value):
        if not re.fullmatch(r"(?:[1-9]|[1-9][0-9])(?:h|d)", value):
            raise ValueError("timespan must be 1–99 hours or days")
        return value


class RedditConfig(ProviderConfig):
    client_id: SecretStr | None = Field(default=None, repr=False)
    client_secret: SecretStr | None = Field(default=None, repr=False)
    user_agent: str = Field(default="SentAnalysis/0.2 (configured operator required)", min_length=10, max_length=200)
    subreddits: list[str] = Field(default_factory=lambda: ["stocks", "investing", "wallstreetbets", "cryptocurrency"], min_length=1, max_length=10)
    page_size: int = Field(default=25, ge=1, le=100)
    max_pages: int = Field(default=2, ge=1, le=5)

    @field_validator("user_agent")
    @classmethod
    def header_safe(cls, value):
        if any(ord(c) < 32 or ord(c) > 126 for c in value):
            raise ValueError("user agent must be printable ASCII")
        return value

    @field_validator("subreddits")
    @classmethod
    def valid_subreddits(cls, values):
        if any(not re.fullmatch(r"[A-Za-z0-9_]{2,30}", value) for value in values):
            raise ValueError("invalid subreddit name")
        return list(dict.fromkeys(value.lower() for value in values))


class Instrument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=30)
    provider_symbol: str = Field(min_length=1, max_length=30)
    asset_type: Literal["stock", "etf", "crypto", "index"] = "stock"

    @field_validator("ticker", "provider_symbol")
    @classmethod
    def valid_symbol(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9^][A-Za-z0-9.^/_-]{0,29}", value):
            raise ValueError("invalid symbol")
        return value.upper()


class MarketConfig(ProviderConfig):
    api_key: SecretStr | None = Field(default=None, repr=False)
    instruments: list[Instrument] = Field(default_factory=list, max_length=8)
    # Intraday only: Twelve Data ignores timezone=UTC for daily bars.
    interval: Literal["1min", "5min", "15min", "30min", "1h"] = "1h"
    outputsize: int = Field(default=100, ge=1, le=1000)

    @field_validator("instruments")
    @classmethod
    def unique_instruments(cls, values):
        if len({v.ticker for v in values}) != len(values):
            raise ValueError("instrument tickers must be unique")
        return values

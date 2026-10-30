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



class HackerNewsConfig(ProviderConfig):
    """Public Algolia search: one bounded query per configured keyword."""
    keywords: list[str] = Field(default_factory=lambda: ["Apple", "Microsoft", "Tesla", "Nvidia", "Bitcoin"], min_length=1, max_length=10)
    hits_per_page: int = Field(default=10, ge=1, le=100)
    max_pages: int = Field(default=1, ge=1, le=5)
    max_records: int = Field(default=50, ge=1, le=1000)

    @field_validator("keywords")
    @classmethod
    def valid_keywords(cls, values):
        seen, result = set(), []
        for value in values:
            value = value.strip()
            if not value or len(value) > 80 or any(ord(c) < 32 for c in value):
                raise ValueError("keywords must contain 1–80 characters without control characters")
            if value.casefold() not in seen:
                result.append(value)
                seen.add(value.casefold())
        return result


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

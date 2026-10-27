"""Validated settings; loaded lazily from INGESTION_* environment variables."""
import json
import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.provider_config import NewsAPIConfig, GDELTConfig, RedditConfig, MarketConfig


class FeedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    url: str

    @field_validator("url")
    @classmethod
    def valid_url(cls, value):
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("feed URL must be HTTP(S), without credentials")
        _ = parts.port
        return value


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sources: list[FeedSource] = Field(default_factory=list, max_length=20)
    timeout_seconds: float = Field(default=10, gt=0, le=60)
    max_response_bytes: int = Field(default=2_000_000, ge=1024, le=10_000_000)
    max_documents: int = Field(default=100, ge=1, le=100)
    max_entries_per_source: int = Field(default=200, ge=1, le=1000)
    company_tickers: dict[str, str] = Field(default_factory=dict)
    newsapi: NewsAPIConfig = Field(default_factory=NewsAPIConfig)
    gdelt: GDELTConfig = Field(default_factory=GDELTConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)

    @field_validator("sources")
    @classmethod
    def unique_sources(cls, values):
        if len({s.name for s in values}) != len(values):
            raise ValueError("source names must be unique")
        return values

    @field_validator("company_tickers")
    @classmethod
    def valid_tickers(cls, values):
        import re
        if any(not name.strip() or not re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:[.-][A-Z0-9]{1,3})?", ticker.upper()) for name, ticker in values.items()):
            raise ValueError("company mappings require nonempty names and valid tickers")
        return {name.strip(): ticker.upper() for name, ticker in values.items()}

    @classmethod
    def from_env(cls):
        values = {}
        for name in cls.model_fields:
            if name in {"newsapi", "gdelt", "reddit", "market"}:
                provider = cls.model_fields[name].annotation
                nested = {}
                for field in provider.model_fields:
                    raw = os.getenv(f"INGESTION_{name.upper()}_{field.upper()}")
                    if raw is not None:
                        nested[field] = json.loads(raw) if field in {"keywords", "tickers", "subreddits", "instruments"} else raw
                values[name] = nested
                continue
            raw = os.getenv("INGESTION_" + name.upper())
            if raw is not None:
                values[name] = json.loads(raw) if name in {"sources", "company_tickers"} else raw
        return cls.model_validate(values)

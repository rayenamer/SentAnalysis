"""Insight communication and scheduling are explicit, bounded opt-ins."""
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

Provider = Literal["rss", "newsapi", "gdelt", "market", "hackernews"]


class InsightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    service_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: float = Field(default=15, gt=0, le=60)
    max_retries: int = Field(default=1, ge=0, le=3)
    retry_backoff_seconds: float = Field(default=0.25, ge=0, le=5)
    max_response_bytes: int = Field(default=2_000_000, ge=1024, le=10_000_000)

    @field_validator("service_url")
    @classmethod
    def validate_url(cls, value):
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment):
            raise ValueError("Insight URL must be HTTP(S) without credentials, query or fragment")
        _ = parts.port
        return value.rstrip("/")


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    providers: list[Provider] = Field(default_factory=lambda: ["rss"], min_length=1, max_length=5)
    rss_interval_seconds: float = Field(default=300, ge=60, le=86400)
    newsapi_interval_seconds: float = Field(default=900, ge=60, le=86400)
    gdelt_interval_seconds: float = Field(default=900, ge=60, le=86400)
    market_interval_seconds: float = Field(default=900, ge=60, le=86400)
    hackernews_interval_seconds: float = Field(default=900, ge=60, le=86400)
    analyze_documents: bool = False
    analysis_model: Literal["vader", "finbert"] = "vader"
    analysis_chunk_size: int = Field(default=4, ge=1, le=20)
    shutdown_timeout_seconds: float = Field(default=5, gt=0, le=30)

    @field_validator("providers")
    @classmethod
    def unique_providers(cls, values):
        return list(dict.fromkeys(values))

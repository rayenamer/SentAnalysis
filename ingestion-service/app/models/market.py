"""OHLCV contracts, deliberately independent of sentiment documents."""
from datetime import timezone
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class MarketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbols: list[str] | None = Field(default=None, min_length=1, max_length=8)
    max_candles: int | None = Field(default=None, ge=1, le=1000, strict=True)

    @field_validator("symbols")
    @classmethod
    def normalized_symbols(cls, values):
        return list(dict.fromkeys(value.upper() for value in values)) if values else values


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ticker: str
    provider_symbol: str
    asset_type: Literal["stock", "etf", "crypto", "index"]
    timestamp: AwareDatetime
    close_timestamp: AwareDatetime
    timestamp_semantics: Literal["bar_open"] = "bar_open"
    interval: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def no_booleans(cls, value):
        if isinstance(value, bool):
            raise ValueError("booleans are not prices or volume")
        return value

    @field_validator("timestamp", "close_timestamp")
    @classmethod
    def utc(cls, value):
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def valid_ohlc(self):
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close) or self.low > self.high:
            raise ValueError("inconsistent OHLC range")
        if self.close_timestamp <= self.timestamp:
            raise ValueError("close timestamp must follow bar open")
        return self

    def insight_price(self):
        """Projection only; callers must select a compatible bucket/alignment policy."""
        return {"ticker": self.ticker, "timestamp": self.close_timestamp.isoformat(), "close_price": self.close}


class MarketDiagnostics(BaseModel):
    collected: int = 0
    rejected: int = 0
    deduplicated: int = 0
    conflicting_timestamps: int = 0
    incomplete: int = 0
    missing_volume: int = 0
    gaps: int = 0
    limited: int = 0
    returned: int = 0


class MarketSourceResult(BaseModel):
    ticker: str
    provider_symbol: str
    status: Literal["success", "error"]
    error_code: str | None = None
    diagnostics: MarketDiagnostics = Field(default_factory=MarketDiagnostics)


class MarketResponse(BaseModel):
    schema_version: str = "1.0"
    provider: Literal["twelvedata"] = "twelvedata"
    collected_at: AwareDatetime
    candles: list[Candle]
    sources: list[MarketSourceResult]
    diagnostics: MarketDiagnostics

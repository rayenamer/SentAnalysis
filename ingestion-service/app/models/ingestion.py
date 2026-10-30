from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_names: list[str] | None = Field(default=None, min_length=1, max_length=20)
    max_documents: int | None = Field(default=None, ge=1, le=100, strict=True)
    providers: list[Literal["rss", "newsapi", "gdelt", "hackernews"]] = Field(default_factory=lambda: ["rss"], min_length=1, max_length=4)


class Document(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str
    tickers: list[str]
    source: str
    original_url: str | None
    canonical_url: str | None
    collected_at: datetime
    published_at: datetime | None
    metadata: dict = Field(default_factory=dict)


class SourceResult(BaseModel):
    name: str
    url: str
    status: Literal["success", "error"]
    collected: int = 0
    entries_truncated: bool = False
    error_code: str | None = None
    retry_after_seconds: float | None = None


class Diagnostics(BaseModel):
    collected: int = 0
    rejected: int = 0
    deduplicated: int = 0
    returned: int = 0
    limited: int = 0
    published_at_missing_or_invalid: int = 0


class IngestResponse(BaseModel):
    schema_version: str = "1.0"
    collected_at: datetime
    documents: list[Document]
    sources: list[SourceResult]
    diagnostics: Diagnostics

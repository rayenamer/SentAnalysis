"""Existing contracts relocated without validation changes."""

from typing import Literal
from pydantic import BaseModel, Field, model_validator
from app.models.finbert import FinBertResult
from app.models.sentiment import AnalyzeRequest, SentimentResult


class BatchAnalyzeRequest(AnalyzeRequest):
    chunk_size: int = Field(default=4, ge=1, le=20, strict=True)



class DocumentError(BaseModel):
    code: Literal["inference_failed", "inference_contract_error"]
    message: str



class DocumentOutcome(BaseModel):
    input_index: int = Field(ge=0)
    document_id: str
    tickers: list[str]
    status: Literal["success", "error"]
    result: FinBertResult | SentimentResult | None = None
    error: DocumentError | None = None

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "success":
            if self.result is None or self.error is not None:
                raise ValueError("success requires a result and no error")
            if self.result.document_id != self.document_id or self.result.tickers != self.tickers:
                raise ValueError("result identity must match outcome")
        elif self.error is None or self.result is not None:
            raise ValueError("error requires error details and no result")
        return self



class BatchAnalyzeResponse(BaseModel):
    schema_version: str = "1.0"
    model: Literal["vader", "finbert", "loughran_mcdonald"]
    outcomes: list[DocumentOutcome]
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)


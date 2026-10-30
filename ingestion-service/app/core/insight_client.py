"""Dedicated stateless scoring client. Never forward upstream error messages."""
import asyncio
from typing import get_args

from pydantic import ValidationError
from app.helper.metrics import record

from app.helper.provider_http import SourceError, fetch_json
from app.models.insight import AnalysisModel, AnalysisOutcome, AnalysisReport, WireBatch


class InsightClient:
    def __init__(self, config, http_client):
        self.config, self.http_client = config, http_client

    async def _batch(self, documents, model, chunk_size):
        config = self.config
        # Scoring is stateless/read-only despite POST. Never use this retry policy
        # for persistence or publishing. Invalid requests/contracts are terminal.
        body = {"documents": [{name: getattr(doc, name) for name in
                                ("document_id", "title", "text", "language", "tickers")} for doc in documents],
                "model": model, "chunk_size": chunk_size}
        for attempt in range(config.max_retries + 1):
            try:
                payload = await fetch_json(self.http_client, "POST", config.service_url + "/analyze/batch",
                                           timeout=config.request_timeout_seconds, max_bytes=config.max_response_bytes,
                                           json=body)
                break
            except SourceError as exc:
                transient = exc.retryable or exc.status_code in {429, 500, 502, 503, 504}
                if not transient or attempt == config.max_retries:
                    raise
                delay = min(5, config.retry_backoff_seconds * 2 ** attempt)
                if exc.retry_after_seconds is not None:
                    if exc.retry_after_seconds > 5:
                        # Do not shorten a server-directed pause or hang the API.
                        raise
                    delay = max(delay, exc.retry_after_seconds)
                await asyncio.sleep(delay)
        try:
            batch = WireBatch.model_validate(payload)
            if batch.model != model or len(batch.outcomes) != len(documents):
                raise ValueError("batch shape mismatch")
            for index, (outcome, doc) in enumerate(zip(batch.outcomes, documents)):
                if (outcome.input_index != index or outcome.document_id != doc.document_id
                        or outcome.tickers != doc.tickers or (outcome.result and outcome.result.model != model)):
                    raise ValueError("batch identity mismatch")
            return batch
        except (ValidationError, ValueError) as exc:
            raise SourceError("invalid_response") from exc

    async def analyze(self, documents, model="vader", chunk_size=4):
        if model not in get_args(AnalysisModel) or type(chunk_size) is not int or not 1 <= chunk_size <= 20:
            raise ValueError("invalid_analysis_options")
        outcomes = []
        for start in range(0, len(documents), 100):
            chunk = documents[start:start + 100]
            try:
                batch = await self._batch(chunk, model, chunk_size)
            except SourceError as exc:
                record("insight_transfers_total", "failure")
                for offset, doc in enumerate(chunk):
                    outcomes.append(AnalysisOutcome(input_index=start + offset, document_id=doc.document_id,
                                                    tickers=doc.tickers, status="error", error_code=exc.code))
                continue
            record("insight_transfers_total", "failure" if batch.error_count else "success")
            for outcome in batch.outcomes:
                outcomes.append(AnalysisOutcome(input_index=start + outcome.input_index, document_id=outcome.document_id,
                                                tickers=outcome.tickers, status=outcome.status, result=outcome.result,
                                                error_code=outcome.error.code if outcome.error else None))
        successes = sum(outcome.status == "success" for outcome in outcomes)
        errors = len(outcomes) - successes
        status = "empty" if not outcomes else "partial" if successes and errors else "error" if errors else "success"
        return AnalysisReport(status=status, model=model, outcomes=outcomes, success_count=successes, error_count=errors)

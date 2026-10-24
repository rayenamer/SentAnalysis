"""Additive batch outcomes, reusing existing scorers without changing their APIs.

Invalid input is rejected as a whole by the existing Pydantic contract. Runtime
chunk failures are retried once per document with the SAME model, serially.
Contract violations are never guessed around: the affected chunk gets errors.
Input indices disambiguate repeated document IDs. No partial result disappears.
Known initialization failure is terminal for this request: remaining documents
get explicit errors without further model calls. The failure is not cached
across requests, so the next request may retry initialization once. Ordinary
document-level failures retain the existing same-model individual retry policy.
"""

from app.models.batch import (BatchAnalyzeRequest, DocumentError, DocumentOutcome, BatchAnalyzeResponse)


from app.models.finbert import FinBertResult
from app.helper.inference_errors import ModelInitializationError
from app.models.sentiment import AnalyzeRequest, SentimentResult


class InferenceContractError(ValueError):
    pass


def _run_model(request: AnalyzeRequest):
    if request.model == "vader":
        from app.core.sentiment.lexicon import analyze_documents
        return analyze_documents(request)
    from app.core.sentiment.finbert import get_finbert_service
    try:
        service = get_finbert_service()
    except Exception as exc:
        raise ModelInitializationError("FinBERT initialization failed") from exc
    return service.analyze(request)


def _checked_results(response, request):
    try:
        results = response.results
        if len(results) != len(request.documents):
            raise ValueError("wrong result count")
        checked = []
        for document, result in zip(request.documents, results):
            contract = FinBertResult if request.model == "finbert" else SentimentResult
            result = contract.model_validate(result.model_dump())
            if (result.document_id != document.document_id or result.tickers != document.tickers
                    or result.model != request.model):
                raise ValueError("result order or identity mismatch")
            checked.append(result)
        return checked
    except Exception as exc:
        raise InferenceContractError("Inference returned invalid or misaligned results") from exc


def analyze_batch(request: BatchAnalyzeRequest, *, runner=None) -> BatchAnalyzeResponse:
    runner = runner or _run_model
    outcomes = []
    initialization_failure = None

    def attempt(documents, start):
        nonlocal initialization_failure
        if initialization_failure is not None:
            for offset, document in enumerate(documents):
                outcomes.append(DocumentOutcome(input_index=start + offset,
                    document_id=document.document_id, tickers=list(document.tickers),
                    status="error", error=initialization_failure))
            return
        incoming = AnalyzeRequest(documents=documents, model=request.model)
        try:
            response = runner(incoming)
        except ModelInitializationError:
            failure = DocumentError(code="inference_failed", message="Model initialization failed")
            initialization_failure = failure
        except Exception:
            if len(documents) > 1:
                for offset, document in enumerate(documents):
                    attempt([document], start + offset)
                return
            failure = DocumentError(code="inference_failed", message="Document inference failed")
        else:
            try:
                results = _checked_results(response, incoming)
            except InferenceContractError:
                failure = DocumentError(code="inference_contract_error",
                                        message="Inference returned invalid or misaligned results")
            else:
                for offset, (document, result) in enumerate(zip(documents, results)):
                    outcomes.append(DocumentOutcome(input_index=start + offset,
                        document_id=document.document_id, tickers=list(document.tickers),
                        status="success", result=result))
                return
        for offset, document in enumerate(documents):
            outcomes.append(DocumentOutcome(input_index=start + offset,
                document_id=document.document_id, tickers=list(document.tickers),
                status="error", error=failure))

    for start in range(0, len(request.documents), request.chunk_size):
        attempt(request.documents[start:start + request.chunk_size], start)
    successes = sum(outcome.status == "success" for outcome in outcomes)
    return BatchAnalyzeResponse(model=request.model, outcomes=outcomes,
                                success_count=successes, error_count=len(outcomes) - successes)

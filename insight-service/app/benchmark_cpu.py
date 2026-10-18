"""Explicit CPU benchmark; reports to stdout and writes no result files.

Supply JSON with at least 20 real financial documents (AnalyzeRequest shape).
FinBERT requires --enable-real-finbert and --onnx-dir containing an existing
export, so the benchmark never exports or downloads model weights. Inputs of
size 1/5/10/20 use the first N supplied documents. Loading and one warm-up are
reported separately from steady-state measurements, including tokenization.
Memory is approximate process working set and process-lifetime peak, not an
isolated per-batch model allocation. Use separate processes to compare peaks.
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from app.models.sentiment import AnalyzeRequest

BATCH_SIZES = (1, 5, 10, 20)


def process_memory():
    if os.name != "nt":
        return {"working_set_mib": None, "process_lifetime_peak_mib": None}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_mib": counters.WorkingSetSize / 1024 ** 2,
            "process_lifetime_peak_mib": counters.PeakWorkingSetSize / 1024 ** 2}


def _verify(response, request):
    if len(response.results) != len(request.documents) or any(
        result.document_id != doc.document_id or result.tickers != doc.tickers
        or result.model != request.model
        for result, doc in zip(response.results, request.documents)
    ):
        raise ValueError("Benchmark inference lost or misaligned documents")


def benchmark(analyze, request: AnalyzeRequest, *, repeats=3, timer=perf_counter,
              memory=process_memory):
    if len(request.documents) < 20 or repeats < 1:
        raise ValueError("At least 20 documents and one repeat are required")
    warmup = AnalyzeRequest(documents=request.documents[:1], model=request.model)
    start = timer(); _verify(analyze(warmup), warmup); load_warmup = timer() - start
    measurements = []
    for size in BATCH_SIZES:
        incoming = AnalyzeRequest(documents=request.documents[:size], model=request.model)
        before = memory()
        durations = []
        for _ in range(repeats):
            start = timer(); response = analyze(incoming); elapsed = timer() - start
            _verify(response, incoming)
            durations.append(elapsed)
        total = sum(durations)
        measurements.append({"documents": size, "repeats": repeats,
            "total_processing_seconds": total, "mean_batch_seconds": total / repeats,
            "average_seconds_per_document": total / (size * repeats),
            "documents_per_second": size * repeats / total if total > 0 else None,
            "memory_before": before, "memory_after": memory()})
    return {"model": request.model, "load_and_warmup_seconds": load_warmup,
            "memory_scope": "whole process; peak is cumulative across its lifetime",
            "measurements": measurements}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-json", type=Path, required=True)
    parser.add_argument("--model", choices=["vader", "finbert"], default="vader")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--enable-real-finbert", action="store_true")
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--batch-size", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--max-length", type=int, choices=range(2, 513), default=256)
    args = parser.parse_args(argv)
    if args.model == "finbert" and (not args.enable_real_finbert or args.onnx_dir is None):
        parser.error("FinBERT requires --enable-real-finbert and --onnx-dir (existing local export)")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    data = json.loads(args.documents_json.read_text(encoding="utf-8"))
    data["model"] = args.model
    request = AnalyzeRequest.model_validate(data)
    if len(request.documents) < 20:
        parser.error("--documents-json must contain at least 20 documents")
    if args.model == "finbert":
        if not args.onnx_dir.is_dir():
            parser.error("--onnx-dir must exist")
        from app.finbert import FinBertService, FinBertSettings
        service = FinBertService(FinBertSettings(onnx_dir=args.onnx_dir,
            batch_size=args.batch_size, max_length=args.max_length))
        analyze = service.analyze
    else:
        from app.vader import analyze_documents
        analyze = analyze_documents
    report = benchmark(analyze, request, repeats=args.repeats)
    report["settings"] = {"inference_batch_size": args.batch_size if args.model == "finbert" else None,
                          "max_length": args.max_length if args.model == "finbert" else None}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

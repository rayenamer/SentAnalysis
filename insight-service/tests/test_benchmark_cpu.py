from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.benchmark_cpu import benchmark, main, process_memory
from app.models.sentiment import AnalyzeRequest


def request(count=20):
    return AnalyzeRequest(documents=[dict(document_id=str(i), title="Results", text="Report", tickers=[])
                                     for i in range(count)])


def inference(incoming):
    return SimpleNamespace(results=[SimpleNamespace(document_id=d.document_id,
        tickers=d.tickers, model=incoming.model) for d in incoming.documents])


def test_benchmark_metrics_use_measured_duration_and_expected_sizes():
    # Every measured interval is two seconds; warmup is separate.
    timer = Mock(side_effect=range(0, 100, 2))
    memory = Mock(return_value=dict(working_set_mib=100, process_lifetime_peak_mib=110))
    analyze = Mock(side_effect=inference)
    report = benchmark(analyze, request(), repeats=2, timer=timer, memory=memory)
    assert report["load_and_warmup_seconds"] == 2
    assert [r["documents"] for r in report["measurements"]] == [1, 5, 10, 20]
    assert [len(c.args[0].documents) for c in analyze.call_args_list] == [1, 1, 1, 5, 5, 10, 10, 20, 20]
    for value in report["measurements"]:
        size = value["documents"]
        assert value["total_processing_seconds"] == 4
        assert value["mean_batch_seconds"] == 2
        assert value["average_seconds_per_document"] == 2 / size
        assert value["documents_per_second"] == size / 2
        assert value["memory_after"]["working_set_mib"] == 100


def test_benchmark_rejects_lost_results():
    with pytest.raises(ValueError, match="lost or misaligned"):
        benchmark(lambda request: SimpleNamespace(results=[]), request())


@pytest.mark.parametrize("count,repeats", [(19, 1), (20, 0)])
def test_benchmark_invalid_input(count, repeats):
    with pytest.raises(ValueError):
        benchmark(Mock(), request(count), repeats=repeats)


def test_real_finbert_benchmark_requires_explicit_flag_before_opening_files():
    with pytest.raises(SystemExit) as exc:
        main(["--model", "finbert", "--documents-json", "does-not-exist.json"])
    assert exc.value.code == 2


def test_platform_memory_reader_is_safe():
    values = process_memory()
    assert set(values) == {"working_set_mib", "process_lifetime_peak_mib"}
    assert all(value is None or value > 0 for value in values.values())

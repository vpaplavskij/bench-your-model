"""Perf harness tests: median-run diagnosis (A2), thinking-aware prefill (A3)."""
from bench.client import ChatResult, StreamMetrics, StreamSample
from bench.perf import PerfPoint, PerfRun, _time_request, diagnose


class FakePerfClient:
    def __init__(self, res):
        self.res = res

    def chat(self, messages, **kw):
        return self.res


def test_diagnose_uses_the_median_run():
    # The fastest run alone would read DECODE-DOMINATED, the slowest PREFILL;
    # only the median run gives MIXED — proves the median is what's used.
    point = PerfPoint(prompt_size=1000)
    for first_ms in (10, 50, 70):
        point.runs.append(PerfRun(ok=True, ttft_ms=first_ms,
                                  ttft_first_token_ms=first_ms, total_ms=100))
    diag = diagnose([point])
    assert diag[0].startswith("MIXED"), diag


def test_prefill_is_measured_even_without_content_tokens():
    # A thinking run that spends the whole cap on reasoning has no content TTFT,
    # but its prefill must still be measured from the first (reasoning) token.
    res = ChatResult(ok=True, content="", ttft_ms=None, ttft_first_token_ms=100.0,
                     total_ms=300.0, prompt_tokens=1000, completion_tokens=200,
                     stream_chunks=0, reasoning_chunks=200, raw_finish_reason="length",
                     all_stream_metrics=StreamMetrics([StreamSample(t=0.1),
                                                       StreamSample(t=0.3)]))
    run = _time_request(FakePerfClient(res), "prompt", 200)
    assert run.prefill_tps == 10000.0   # 1000 prompt tok / 0.1 s first token
    assert run.decode_tps is None       # no content window to measure
    assert run.decode_tps_all == 5.0    # (2 arrivals - 1) / 0.2 s thinking window


def test_reasoning_only_runs_are_reported_as_such():
    point = PerfPoint(prompt_size=1000)
    res = ChatResult(ok=True, content="", ttft_ms=None, ttft_first_token_ms=100.0,
                     total_ms=300.0, prompt_tokens=1000, stream_chunks=0,
                     reasoning_chunks=200, raw_finish_reason="length")
    point.runs.append(_time_request(FakePerfClient(res), "prompt", 200))
    diag = diagnose([point])
    assert any("spent the entire --output-tokens budget on reasoning" in d for d in diag)

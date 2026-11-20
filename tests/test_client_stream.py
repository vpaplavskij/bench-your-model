"""Streaming-client tests against the mock's Phase 0 modes (fixes A3, B1)."""
from bench.client import BenchClient
from conftest import MockServer

PROMPT = [{"role": "user", "content": "What is 17 * 23?"}]


def chat(endpoint, max_tokens=None, json_schema=None):
    c = BenchClient(endpoint=endpoint, model="mock-model")
    try:
        return c.chat(PROMPT, max_tokens=max_tokens, json_schema=json_schema)
    finally:
        c.close()


def test_default_stream_is_content_only(mock_server):
    # s1-style schema so the mock answers "391" instead of free-generating filler.
    res = chat(mock_server.endpoint, json_schema={"type": "object",
                                                  "properties": {"answer": {"type": "string"}}})
    assert res.ok
    assert "396" in res.content
    assert res.reasoning_chunks == 0
    assert res.ttft_ms is not None
    assert res.ttft_first_token_ms == res.ttft_ms  # no thinking: same first token


def test_reasoning_then_content():
    srv = MockServer(think_tokens=5)
    try:
        res = chat(srv.endpoint)
        assert res.ok and res.content
        assert res.reasoning_chunks == 5
        assert res.ttft_ms is not None            # content-only TTFT still recorded
        assert res.ttft_first_token_ms <= res.ttft_ms
        assert res.all_stream_metrics.n_tokens == res.reasoning_chunks + res.stream_chunks
    finally:
        srv.stop()


def test_reasoning_spends_the_whole_max_tokens_cap():
    srv = MockServer(think_tokens=300)
    try:
        res = chat(srv.endpoint, max_tokens=256)
        assert res.ok
        assert res.content == ""
        assert res.reasoning_chunks == 256
        assert res.raw_finish_reason == "length"
        assert res.ttft_ms is None                # never saw a content token
        assert res.ttft_first_token_ms is not None
    finally:
        srv.stop()


def test_mid_stream_sse_error_is_a_failed_request():
    srv = MockServer(sse_error_every=1)
    try:
        res = chat(srv.endpoint)
        assert not res.ok
        assert res.error.startswith("stream error:")
    finally:
        srv.stop()


def test_http_500_is_a_failed_request():
    srv = MockServer(http_error_every=1)
    try:
        res = chat(srv.endpoint)
        assert not res.ok
        assert res.error.startswith("HTTP 500:")
    finally:
        srv.stop()

"""OpenAI-compatible chat client with per-request timing and token accounting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class ChatResult:
    """One completed (or failed) request."""

    ok: bool
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw_finish_reason: str | None = None
    error: str | None = None

    # Timing
    ttft_ms: float | None = None          # time to first CONTENT token (streaming)
    ttft_first_token_ms: float | None = None  # time to first token incl. reasoning_content
    total_ms: float = 0.0                # wall time for the whole request
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    stream_chunks: int = 0
    stream_metrics: "StreamMetrics | None" = None

    @property
    def tokens_per_s(self) -> float | None:
        # Content chunks ≈ content tokens. `usage` completion_tokens includes
        # server-side thinking: tool-call-only answers must NOT report a speed
        # built from thinking tokens (AGENTS.md gotcha).
        n = self.stream_chunks
        if not n or not self.total_ms:
            return None
        return n / (self.total_ms / 1000.0)


@dataclass
class StreamSample:
    """One token-arrival sample for per-token latency analysis."""

    t: float          # seconds since request start


class StreamMetrics:
    """Per-token arrival analysis for a streaming response.

    ``samples`` is a list of :class:`StreamSample` (one per content chunk,
    arrival time in seconds since request start). OpenAI-style SSE deltas
    carry no token index, so each content chunk counts as one token sample.
    """

    def __init__(self, samples: list[StreamSample]) -> None:
        self.samples = samples

    @property
    def n_tokens(self) -> int:
        return len(self.samples)

    def inter_token_ms(self) -> float | None:
        """Mean ITL — mean gap between consecutive token arrivals."""
        if len(self.samples) < 2:
            return None
        gaps = [(b.t - a.t) * 1000.0 for a, b in zip(self.samples, self.samples[1:])]
        return sum(gaps) / len(gaps) if gaps else None

    def p95_itl_ms(self) -> float | None:
        """P95 inter-token latency — catches stalls/spikes the mean hides."""
        if len(self.samples) < 2:
            return None
        gaps = sorted((b.t - a.t) * 1000.0 for a, b in zip(self.samples, self.samples[1:]))
        idx = max(0, int(round(0.95 * (len(gaps) - 1))))
        return gaps[idx]

    def max_itl_ms(self) -> float | None:
        if len(self.samples) < 2:
            return None
        return max((b.t - a.t) * 1000.0 for a, b in zip(self.samples, self.samples[1:]))

    def stall_count_ms(self, threshold_ms: float = 500.0) -> int:
        """Number of inter-token gaps exceeding ``threshold_ms``."""
        if len(self.samples) < 2:
            return 0
        return sum(
            1 for a, b in zip(self.samples, self.samples[1:])
            if (b.t - a.t) * 1000.0 > threshold_ms
        )


def _merge_tool_call(index: dict[int, dict], tc: dict) -> None:
    """Merge one streamed ``delta.tool_calls`` fragment into the accumulator.

    OpenAI streams tool calls as partial fragments keyed by ``index``; the
    function name and JSON argument string can each arrive split across chunks.
    llama.cpp (and friends) typically send the whole call in a single chunk,
    which this handles identically.
    """
    idx = tc.get("index", 0) or 0
    slot = index.setdefault(
        idx, {"id": "", "name": "", "arguments_text": ""}
    )
    if tc.get("id"):
        slot["id"] = tc["id"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        slot["name"] += fn["name"]
    if fn.get("arguments"):
        slot["arguments_text"] += fn["arguments"]


def _finalize_tool_calls(index: dict[int, dict]) -> list[dict]:
    """Turn the accumulated fragments into ``[{"name", "arguments"}]``."""
    calls: list[dict] = []
    for idx in sorted(index):
        slot = index[idx]
        if not slot["name"] and not slot["arguments_text"]:
            continue
        arguments: dict = {}
        text = slot["arguments_text"].strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    arguments = parsed
            except (json.JSONDecodeError, ValueError):
                # Keep the raw string so a checker can still see what arrived.
                arguments = {"_raw_arguments": text}
        calls.append({"name": slot["name"], "arguments": arguments})
    return calls


class BenchClient:
    """Thin wrapper over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "none",
        timeout_s: float = 300.0,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        import httpx  # local import so --help works without deps installed

        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._http = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "BenchClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Models endpoint (server / model sanity check)
    # ------------------------------------------------------------------ #
    def list_models(self) -> list[str]:
        r = self._http.get("/models")
        r.raise_for_status()
        data = r.json()
        return [m.get("id", "") for m in data.get("data", [])]

    # ------------------------------------------------------------------ #
    # Chat completion (streaming, with response_format support)
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict],
        json_schema: dict | None = None,
        tools: list[dict] | None = None,
        extra_body: dict | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Send a streaming chat request and measure TTFT / total time / tokens.

        If ``json_schema`` is given, strict structured output is requested via
        ``response_format={"type": "json_schema", ...}`` (OpenAI-style).
        If ``tools`` is given (OpenAI function-calling format), native tool
        calling is requested and the model's calls are returned in
        ``ChatResult.tool_calls`` as ``[{"name": ..., "arguments": {...}}]``.
        ``max_tokens`` overrides the client default for this one call.
        """
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
            # Ask OpenAI-compatible streaming servers to append a final chunk
            # carrying `usage`, so prompt/completion token counts (and thus
            # prefill speed) are available. Servers that ignore the field are
            # unaffected; we fall back to chunk counting when usage is absent.
            "stream_options": {"include_usage": True},
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "bench_output", "strict": True, "schema": json_schema},
            }
        if tools:
            body["tools"] = tools
        if extra_body:
            body.update(extra_body)

        res = ChatResult(ok=False)
        samples: list[StreamSample] = []
        t_start = time.perf_counter()
        first_token_at: float | None = None
        first_content_at: float | None = None
        pieces: list[str] = []
        # Native tool-call deltas arrive as partial fragments that must be
        # merged by index (OpenAI streaming contract); llama.cpp sends the full
        # call in one chunk, which this handles trivially.
        tc_index: dict[int, dict] = {}

        try:
            with self._http.stream("POST", "/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    detail = r.read().decode("utf-8", "replace")[:500]
                    res.error = f"HTTP {r.status_code}: {detail}"
                    res.total_ms = (time.perf_counter() - t_start) * 1000
                    return res
                for line in r.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    now = time.perf_counter() - t_start
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    token = delta.get("content")
                    if token:
                        if first_content_at is None:
                            first_content_at = now
                        if first_token_at is None:
                            first_token_at = now
                        samples.append(StreamSample(t=now))
                        pieces.append(token)
                        res.stream_chunks += 1
                    elif delta.get("reasoning_content") and first_token_at is None:
                        # Thinking models stream reasoning_content first: it IS
                        # the first token. ttft_ms stays content-only (the decode
                        # window must exclude thinking), but ttft_first_token_ms
                        # records the real prefill end for prefill tok/s.
                        first_token_at = now
                    tc_deltas = delta.get("tool_calls") or []
                    if tc_deltas:
                        # A tool-call delta IS the first token; tool-only answers
                        # must not record ttft=None (seen on s3).
                        if first_content_at is None:
                            first_content_at = now
                        if first_token_at is None:
                            first_token_at = now
                    for tc in tc_deltas:
                        _merge_tool_call(tc_index, tc)
                    usage = chunk.get("usage")
                    if usage:
                        res.prompt_tokens = usage.get("prompt_tokens", res.prompt_tokens)
                        res.completion_tokens = usage.get("completion_tokens", res.completion_tokens)
                    fr = choice.get("finish_reason")
                    if fr:
                        res.raw_finish_reason = fr
        except Exception as e:  # network / timeout / server crash
            res.error = f"{type(e).__name__}: {e}"
            res.total_ms = (time.perf_counter() - t_start) * 1000
            return res

        res.ok = True
        # Fallback when the server omits `usage`: each content chunk ≈ one token,
        # so tokens_per_s / decode speed stay available instead of silently None.
        if res.completion_tokens is None:
            res.completion_tokens = len(pieces) or res.stream_chunks
        res.content = "".join(pieces)
        res.tool_calls = _finalize_tool_calls(tc_index)
        res.total_ms = (time.perf_counter() - t_start) * 1000
        if first_token_at is not None:
            res.ttft_first_token_ms = first_token_at * 1000
        if first_content_at is not None:
            res.ttft_ms = first_content_at * 1000
        res.stream_metrics = StreamMetrics(samples)
        return res

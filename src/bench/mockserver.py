"""A tiny OpenAI-compatible mock server for smoke-testing the harness.

Runs on the stdlib only (``http.server``) so it needs no dependencies:

    uv run bench mock --port 8080 --model mock-model

It answers ``GET /v1/models`` and ``POST /v1/chat/completions`` with
streaming SSE, simulating prefill delay + per-token generation. Responses are
good enough to exercise every harness code path (structured JSON via
``response_format``, tool-call JSON, LLM-judge rubric output, dev-task code,
and fixed-length perf outputs) — it is NOT a model and will not produce
meaningful scores; use it to verify the plumbing, then point ``bench`` at a
real server for real numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- #
# canned answers (good enough to exercise the harness, not to score)
# --------------------------------------------------------------------------- #
ANSWERS = {
    "s1": '{"answer": 396}',
    "s2": '{"answer": 100}',
    "s3": (
        '{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Tokyo", "unit": "celsius"}}]}'
    ),
    "s4": (
        '{"tool_calls": ['
        '{"name": "book_flight", "arguments": {"from_city": "Paris", '
        '"to_city": "Berlin", "date": "2026-12-05", "passengers": 2, '
        '"cabin": "economy"}},'
        '{"name": "get_price", "arguments": {"booking_id": "bk_42", '
        '"currency": "EUR"}}]}'
    ),
    "s5": (
        '{"countries": ["France", "Germany", "Switzerland", "Japan"]}'
    ),
}

DEV_CODE = {
    "parse_log_line": '''\
import re


def parse_log_line(line):
    line = line.strip()
    m = re.match(r"^\\[(?P<ts>[^\\]]+)\\]\\s+(?P<level>\\w+)\\s*(?P<rest>.*)$", line)
    if not m:
        return {}
    rest = m.group("rest").strip()
    fields = {}
    while rest:
        mm = re.match(r"(\\w+)=\\s*", rest)
        if not mm:
            break
        key = mm.group(1)
        after = rest[mm.end():]
        if after.startswith('"'):
            end = after.index('"', 1)
            fields[key] = after[1:end]
            rest = after[end + 1:].strip()
        else:
            vm = re.match(r"\\S+", after)
            val = vm.group(0)
            fields[key] = int(val) if re.fullmatch(r"-?\\d+", val) else val
            rest = after[vm.end():].strip()
    return {"timestamp": m.group("ts"), "level": m.group("level"), "fields": fields}
''',
    "total": '''\
def total(nums):
    s = 0
    for x in nums:
        s += x
    return s
''',
    "first_even": '''\
def first_even(nums):
    return next(filter(lambda n: n % 2 == 0, nums), None)
''',
    "word_frequencies": '''\
import re


def word_frequencies(text):
    words = re.findall(r"[A-Za-z]+", text)
    freq = {}
    for w in (w.lower() for w in words):
        freq[w] = freq.get(w, 0) + 1
    return freq


def top_words(text, k):
    freq = word_frequencies(text)
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:k]
''',
    "Counter": '''
class Counter:
    def __init__(self):
        self._vals: list[int] = []

    def add(self, n: int) -> None:
        self._vals.append(n)

    def mean(self) -> float | None:
        if not self._vals:
            return None
        return sum(self._vals) / len(self._vals)

    def median(self) -> float | None:
        if not self._vals:
            return None
        s = sorted(self._vals)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return float(s[mid])
        return (s[mid - 1] + s[mid]) / 2.0
''',
    "lis": '''\
def lis(nums):
    if not nums:
        return 0
    tails = []
    for x in nums:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails)
''',
    "merge_intervals": '''\
def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
''',
    # d6/d9 thinking tasks (JSON answers, no code symbol) — matched by unique
    # words in the prompt. Must come LAST: the generic DEV_CODE substring scan
    # runs dict-ordered and "total" would otherwise match the clock prompt.
    "clock": '{"steps": ["Monday noon to Friday noon is 4 days", "Clock gains 5 min/day x 4 = 20 min fast", "Noon + 20 min = 12:20", "12:20 = 740 minutes after midnight"], "answer": 740}',
    "train": '{"steps": ["First 30 km costs $10", "Remaining distance: 120 - 30 = 90 km", "90 / 15 = six full chunks at $4 each = $24", "Total: 10 + 24 = 34"], "answer": 34}',
}

S6_ANSWER = (
    '{"steps": ["Monday noon to Friday noon is 4 days", '
    '"Clock gains 5 min/day x 4 = 20 min fast", '
    '"Noon + 20 min = 12:20", '
    '"12:20 = 740 minutes after midnight"], "answer": 740}'
)

OPEN_ANSWERS = {
    "o1": (
        "The Great Barrier Reef, off Queensland in Australia, is the world's largest "
        "structure made by living organisms and spans over 2,300 kilometers. "
        "Rising temperatures, pollution, and storms are causing widespread coral "
        "bleaching that threatens its ecosystem."
    ),
    "o2": (
        "- Improves cardiovascular health\n"
        "- Strengthens muscles and bones\n"
        "- Boosts mood and reduces stress\n"
        "- Helps maintain a healthy weight\n"
        "- Improves sleep quality"
    ),
    "o3": (
        "Subject: One-week delay on your order — 10% discount inside\n\n"
        "Dear Customer,\n\n"
        "I'm writing to let you know that your order will arrive one week later than "
        "planned. A supplier issue is holding up production, and we expect to "
        "resolve it within the week.\n\n"
        "As a gesture of goodwill, we've applied a 10% discount to your order — no "
        "action needed on your part.\n\n"
        "Thank you for your patience.\n\n"
        "Best regards,\n"
        "Support Team"
    ),
}

JUDGE_ANSWER = (
    '{"scores": {"faithfulness": 1.0, "brevity": 1.0, '
    '"instruction_following": 1.0, "quality": 1.0, "tone": 1.0}, '
    '"justification": "Response meets the rubric on every dimension."}'
)

PERF_TOKEN = "tok"


def _detect(messages: list[dict], body: dict) -> str | None:
    """Pick a canned answer for the request, or None for free generation."""
    user = messages[-1].get("content", "") if messages else ""
    # Judge requests are always last (the harness judges after answering).
    if "RUBRIC:" in user and "Score each dimension" in user:
        return JUDGE_ANSWER
    schema = (body.get("response_format") or {}).get("json_schema", {})
    props = (schema.get("schema") or {}).get("properties", {})
    if "answer" in props:
        # decide which arithmetic/reasoning task by the prompt text
        for marker, key in (("17 * 23", "s1"), ("shelves", "s2")):
            if marker in user.lower() or marker in user:
                return ANSWERS[key]
        if "clock" in user.lower():
            return S6_ANSWER
        return ANSWERS["s1"]
    if "countries" in props:
        return ANSWERS["s5"]
    if "tool_calls" in props:
        return ANSWERS["s4"] if "flight" in user.lower() else ANSWERS["s3"]
    # Tool-call tasks may arrive WITHOUT a schema (the model emits the JSON
    # itself); match on intent keywords instead.
    if "book a flight" in user.lower():
        return ANSWERS["s4"]
    if "weather" in user.lower():
        return ANSWERS["s3"]
    # Thinking tasks (d6 clock / d9 train) carry no schema and no unique code
    # symbol — match on a phrase from the problem before the generic DEV_CODE
    # scan (whose "total" key would otherwise match the clock prompt).
    if "gains 5 minutes every day" in user:
        return DEV_CODE["clock"]
    if "train ride costs" in user:
        return DEV_CODE["train"]
    # Dev-task prompts carry no schema; match on the function name to implement.
    for key, code in DEV_CODE.items():
        if re.search(rf"\b{re.escape(key)}\b", user):
            return code
    for tid, text in (("o1", "Great Barrier Reef"), ("o2", "exactly 5 benefits"),
                      ("o3", "delayed by one week")):
        if text in user:
            return OPEN_ANSWERS[tid]
    return None


class _Handler(BaseHTTPRequestHandler):
    model_name: str = "mock-model"

    def log_message(self, fmt, *args):  # keep stdout clean
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _native_tool_calls(
        answer: str | None,
        tools: list[dict] | None,
        messages: list[dict] | None = None,
    ) -> list[dict]:
        """Build the native function calls to emit when the request carries
        real tool schemas (i.e. it is a ``kind == "tools"`` or
        ``kind == "agent"`` task).

        One-shot tasks: s3 → one ``get_weather`` call; s4 → both required
        calls (``book_flight`` + ``get_price`` with a placeholder booking_id,
        since this one-shot mock cannot model the follow-up turn where the real
        id would come from).

        Agent tasks are driven by inspecting the conversation: each turn the
        mock emits ONE tool call (deterministic progress), and once the task
        is complete it returns ``[]`` so the harness sees a final answer with
        no tool calls.
        """
        if not tools:
            return []
        names = [
            (tool.get("function") or {}).get("name", "")
            for tool in tools
        ]
        # Agent tasks are driven by the conversation, not by a canned answer —
        # _detect returns None for them, so we must NOT bail on answer is None.
        _is_agent = (
            ("list_events" in names and "schedule_event" in names)
            or ("search_kb" in names and "get_facts" in names)
            or ("list_items" in names and "adjust_stock" in names)
        )
        if not _is_agent and answer is None:
            return []
        # --- one-shot tool tasks -------------------------------------------
        if "get_weather" in names:
            return [{
                "name": "get_weather",
                "arguments": {"city": "Tokyo", "unit": "celsius"},
            }]
        if "book_flight" in names:
            calls = [{
                "name": "book_flight",
                "arguments": {
                    "from_city": "Paris", "to_city": "Berlin",
                    "date": "2026-12-05", "passengers": 2, "cabin": "economy",
                },
            }]
            if "get_price" in names:
                calls.append({
                    "name": "get_price",
                    "arguments": {"booking_id": "bk_mock_42", "currency": "EUR"},
                })
            return calls

        # --- agent tasks (multi-turn): drive by conversation state ---------
        msgs = messages or []
        tool_results = [
            m for m in msgs if m.get("role") == "tool"
        ]
        user_text = msgs[0].get("content", "") if msgs else ""

        # a1: calendar scheduling (list_events → find_free_slots → schedule_event)
        if "list_events" in names and "schedule_event" in names:
            if not tool_results:
                return [{"name": "list_events", "arguments": {"date": "2026-11-11"}}]
            if len(tool_results) == 1:
                return [{"name": "find_free_slots", "arguments": {
                    "date": "2026-11-11", "start_after": "09:00",
                    "end_before": "17:00", "duration_min": 30,
                }}]
            if len(tool_results) == 2:
                return [{"name": "schedule_event", "arguments": {
                    "date": "2026-11-11", "title": "Design sync",
                    "start": "15:30", "end": "16:00", "location": "",
                }}]
            return []  # complete → final answer

        # a2: knowledge-base research (search_kb → get_facts → final answer)
        if "search_kb" in names and "get_facts" in names:
            if not tool_results:
                return [{"name": "search_kb", "arguments": {"query": "invoice_total_usd"}}]
            if len(tool_results) == 1:
                return [{"name": "get_facts", "arguments": {"topic": "vat"}}]
            return []  # complete → final answer

        # a3: inventory audit (list_items → get_item x3 → adjust_stock x2 → final answer)
        if "list_items" in names and "adjust_stock" in names:
            n = len(tool_results)
            if n == 0:
                return [{"name": "list_items", "arguments": {}}]
            if n == 1:
                return [{"name": "get_item", "arguments": {"name": "laptop"}}]
            if n == 2:
                return [{"name": "get_item", "arguments": {"name": "mouse"}}]
            if n == 3:
                return [{"name": "get_item", "arguments": {"name": "keyboard"}}]
            if n == 4:
                return [{"name": "adjust_stock", "arguments": {"name": "laptop", "delta": 10}}]
            if n == 5:
                return [{"name": "adjust_stock", "arguments": {"name": "keyboard", "delta": 10}}]
            return []  # complete → final answer

        return []

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": self.model_name, "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return

        messages = body.get("messages", [])
        answer = _detect(messages, body)
        stream = bool(body.get("stream"))
        max_tokens = int(body.get("max_tokens") or 256)
        # When the request carries real tool schemas, answer with a native
        # function call (like llama.cpp would) instead of JSON text. For agent
        # tasks the mock inspects the conversation to decide which single tool
        # call to emit this turn (or [] when the workflow is complete).
        tool_calls = self._native_tool_calls(answer, body.get("tools"), messages)

        # Agent final answers: once the mock stops calling tools it must emit
        # a text answer that satisfies the deterministic checker.
        if not tool_calls and answer is None and body.get("tools"):
            tool_names = [(t.get("function") or {}).get("name", "") for t in (body.get("tools") or [])]
            if "list_events" in tool_names and "schedule_event" in tool_names:
                answer = "Scheduled 'Design sync' on 2026-11-11 at 15:30."
            elif "search_kb" in tool_names and "get_facts" in tool_names:
                answer = "Invoice total $240 + 15% VAT = $276. Total including VAT: 276 USD."
            elif "list_items" in tool_names and "adjust_stock" in tool_names:
                answer = "Reordered laptop (+10) and keyboard (+10); both were below the 10-unit threshold."

        if answer is None:
            # perf-style free generation: exactly max_tokens tokens
            text = (PERF_TOKEN + " ") * max_tokens
            n_prompt_words = sum(len(str(m.get("content", "")).split()) for m in messages)
        else:
            text = "" if tool_calls else answer
            n_prompt_words = sum(len(str(m.get("content", "")).split()) for m in messages)

        finish_reason = "tool_calls" if tool_calls else "stop"

        if not stream:
            choice_message: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                choice_message["tool_calls"] = [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": tc["name"],
                                   "arguments": json.dumps(tc["arguments"])}}
                    for i, tc in enumerate(tool_calls)
                ]
            self._json({
                "id": "chatcmpl-mock", "object": "chat.completion",
                "model": self.model_name,
                "choices": [{"index": 0, "message": choice_message,
                             "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": n_prompt_words,
                          "completion_tokens": len(text.split()),
                          "total_tokens": n_prompt_words + len(text.split())},
            })
            return

        # --- streaming: simulate prefill then one token per SSE chunk --------
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        prefill_s = min(0.15 + n_prompt_words * 0.004, 3.0)  # ~ms per prompt word
        time.sleep(prefill_s)

        if tool_calls:
            # Native tool-call stream: one chunk carrying the full call(s),
            # mirroring how llama.cpp emits function calls in a single delta.
            completion_tokens = sum(1 + len(tc["arguments"]) for tc in tool_calls)
            tc_deltas = [
                {"index": i, "id": f"call_{i}", "type": "function",
                 "function": {"name": tc["name"],
                               "arguments": json.dumps(tc["arguments"])}}
                for i, tc in enumerate(tool_calls)
            ]
            chunk = {
                "id": "chatcmpl-mock", "object": "chat.completion.chunk",
                "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant",
                                                    "tool_calls": tc_deltas},
                              "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        else:
            # Tokenize like a real server: each chunk carries its own whitespace
            # so concatenating delta contents reproduces the original text exactly.
            tokens = re.findall(r"\S+\s*|\s+", text) or [""]
            completion_tokens = len(tokens)
            for tok in tokens[:max_tokens]:
                delta = {"role": "assistant", "content": tok}
                chunk = {
                    "id": "chatcmpl-mock", "object": "chat.completion.chunk",
                    "model": self.model_name,
                    "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)  # ~100 tok/s decode

        final = {
            "id": "chatcmpl-mock", "object": "chat.completion.chunk",
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": n_prompt_words,
                      "completion_tokens": completion_tokens,
                      "total_tokens": n_prompt_words + completion_tokens},
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int = 8080, model: str = "mock-model") -> None:
    handler = type("H", (_Handler,), {"model_name": model})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"mock OpenAI server on http://127.0.0.1:{port}/v1  (model: {model!r})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench mock", description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="mock-model")
    args = p.parse_args(argv)
    serve(port=args.port, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

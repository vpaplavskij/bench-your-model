"""Scoring: deterministic checks for structured tasks, LLM judge for open tasks."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | None:
    """Best-effort JSON *object* extraction from a model response.

    Tries the whole string first, then each `{` position with a strict
    ``raw_decode`` — the old greedy first-brace-to-last-brace regex swallowed
    several objects or trailing prose and failed to parse.
    """
    text = text.strip()
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return v
    except (json.JSONDecodeError, ValueError):
        pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def score_structured(task: dict, content: str, native_calls: list[dict] | None = None) -> dict:
    """Score a structured task. Returns {"correct": bool, "detail": str}.

    For ``kind == "tools"`` tasks the calls come from ``native_calls`` (the
    server's real function-calling output) when present; otherwise we fall back
    to parsing ``{"tool_calls": [...]}`` out of the text content, so servers
    without native tool support are still testable.

    ``kind == "agent"`` tasks are scored by ``runner._run_agent_task`` (it owns
    the multi-turn loop and the live env), never here.
    """
    check = task["check"]
    if task["kind"] == "agent":
        raise ValueError("agent tasks are scored by runner._run_agent_task, not score_structured")
    if task["kind"] == "tools":
        source = "native"
        calls = list(native_calls or [])
        if not calls:
            parsed_text = extract_json(content)
            if isinstance(parsed_text, dict):
                text_calls = parsed_text.get("tool_calls")
                if isinstance(text_calls, list):
                    calls = [c for c in text_calls if isinstance(c, dict)]
                    source = "text"
        try:
            ok = bool(check(calls))
        except Exception as e:
            return {"correct": False, "detail": f"checker error: {e}"}
        names = [c.get("name") for c in calls]
        detail = (
            f"expected tool calls ({source}); got {names}"
            if not ok else f"tool calls accepted ({source}): {names}"
        )
        return {"correct": ok, "detail": detail}

    parsed = extract_json(content)
    if parsed is None:
        return {"correct": False, "detail": "no valid JSON object in response"}
    try:
        ok = bool(check(parsed))
    except Exception as e:
        return {"correct": False, "detail": f"checker error: {e}"}
    detail = f"parsed={json.dumps(parsed)[:200]}"

    return {"correct": ok, "detail": detail}


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator. You will be given a TASK, a MODEL "
    "RESPONSE, and a RUBRIC with dimensions. For EACH dimension in the rubric, "
    "score 0 to 1 in 0.25 steps based ONLY on the rubric criterion. "
    'Return JSON: {"scores": {"<dimension>": <number>}, "justification": "<one short sentence>"}'
)


def build_judge_messages(task: dict, response_text: str) -> list[dict]:
    rubric_lines = "\n".join(f"- {k}: {v}" for k, v in task["judge_rubric"].items())
    user = (
        f"TASK:\n{task['messages'][-1]['content']}\n\n"
        f"MODEL RESPONSE:\n{response_text}\n\n"
        f"RUBRIC:\n{rubric_lines}\n\n"
        "Score each dimension now."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_judge_scores(content: str, dimensions: list[str]) -> dict | None:
    """Parse judge output into {dim: float}; None if unparseable."""
    parsed = extract_json(content)
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("scores", {})
    out: dict[str, float] = {}
    for dim in dimensions:
        v = raw.get(dim)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        out[dim] = max(0.0, min(1.0, v))
    return out or None


def score_open(task: dict, content: str) -> dict:
    """Placeholder scoring for open tasks BEFORE the judge runs.

    The actual LLM-judge call happens in runner.py (it needs a client); this
    function only provides the deterministic part of the record so results
    are uniform.
    """
    dims = list(task["judge_rubric"].keys())
    return {"dimensions": dims, "scores": None, "detail": "pending judge"}

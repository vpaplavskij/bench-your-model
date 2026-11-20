"""Run orchestrator: executes tasks, collects metrics, writes results."""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone

from .client import BenchClient
from .config import RunConfig
from .humanize import fmt_ms, fmt_tps, glossary_md, pct
from .scoring import build_judge_messages, parse_judge_scores, score_open, score_structured
from .tasks import tasks_for_suite
from .textutil import strip_think


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def run_benchmark(cfg: RunConfig) -> dict:
    """Execute all selected tasks and return the full result document."""
    client = BenchClient(
        endpoint=cfg.endpoint,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    # Server sanity check (non-fatal)
    server_info: dict = {"endpoint": cfg.endpoint, "model_available": False}
    try:
        available = client.list_models()
        server_info["available_models"] = available
        server_info["model_available"] = cfg.model in available
    except Exception as e:
        server_info["error"] = f"{type(e).__name__}: {e}"

    tasks = [t for t in tasks_for_suite(cfg.suite) if not cfg.task_ids or t["id"] in cfg.task_ids]
    records: list[dict] = []
    judge_model = cfg.resolve_judge()
    if judge_model == cfg.model:
        print("WARNING: LLM judge = model under test (self-assessment). Pass --judge-model "
              "to score open tasks with a different model.", file=sys.stderr)

    # A --judge-model different from the model under test needs its own client;
    # otherwise the model under test would judge itself even when the flag is set.
    judge_client = client
    judge_available = True
    if cfg.judge_model and cfg.judge_model != cfg.model:
        avail = server_info.get("available_models")
        if avail and cfg.judge_model not in avail:
            judge_available = False
            print(f"WARNING: --judge-model '{cfg.judge_model}' is not served by "
                  f"{cfg.endpoint} — open tasks cannot be judged.", file=sys.stderr)
        else:
            judge_client = BenchClient(
                endpoint=cfg.endpoint, model=cfg.judge_model, api_key=cfg.api_key,
                timeout_s=cfg.timeout_s, max_tokens=cfg.max_tokens, temperature=0.0,
            )

    print(f"Running {len(tasks)} task(s) against {cfg.endpoint} (model={cfg.model})")
    for task in tasks:
        rec = _run_task(client, cfg, task, judge_model, judge_client, judge_available)
        records.append(rec)
        status = "OK " if rec["request_ok"] else "ERR"
        score = rec.get("score") or {}  # absent when the request failed
        score_txt = ""
        if rec["suite"] == "structured" and rec["request_ok"]:
            score_txt = "correct" if score.get("correct") else "WRONG"
        elif score.get("scores"):
            avg = sum(score["scores"].values()) / len(score["scores"])
            score_txt = f"judge={avg:.2f}"
        print(f"  [{status}] {task['id']} {task['name']}: ttft={fmt_ms(rec['ttft_ms'])} "
              f"total={fmt_ms(rec['total_ms'])} {score_txt}")

    unjudged = [r for r in records if r["suite"] == "open" and r["request_ok"]
                and not (r.get("score") or {}).get("scores")]
    if unjudged:
        err = (unjudged[0].get("judge") or {}).get("error") or "judge output could not be parsed"
        print(f"WARNING: {len(unjudged)} open task(s) could not be judged — {err}. "
              "Their writing-quality scores are missing, not zero.", file=sys.stderr)

    client.close()
    if judge_client is not client:
        judge_client.close()
    return _assemble(cfg, server_info, records)


def _run_agent_task(client: BenchClient, cfg: RunConfig, task: dict) -> dict:
    """Run a multi-turn agentic workflow.

    The harness drives the loop: the model calls tools (native function
    calling), the harness executes them against ``task["env"]`` — a
    deterministic in-process environment — and feeds the results back as
    ``tool`` messages until the model stops calling tools or ``max_steps`` is
    reached. Scoring is exact: final answer text AND terminal env state.
    """
    def _rec_base() -> dict:
        return {
            "task_id": task["id"],
            "name": task["name"],
            "suite": task["suite"],
            "request_ok": True,
            "error": None,
            "finish_reason": None,
            "content": None,
            "tool_calls": None,
            "ttft_ms": None,
            "total_ms": None,
            "prompt_tokens": None,
            "completion_tokens": 0,
            "tokens_per_s": None,
        }

    messages: list[dict] = list(task["messages"])
    tools = task.get("tools")
    env = task["env_factory"]()
    max_steps = int(task.get("max_steps", 8))
    all_calls: list[dict] = []
    comp_tokens = 0
    call_seq = 1  # tool_call ids must stay unique across turns (strict servers reject dupes)
    t_start = time.perf_counter()
    first_ttft: float | None = None
    content: str | None = None

    for _ in range(max_steps):
        res = client.chat(messages, tools=tools)
        if not res.ok:
            rec = _rec_base()
            rec["request_ok"] = False
            rec["error"] = res.error
            return rec
        if first_ttft is None and res.ttft_ms is not None:
            first_ttft = res.ttft_ms
        comp_tokens += res.stream_chunks or res.completion_tokens or 0

        calls = list(res.tool_calls or [])
        if not calls:
            # Model stopped calling tools → final answer.
            content = res.content
            break

        all_calls.extend(calls)
        # Parallel calls belong in ONE assistant message (OpenAI protocol),
        # followed by one tool result per call id.
        assistant_calls: list[dict] = []
        tool_results: list[dict] = []
        for call in calls:
            call_id = f"call_{call_seq}"
            call_seq += 1
            name = call.get("name", "")
            args = call.get("arguments") or {}
            raw = args.get("_raw_arguments")
            if raw is not None:
                # F5: streamed arguments were invalid JSON. Do not run the tool
                # ({} would be an action the model never asked for); hand the
                # error back and echo the model's raw string verbatim.
                result = {"error": f"invalid JSON arguments: {raw[:200]}"}
            else:
                try:
                    result = task["step"](env, name, args) if env is not None else {"error": "no env"}
                except Exception as e:  # noqa: BLE001 - a bad tool argument (e.g. int("ten")) is a task failure, not a run abort
                    result = {"error": f"tool call failed: {type(e).__name__}: {e}"}
            assistant_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args)},
            })
            tool_results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result),
            })
        messages.append({"role": "assistant", "content": None, "tool_calls": assistant_calls})
        messages.extend(tool_results)

    rec = _rec_base()
    rec["request_ok"] = True
    rec["finish_reason"] = "stop" if content is not None else "max_steps"
    rec["content"] = content
    rec["tool_calls"] = all_calls or None
    rec["ttft_ms"] = first_ttft
    rec["total_ms"] = (time.perf_counter() - t_start) * 1000.0
    rec["completion_tokens"] = comp_tokens

    if content is not None:
        # Normal completion — score it.
        try:
            ok = bool(task["check"](env, content))
        except Exception as e:
            ok = False
            rec["checker_error"] = f"checker error (harness): {type(e).__name__}: {e}"
            detail = rec["checker_error"]
        else:
            detail = (
                "agent workflow completed: final answer + environment state verified"
                if ok else "agent workflow failed: environment state and/or final answer wrong"
            )
        rec["score"] = {"correct": ok, "detail": detail}
    else:
        # Hit max_steps without a final answer — the model never stopped calling tools.
        rec["score"] = {
            "correct": False,
            "detail": f"hit max_steps={max_steps} without a final answer",
        }
    return rec


def _run_task(client: BenchClient, cfg: RunConfig, task: dict, judge_model: str,
              judge_client: BenchClient | None = None, judge_available: bool = True) -> dict:
    # Agent tasks run their own multi-turn loop.
    if task.get("kind") == "agent":
        return _run_agent_task(client, cfg, task)

    messages = list(task["messages"])
    schema = task.get("schema")
    # "tools" tasks send the real tool schemas to the server and score native
    # function calls (ChatResult.tool_calls). This exercises the model's actual
    # tool-calling path — the old design of hiding the schemas and hoping the
    # model emits JSON text made every model refuse ("no access to real-time
    # data") or answer in prose, so no configuration could ever pass.
    tools = task.get("tools") if task.get("kind") == "tools" else None

    res = client.chat(messages, json_schema=schema, tools=tools)

    rec: dict = {
        "task_id": task["id"],
        "name": task["name"],
        "suite": task["suite"],
        "request_ok": res.ok,
        "error": res.error,
        "finish_reason": res.raw_finish_reason,
        "content": res.content,
        "tool_calls": res.tool_calls or None,
        "ttft_ms": res.ttft_ms,
        "total_ms": res.total_ms,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "tokens_per_s": res.tokens_per_s,
    }

    # Score on the answer with inline <think>…</think> blocks removed; the raw
    # content stays in the record for auditability.
    answer = strip_think(res.content)

    # Deterministic scoring for structured tasks
    if task["suite"] == "structured" and res.ok:
        rec["score"] = score_structured(task, answer, native_calls=res.tool_calls)
    elif task["suite"] == "open" and res.ok:
        rec["score"] = score_open(task, answer)
        if not judge_available:
            rec["judge"] = {"model": judge_model,
                            "error": f"judge model '{judge_model}' not served by endpoint"}
        elif judge_client is None:
            rec["judge"] = {"model": judge_model, "error": "no judge client"}
        else:
            # LLM judge (skipped if the model under test is not usable as judge)
            jres = judge_client.chat(build_judge_messages(task, answer))
            if jres.ok:
                scores = parse_judge_scores(jres.content, rec["score"]["dimensions"])
                rec["judge"] = {"model": judge_model, "raw": jres.content[:500]}
                if scores:
                    rec["score"]["scores"] = scores
                    rec["score"]["detail"] = "judged"
                else:
                    rec["judge"]["error"] = "judge output could not be parsed"
            else:
                rec["judge"] = {"model": judge_model, "error": jres.error}
    return rec


def _assemble(cfg: RunConfig, server_info: dict, records: list[dict]) -> dict:
    ok_reqs = [r for r in records if r["request_ok"]]
    # Harness checker bugs are not the model's: excluded from accuracy scoring.
    structured = [r for r in ok_reqs if r["suite"] == "structured" and not r.get("checker_error")]
    open_tasks = [r for r in ok_reqs if r["suite"] == "open"]

    def _ms(vals: list) -> dict:
        vals = [v for v in vals if v is not None]
        return {
            "mean": round(statistics.mean(vals), 1) if vals else None,
            "median": round(_median(vals), 1) if vals else None,
        }

    ttft = _ms([r["ttft_ms"] for r in ok_reqs])
    total = _ms([r["total_ms"] for r in ok_reqs])
    tps = _ms([r["tokens_per_s"] for r in ok_reqs if r["tokens_per_s"]])

    n_struct = len(structured)
    correct = sum(1 for r in structured if r.get("score", {}).get("correct"))

    judge_scores: dict[str, list[float]] = {}
    for r in open_tasks:
        s = r.get("score", {}).get("scores") or {}
        for dim, v in s.items():
            judge_scores.setdefault(dim, []).append(v)
    judge_means = {d: round(statistics.mean(v), 3) for d, v in judge_scores.items() if v}
    open_unjudged = sum(1 for r in open_tasks if not r.get("score", {}).get("scores"))

    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": cfg.endpoint,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "suite": cfg.suite,
            # How the SERVER serves the model (Unsloth Studio / llama.cpp), not
            # the agent harness. See REPORT_SPEC.md.
            "settings": cfg.settings(),
        },
        "server": server_info,
        "tasks": records,
        "summary": {
            "n_tasks": len(records),
            "n_ok_requests": len(ok_reqs),
            "success_rate": round(len(ok_reqs) / len(records), 3) if records else None,
            "ttft_ms": ttft,
            "total_ms": total,
            "tokens_per_s": tps,
            "structured": {
                "n": n_struct,
                "correct": correct,
                "accuracy": round(correct / n_struct, 3) if n_struct else None,
            },
            "checker_errors": sum(1 for r in records if r.get("checker_error")),
            "judge_dimension_means": judge_means or None,
            "open_unjudged": open_unjudged,
        },
    }


def render_markdown(doc: dict) -> str:
    s = doc["summary"]
    m = doc["meta"]
    st = s["structured"]

    # --- In plain words ----------------------------------------------------
    plain: list[str] = []
    if s["n_ok_requests"] == s["n_tasks"]:
        plain.append(f"- Asked {s['n_tasks']} questions; it answered all of them.")
    else:
        plain.append(f"- Asked {s['n_tasks']} questions; got an answer for "
                     f"{s['n_ok_requests']} of them ({pct(s['success_rate'])}).")
    if st["n"]:
        plain.append(f"- Quiz with exact answers: got **{st['correct']} of {st['n']}** right "
                     f"({pct(st['accuracy'])}).")
    judge = s.get("judge_dimension_means") or {}
    if judge:
        best = max(judge.values())
        plain.append(f"- Writing quality (judged by a second model, 0-1): "
                     f"**{best:.2f}** on every dimension — {best:.2f}/1.00."
                     if all(abs(v - best) < 0.005 for v in judge.values()) else
                     f"- Writing quality (judged by a second model, 0-1): "
                     f"{', '.join(f'{k} {v:.2f}' for k, v in judge.items())}.")
    if s.get("open_unjudged"):
        plain.append(f"- {s['open_unjudged']} open task(s) could not be judged — "
                     "their writing-quality scores are missing, not zero.")
    if s["ttft_ms"]["median"] is not None:
        plain.append(f"- Speed: first word after **{fmt_ms(s['ttft_ms']['median'])}** on average; "
                     f"full answer in {fmt_ms(s['total_ms']['median'])}.")

    lines = [
        f"# Quality benchmark — `{m['model']}`",
        "",
        f"_endpoint: `{m['endpoint']}`  ·  date: {m['timestamp']}_",
        "",
        "## In plain words",
        *plain,
        "",
        "## Speed",
        "",
        "| What | Mean | Median |",
        "|---|---|---|",
        f"| Time to first word (TTFT) | {fmt_ms(s['ttft_ms']['mean'])} | {fmt_ms(s['ttft_ms']['median'])} |",
        f"| Full answer | {fmt_ms(s['total_ms']['mean'])} | {fmt_ms(s['total_ms']['median'])} |",
    ]
    if s["tokens_per_s"]["mean"] is not None:
        lines.append(f"| Writing speed | {fmt_tps(s['tokens_per_s']['mean'])} | {fmt_tps(s['tokens_per_s']['median'])} |")
    _q_total = sum(r["total_ms"] for r in doc["tasks"] if r.get("total_ms") is not None)
    if _q_total > 0:
        lines += ["", f"- **Total execution time** (all {s['n_tasks']} tasks): "
                    f"**{fmt_ms(_q_total)}**"]
    lines += ["", "## Quiz with exact answers (structured)", ""]
    if st["n"]:
        lines.append(f"- accuracy: **{st['correct']}/{st['n']}** ({pct(st['accuracy'])})")
    else:
        lines.append("- no structured tasks run")
    if judge:
        lines += ["", "## Writing quality (open tasks, judged 0-1)", ""]
        lines += [f"- {dim}: **{v:.2f}**" for dim, v in judge.items()]
    lines += ["", "## Per-task detail", "",
              "| task | what it tests | result | first word | total time | what happened |",
              "|---|---|---|---|---|---|"]
    for r in doc["tasks"]:
        score = r.get("score") or {}
        if not r["request_ok"]:
            result, why = "no answer", (r.get("error") or "request failed")[:60]
        elif r["suite"] == "structured":
            result = "✓ correct" if score.get("correct") else "✗ wrong"
            why = _plain_structured_why(r)
        else:
            vals = (score or {}).get("scores") or {}
            avg = sum(vals.values()) / len(vals) if vals else None
            result = f"✓ {avg:.2f}/1.00" if avg is not None else "-"
            why = "judged by second model" if avg is not None else "-"
        lines.append(f"| {r['task_id']} | {r.get('name', '')} | {result} "
                     f"| {fmt_ms(r['ttft_ms'])} | {fmt_ms(r['total_ms'])} | {why} |")
    lines += glossary_md()
    return "\n".join(lines) + "\n"


def _plain_structured_why(r: dict) -> str:
    """One plain-words line explaining a structured task's outcome."""
    score = r.get("score") or {}
    detail = (score.get("detail") or "").strip()
    if score.get("correct"):
        return "right answer"
    content = (r.get("content") or "").strip()
    if not content:
        return "no usable answer"
    try:
        import json as _json
        parsed = _json.loads(content)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        if "answer" in parsed and "tool_calls" not in parsed:
            return f"answered {parsed['answer']} — expected a different value"
        if "countries" in parsed:
            return "wrong list"
        if "tool_calls" in parsed:
            names = [c.get("name") for c in parsed["tool_calls"] if isinstance(c, dict)]
            return f"called {', '.join(n for n in names if n) or 'a tool'} — wrong call/arguments"
    low = content.lower()
    if any(w in low for w in ("i can't", "i cannot", "i'm not able", "not able to",
                              "no access", "can't access", "don't have access")):
        return "refused — said it can't do this"
    if detail and "no valid JSON" in detail:
        return "answered in plain text instead of the required format"
    return detail[:60] or "wrong answer"

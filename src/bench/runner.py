"""Run orchestrator: executes tasks, collects metrics, writes results."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .client import BenchClient
from .config import RunConfig
from .humanize import fmt_ms, fmt_tps, glossary_md, pct
from .scoring import build_judge_messages, parse_judge_scores, score_open, score_structured
from .tasks import tasks_for_suite


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

    print(f"Running {len(tasks)} task(s) against {cfg.endpoint} (model={cfg.model})")
    for task in tasks:
        rec = _run_task(client, cfg, task, judge_model)
        records.append(rec)
        status = "OK " if rec["request_ok"] else "ERR"
        score_txt = ""
        if rec["suite"] == "structured":
            score_txt = "correct" if rec["score"]["correct"] else "WRONG"
        elif rec["score"].get("scores"):
            avg = sum(rec["score"]["scores"].values()) / len(rec["score"]["scores"])
            score_txt = f"judge={avg:.2f}"
        print(f"  [{status}] {task['id']} {task['name']}: ttft={rec['ttft_ms'] and round(rec['ttft_ms'])}ms "
              f"total={round(rec['total_ms'])}ms {score_txt}")

    client.close()
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

        calls = list(res.tool_calls or [])
        if not calls:
            # Model stopped calling tools → final answer.
            content = res.content
            break

        all_calls.extend(calls)
        for i, call in enumerate(calls):
            name = call.get("name", "")
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                # arguments may have failed to parse; keep raw text.
                try:
                    args = json.loads(call.get("_raw_arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
            result = task["step"](env, name, args) if env is not None else {"error": "no env"}
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": json.dumps(result),
            })

    rec = _rec_base()
    rec["request_ok"] = True
    rec["finish_reason"] = "stop" if content is not None else "max_steps"
    rec["content"] = content
    rec["tool_calls"] = all_calls or None
    rec["ttft_ms"] = first_ttft
    rec["total_ms"] = (time.perf_counter() - t_start) * 1000.0
    rec["completion_tokens"] = len(all_calls)

    if content is not None:
        # Normal completion — score it.
        try:
            ok = bool(task["check"](env, content))
        except Exception as e:
            ok = False
            detail = f"checker error: {e}"
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


def _run_task(client: BenchClient, cfg: RunConfig, task: dict, judge_model: str) -> dict:
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

    # Deterministic scoring for structured tasks
    if task["suite"] == "structured" and res.ok:
        rec["score"] = score_structured(task, res.content, native_calls=res.tool_calls)
    elif task["suite"] == "open" and res.ok:
        rec["score"] = score_open(task, res.content)
        # LLM judge (skipped if the model under test is not usable as judge)
        jres = client.chat(build_judge_messages(task, res.content))
        if jres.ok:
            scores = parse_judge_scores(jres.content, rec["score"]["dimensions"])
            rec["judge"] = {"model": judge_model, "raw": jres.content[:500]}
            if scores:
                rec["score"]["scores"] = scores
                rec["score"]["detail"] = "judged"
        else:
            rec["judge"] = {"model": judge_model, "error": jres.error}
    return rec


def _assemble(cfg: RunConfig, server_info: dict, records: list[dict]) -> dict:
    ok_reqs = [r for r in records if r["request_ok"]]
    structured = [r for r in ok_reqs if r["suite"] == "structured"]
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

    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": cfg.endpoint,
            "model": cfg.model,
            "concurrency": cfg.concurrency,
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
            "judge_dimension_means": judge_means or None,
        },
    }


def write_results(doc: dict, out_dir: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    md_path = out / "summary.md"
    json_path.write_text(json.dumps(doc, indent=2))
    md_path.write_text(render_markdown(doc))
    return json_path, md_path


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

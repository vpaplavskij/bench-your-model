"""Developer/coding benchmark: accuracy, runtime errors, total time, ranking.

For each task in :data:`devtasks.DEV_TASKS`:
  1. ask the model for code (single request),
  2. extract the Python source,
  3. run it in a sandboxed namespace and apply the task's deterministic checker,
  4. record pass/fail, any runtime errors, and total wall time.

A composite score ranks which model best suits developer work:
    accuracy (pass rate) is primary; runtime-error-free execution and lower
    total time break ties / add signal.
"""

from __future__ import annotations

import signal
import statistics
import time
import traceback
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .client import BenchClient
from .config import RunConfig
from .version import HARNESS_REV
from .devtasks import DEV_TASKS, extract_code
from .humanize import fmt_ms, glossary_md, pct
from .textutil import strip_think


# --------------------------------------------------------------------------- #
# Sandboxed execution
# --------------------------------------------------------------------------- #
# Top-level modules blocked in the exec namespace: fs/net/subprocess access.
# The dev tasks and their checkers only need json/re/math/collections/itertools.
# shortcut: blocks imports + open(), not attribute-level escapes (e.g. builtins
# reached via gc or __subclasses__) — documented in README/AGENTS; upgrade to a
# subprocess sandbox with rlimits (brief C4) if the harness ever faces remote servers.
_BLOCKED_MODULES = frozenset({"os", "sys", "socket", "subprocess", "shutil", "tempfile",
                              "http", "urllib", "pathlib", "ctypes", "pty", "signal",
                              "multiprocessing", "threading",
                              # io/builtins/importlib/posix were the escape hatch:
                              # io.open and builtins.open bypassed the popped open().
                              "io", "builtins", "importlib", "posix",
                              # second-wave escapes (brief C4 fallback): async/task
                              # spawning, module runners, deserialization payloads,
                              # interpreter debugging, and less-common fs/net modules.
                              "asyncio", "concurrent", "_thread", "runpy", "pickle",
                              "marshal", "code", "pdb", "gc", "glob", "fileinput",
                              "zipfile", "shelve", "dbm", "ssl", "smtplib", "ftplib",
                              "webbrowser"})

# SIGALRM guard (Linux, single-threaded main loop): an infinite loop in
# model-written code would otherwise hang the whole bench run forever.
_DEV_EXEC_TIMEOUT_S = 15.0


class _ExecTimeout(BaseException):
    """BaseException on purpose: model code doing `except Exception:` cannot swallow it."""


def _alarm_handler(signum, frame):
    raise _ExecTimeout(f"dev exec timed out after {int(_DEV_EXEC_TIMEOUT_S)}s (infinite loop?)")


def _safe_builtins() -> dict:
    import builtins
    bi = dict(builtins.__dict__)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in _BLOCKED_MODULES:
            raise ImportError(f"'{name}' is blocked in the dev sandbox (no fs/net/subprocess)")
        return builtins.__import__(name, globals, locals, fromlist, level)

    bi["__import__"] = guarded_import
    bi.pop("open", None)
    return bi


def run_code(code_src: str) -> dict:
    """Execute ``code_src`` in a fresh namespace.

    Returns {"ok": bool, "error": str|None, "ns": dict}. Never raises —
    import/syntax/runtime failures are captured as errors (that's a signal).
    """
    ns: dict = {"__builtins__": _safe_builtins()}
    try:
        code = compile(code_src, "<devtask>", "exec")
        signal.signal(signal.SIGALRM, _alarm_handler)
        # Repeating interval: a bare `except:` that swallows one alarm cannot
        # finish the task; code keeps getting interrupted until exec returns.
        # shortcut: code that catches every exception forever is unkillable
        # in-process — upgrade to a subprocess watchdog if that ever matters.
        signal.setitimer(signal.ITIMER_REAL, _DEV_EXEC_TIMEOUT_S, _DEV_EXEC_TIMEOUT_S)
        try:
            exec(code, ns)  # noqa: S102 - deliberate sandbox for benchmarking
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return {"ok": True, "error": None, "ns": ns}
    except _ExecTimeout as e:  # noqa: BLE001 - a hang is a scored signal, not a crash
        return {"ok": False, "error": str(e), "ns": ns}
    except Exception as e:  # noqa: BLE001 - we WANT to capture every failure
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "ns": ns}


def _load_checker(checker_src: str):
    """Compile a task's checker source into a callable check(code_src, run)."""
    ns: dict = {}
    exec(compile(checker_src, "<checker>", "exec"), ns)  # noqa: S102
    return ns["check"]


# --------------------------------------------------------------------------- #
# Per-task record
# --------------------------------------------------------------------------- #
@dataclass
class DevTaskResult:
    task_id: str
    name: str
    kind: str
    request_ok: bool = True
    request_error: str | None = None
    code: str = ""
    passed: bool = False
    checker_detail: str = ""
    runtime_error: str | None = None      # error from executing the model's code
    checker_error: str | None = None      # harness fault (checker failed to load) — never penalizes the model
    finish_reason: str | None = None      # server finish reason; "length" = answer cut off at max_tokens
    total_ms: float = 0.0                 # whole task wall time (request + check)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def status(self) -> str:
        if not self.request_ok:
            return "REQUEST-FAIL"
        if self.runtime_error and not self.passed:
            return "RUNTIME-ERROR"
        if self.checker_error:
            return "CHECKER-ERROR"  # harness bug — task not scored
        return "PASS" if self.passed else "FAIL"


def _run_dev_task(client: BenchClient, task: dict) -> DevTaskResult:
    t0 = time.perf_counter()
    res = client.chat([{"role": "user", "content": task["prompt"]}])
    rec = DevTaskResult(task_id=task["id"], name=task["name"], kind=task["kind"])
    if not res.ok:
        rec.request_ok = False
        rec.request_error = res.error
        rec.total_ms = (time.perf_counter() - t0) * 1000
        return rec

    rec.prompt_tokens = res.prompt_tokens
    rec.completion_tokens = res.completion_tokens
    rec.finish_reason = res.raw_finish_reason
    # Inline <think>…</think> drafts may contain fenced code blocks: score the
    # code after the thinking, not a draft from inside it.
    rec.code = extract_code(strip_think(res.content))

    # Thinking tasks answer with JSON, not Python: exec'ing it would turn a
    # valid answer containing null/true into a fake "runtime error". Their
    # checkers never call `run`.
    run_res = {"ok": True, "error": None, "ns": {}} if task["kind"] == "thinking" else run_code(rec.code)
    if not run_res["ok"]:
        rec.runtime_error = run_res["error"]
        rec.passed = False
        rec.checker_detail = f"code failed to import/run: {run_res['error']}"
    else:
        # Genuine harness faults only (checker source won't compile / no check())
        # may mark a task as a checker error; anything raised WHILE checking is
        # the model's output breaking the check and must be scored as a FAIL.
        try:
            check_fn = _load_checker(task["checker_src"])
        except Exception as e:  # noqa: BLE001 - typo in checker src: record, don't crash the run
            rec.checker_error = f"checker load failed: {type(e).__name__}: {e}"
            rec.total_ms = (time.perf_counter() - t0) * 1000
            return rec

        def _run(src: str) -> dict:
            r = run_code(src)
            if r["ok"]:
                # run_code disarms its timer on return; re-arm it so the model's
                # function CALLS made by the checker are watchdog-guarded too
                # (a wrong has_cycle looping forever on a cyclic list would
                # otherwise hang the whole run).
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.setitimer(signal.ITIMER_REAL, _DEV_EXEC_TIMEOUT_S, _DEV_EXEC_TIMEOUT_S)
            return r

        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, _DEV_EXEC_TIMEOUT_S, _DEV_EXEC_TIMEOUT_S)
            try:
                outcome = check_fn(rec.code, _run)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            rec.passed = bool(outcome.get("pass"))
            rec.checker_detail = outcome.get("detail", "")
        except _ExecTimeout as e:  # noqa: BLE001
            # A hang inside the checker's calls is the model's infinite loop —
            # a scored signal, not a harness bug.
            rec.runtime_error = str(e)
            rec.checker_detail = str(e)
        except Exception as e:  # noqa: BLE001 - never exclude a wrong answer from scoring
            detail = f"{type(e).__name__}: {e}"
            frames = traceback.extract_tb(e.__traceback__)
            rec.passed = False
            if any(f.filename == "<devtask>" for f in frames):
                # Raised inside the model's own code: scored runtime error (-5).
                rec.runtime_error = detail
                rec.checker_detail = f"model code raised during checks: {detail}"
            else:
                # Wrong return type etc. (e.g. missing `return` -> None): the
                # model's output broke the checks in the checker's own frame —
                # a plain FAIL, still counted against the model.
                rec.checker_detail = f"model output broke the checks (counted as FAIL): {detail}"

    # A capped response (very common for thinking models: reasoning eats the
    # budget) fails because the code is cut off — say so. Scoring unchanged.
    if rec.finish_reason == "length" and not rec.passed:
        rec.checker_detail = f"response truncated at max_tokens; {rec.checker_detail}"

    rec.total_ms = (time.perf_counter() - t0) * 1000
    return rec


# --------------------------------------------------------------------------- #
# Orchestration + scoring
# --------------------------------------------------------------------------- #
def run_dev_benchmark(cfg: RunConfig, task_ids: list[str] | None = None) -> dict:
    client = BenchClient(
        endpoint=cfg.endpoint, model=cfg.model, api_key=cfg.api_key,
        timeout_s=cfg.timeout_s, max_tokens=cfg.max_tokens, temperature=0.0,
    )
    server_info: dict = {"endpoint": cfg.endpoint, "model_available": False}
    try:
        available = client.list_models()
        server_info["available_models"] = available
        server_info["model_available"] = cfg.model in available
    except Exception as e:
        server_info["error"] = f"{type(e).__name__}: {e}"

    tasks = [t for t in DEV_TASKS if not task_ids or t["id"] in (task_ids or [])]
    records: list[DevTaskResult] = []
    print(f"Running {len(tasks)} dev task(s) against {cfg.endpoint} (model={cfg.model})")
    for task in tasks:
        rec = _run_dev_task(client, task)
        records.append(rec)
        extra = ""
        if rec.runtime_error:
            extra = f" | runtime-error: {rec.runtime_error[:60]}"
        print(f"  [{rec.status}] {task['id']} {task['name']}: "
              f"{round(rec.total_ms)}ms{extra}")

    client.close()
    return _assemble(cfg, server_info, records)


def _med(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 1) if vals else None


def composite_score(records: list[DevTaskResult]) -> float:
    """0-100 developer-suitability score.

    accuracy (pass rate) * 100, minus a small penalty per runtime error and
    per timeout/failure-to-execute, so a model that passes everything but
    crashes on one task ranks below a clean one.
    """
    # Checker bugs are harness errors, not the model's: those tasks are not scored.
    scored = [r for r in records if not r.checker_error]
    n = len(scored)
    if n == 0:
        return 0.0
    passed = sum(1 for r in scored if r.passed)
    accuracy_pts = (passed / n) * 100.0
    runtime_errors = sum(1 for r in scored if r.runtime_error and not r.passed)
    request_fails = sum(1 for r in scored if not r.request_ok)
    penalty = 5.0 * runtime_errors + 10.0 * request_fails
    return max(0.0, round(accuracy_pts - penalty, 1))


def _assemble(cfg: RunConfig, server_info: dict, records: list[DevTaskResult]) -> dict:
    n = len(records)
    scored = [r for r in records if not r.checker_error]  # checker bugs don't count against the model
    passed = sum(1 for r in scored if r.passed)
    runtime_errors = [r for r in records if r.runtime_error and not r.passed]
    request_fails = [r for r in records if not r.request_ok]
    total_ms_all = [r.total_ms for r in records]

    # Breakdown by task kind (which dev skill is strongest/weakest).
    by_kind: dict[str, dict] = {}
    for r in scored:
        k = by_kind.setdefault(r.kind, {"n": 0, "passed": 0})
        k["n"] += 1
        k["passed"] += 1 if r.passed else 0
    for v in by_kind.values():
        v["accuracy"] = round(v["passed"] / v["n"], 3) if v["n"] else None

    error_log = [
        {"task_id": r.task_id, "kind": r.kind, "error": r.runtime_error}
        for r in runtime_errors
    ] + [
        {"task_id": r.task_id, "kind": r.kind, "error": f"request failed: {r.request_error}"}
        for r in request_fails
    ] + [
        {"task_id": r.task_id, "kind": r.kind, "error": r.checker_error}
        for r in records if r.checker_error
    ]

    return {
        "meta": {
            "kind": "dev",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": cfg.model,
            "endpoint": cfg.endpoint,
            # The dev client always samples at 0.0 regardless of cfg.temperature.
            "temperature": 0.0,
            "max_tokens": cfg.max_tokens,
            # How the SERVER serves the model (Unsloth Studio / llama.cpp), not
            # the agent harness. See REPORT_SPEC.md.
            "settings": cfg.settings(),
            "harness_rev": HARNESS_REV,
        },
        "server": server_info,
        "tasks": [
            {
                "task_id": r.task_id, "name": r.name, "kind": r.kind,
                "status": r.status, "passed": r.passed,
                "checker_detail": r.checker_detail,
                "runtime_error": r.runtime_error,
                "checker_error": r.checker_error,
                "finish_reason": r.finish_reason,
                "request_error": r.request_error,
                "total_ms": round(r.total_ms, 1),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "code": r.code,
            }
            for r in records
        ],
        "summary": {
            "n_tasks": n,
            "passed": passed,
            "accuracy": round(passed / len(scored), 3) if scored else None,
            "runtime_errors": len(runtime_errors),
            "request_failures": len(request_fails),
            "checker_errors": sum(1 for r in records if r.checker_error),
            "truncated": sum(1 for r in records if r.finish_reason == "length"),
            "total_execution_ms": {
                "mean": round(sum(total_ms_all) / len(total_ms_all), 1) if total_ms_all else None,
                "median": _med(total_ms_all),
                "max": round(max(total_ms_all), 1) if total_ms_all else None,
            },
            "by_kind_accuracy": by_kind,
            "composite_score": composite_score(records),
        },
        "error_log": error_log,
    }


def render_dev_markdown(doc: dict) -> str:
    s = doc["summary"]
    m = doc["meta"]

    plain: list[str] = [
        f"- Gave it {s['n_tasks']} real coding jobs (write, fix, refactor, build). "
        f"It finished them all and we ran its code to check the work."
    ]
    chk = s["checker_errors"]
    n_scored = s["n_tasks"] - chk
    if chk:
        plain.append(f"- {chk} job(s) hit a **harness checker error** — a bug in the "
                     "test harness, not the model; those jobs are excluded from scoring.")
    if s["passed"] == n_scored:
        plain.append(f"- **All {n_scored} scored jobs passed** — every program it wrote actually works.")
    else:
        plain.append(f"- **{s['passed']}/{n_scored} scored jobs passed.** "
                     f"{'One job' if n_scored - s['passed'] == 1 else 'Some jobs'} did not work out — see below.")
    if s["runtime_errors"]:
        n = s["runtime_errors"]
        plain.append(f"- {n} of its programs **crashed when we ran them** — "
                     f"the code it wrote has a bug in it.")
    if s["request_failures"]:
        plain.append(f"- {s['request_failures']} request(s) never even got an answer back.")
    if s["truncated"]:
        plain.append(f"- {s['truncated']} answer(s) hit the max_tokens cap and were cut off — "
                     "raise --max-tokens (thinking models spend the budget on reasoning).")
    plain.append(f"- A typical job took **{fmt_ms(s['total_execution_ms']['median'])}** from prompt to finished, working code.")
    _d_total = sum(t["total_ms"] for t in doc["tasks"] if t.get("total_ms") is not None)
    if _d_total > 0:
        plain.append(f"- **Total execution time** (all {s['n_tasks']} jobs): **{fmt_ms(_d_total)}**.")
    plain.append(f"- Overall dev-suitability score: **{s['composite_score']}/100** "
                 "(mix of pass rate and how clean the code ran).")

    lines = [
        f"# Developer-task benchmark — `{m['model']}`",
        "",
        f"_endpoint: `{m['endpoint']}`  ·  date: {m['timestamp']}_",
        "",
        "## In plain words",
        *plain,
        "",
        "## The jobs",
        "",
        "| job | what it tests | result | took | what happened |",
        "|---|---|---|---|---|",
    ]
    for t in doc["tasks"]:
        status = t["status"]
        if status == "PASS":
            res = "✓ passed"
        elif status == "RUNTIME-ERROR":
            res = "💥 crashed at runtime"
        elif status == "REQUEST-FAIL":
            res = "⚠ no answer received"
        else:
            res = "✗ failed checks"
        lines.append(
            f"| {t['task_id']} — {t.get('name', '')} | {t['kind']} | {res} "
            f"| {fmt_ms(t['total_ms'])} | {t['checker_detail'][:80]} |"
        )
    if s["by_kind_accuracy"]:
        lines += ["", "## Accuracy by skill", ""]
        for kind, v in s["by_kind_accuracy"].items():
            lines.append(f"- **{kind}**: {v['passed']}/{v['n']} passed ({pct(v['accuracy'])})")
    if doc["error_log"]:
        lines += ["", "## What went wrong (error log)", ""]
        for e in doc["error_log"]:
            lines.append(f"- `{e['task_id']}` ({e['kind']}): {e['error']}")
    else:
        lines += ["", "## What went wrong (error log)", "", "- none — every task ran and was scored cleanly"]
    lines += glossary_md()
    return "\n".join(lines) + "\n"

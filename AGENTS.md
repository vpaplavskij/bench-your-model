# AGENTS.md — bench-your-model

Benchmark harness for local LLM servers exposing an OpenAI-compatible API.
Single Python package `src/bench/` (uv, Python ≥3.10, deps: httpx). Linux only —
the `dev` exec timeout uses SIGALRM, which macOS/Windows don't have.

## Commands

```bash
uv sync
uv run bench all --model <name> --endpoint http://localhost:8080/v1 \
    --quant <Q> --reasoning-level <lvl> --reasoning-budget <n> --preserve-thinking
```

- `bench all` runs **quality → perf → dev** back-to-back into ONE run folder. It is the only command you normally need.
- Subcommands: `quality` (default when no subcommand given), `perf`, `dev`, `all`, `mock`.
- Smoke test without a model: `uv run bench mock --port 8090 --model mock-model` (stdlib-only mock OpenAI server). Point a `bench all` run at it to verify plumbing (SSE parsing, scoring, result folders).
- The mock's answers are keyed on prompt substrings in `mockserver._detect`; unmatched prompts get a `tok tok…` filler that scores/fails. A full `bench mock` smoke run passes 11/11 structured (agent tasks s4/a1–a5 are driven conversationally) and 13/13 dev; o1–o3 are judge-scored from a canned rubric answer.
- **No test suite, linter, or CI.** Verify changes by running `bench all` against `bench mock`.

## Results layout (gitignored, never overwritten)

```
results/<MODEL>-<QUANT>/<run-ts>/    # <run-ts> = UTC timestamp; re-runs append
  quality.{json,md}  perf.{json,md}  dev.{json,md}
```

- Model id `/` is sanitized to `_`; the quant is part of the folder key — **one quant, one folder, never mix quants in one folder** (that's the historical failure mode that made old runs uninterpretable).
- No `--quant` → folder is `results/<MODEL>/` (no suffix) + stderr warning.
- Server settings used are stored in every `*.json` under `meta.settings`.

## REPORT_SPEC.md is the contract

- Every run must record how the **server** serves the model: `--quant`, `--reasoning-level`, `--reasoning-budget` (`-1` unlimited, `0` no thinking), `--preserve-thinking`/`--no-preserve-thinking`, `--context-length`.
- These values come from the inference server's (Unsloth Studio / llama.cpp) startup — **not** from any agent harness. Unknown → omit the flag (records `not recorded`); never guess.
- Each standard's `.md` must carry its own **Total execution time** line (quality/dev: sum of per-task `total_ms`; perf: estimated from point aggregates).

## Map: standard → code

| Standard | Code |
|---|---|
| quality (14 tasks: s1–s6 structured, a1–a5 agent, o1–o3 open/LLM-judge) | `runner.py` (orchestrator, agent multi-turn loop), `tasks.py` (task defs), `scoring.py` (checkers + judge), `client.py` (SSE client, token accounting) |
| perf (TTFT, prefill/decode tok/s, ITL, stalls, bottleneck diagnosis) | `perf.py` |
| dev (13 jobs d1–d13: code/debug/refactor/multi/thinking) | `dev.py` (runs model code, composite score), `devtasks.py` (task defs + checkers) |
| shared | `cli.py` (argparse dispatch), `config.py` (`RunConfig`), `report.py` (result-folder path helpers only), `humanize.py` (md formatters), `mockserver.py` |

## Adding / fixing tasks

- A task is a **plain dict** — quality tasks in `src/bench/tasks.py`, dev tasks in `src/bench/devtasks.py`.
- Quality task kinds: `tools` (native function calling, scored exactly; falls back to parsing `{"tool_calls": [...]}` from text if the server lacks native support), `agent` (multi-turn: harness drives the tool loop against `task["env_factory"]()` env, scored on final answer AND terminal env state), structured/open (JSON schema + exact match / LLM-judge rubric).
- Dev tasks need a `prompt` + `checker_src` string defining `check(code_src, run) -> {"pass": bool, "detail": str}`.
- **Keep task ids and names stable** so historical cross-run comparisons stay valid — fix tasks in place, don't renumber.
- Prompt wording is load-bearing for the mock: `_detect` matches substrings like `"17 * 23"`, `"shelves"`, `"book a flight"`, `"gains 5 minutes every day"`, `"train ride costs"`, `"Great Barrier Reef"`, `"exactly 5 benefits"`, `"delayed by one week"`, dev function names. Editing a prompt (or its checker) silently breaks `bench mock` for that task.

## Docs hierarchy (read before acting)

- `RUN.md` — strict runbook for the *model* that runs a benchmark.
- `BENCHMARK_INSTRUCTIONS.md` — rules for the *model under test* (it is a subject, not an operator: no file edits, no commands, no re-runs).
- `REPORT_SPEC.md` — the reporting contract above, with legacy-run backfill procedure.

## Gotchas

- `--endpoint` must include `/v1` — `client.py` POSTs `/chat/completions` directly to it.
- Dead flags: none — `--concurrency` and `--out-dir` were removed (tasks run sequentially; folder is `--root`-based). `BENCH_ENDPOINT` / `BENCH_API_KEY` env vars are read by all subcommand parsers.
- `--temperature` applies to `quality` only — `dev`/`perf` clients hardcode `0.0` (the `dev` parser no longer accepts the flag).
- perf/quality token counting only looks at `delta.content` (content chunks ≈ content tokens), so server-side thinking tokens are excluded from content-only `ttft_ms` / `decode_tps` / ITL / writing speed. Thinking deltas (`reasoning_content`, `vLLM/OpenRouter `reasoning`, `thinking`) are counted separately: `ttft_first_token_ms`, `reasoning_chunks`, and the thinking-inclusive `decode_tps_all` / `*_all` ITL metrics measure the whole generation phase. `completion_tokens` in the json still carries the server `usage` value, thinking included. Agent tasks sum content chunks across turns (older runs recorded the tool-call *count* there — not comparable).
- perf `diagnose` uses first-token time (thinking included) for the prefill/decode share and the `*_all` metrics when reasoning was streamed; a run whose `--output-tokens` budget was spent entirely on reasoning is reported as such.
- Scoring input goes through `textutil.strip_think()` (inline `<think>…</think>` chain-of-thought is removed before JSON/code parsing); records keep the raw content for auditability.
- dev `summary.truncated` counts answers cut off at `max_tokens` (`finish_reason == "length"`); their `checker_detail` is prefixed with the truncation note — scoring is unchanged.
- Exit codes are 0 even when tasks fail or hit runtime errors — only request/connection errors abort; failed tasks print a stderr WARNING. Read the `*.json`, not the exit code.
- `bench all` / `bench dev` take ~5–10 min against a real model; run in background, don't poll.
- The dev harness executes model code in a guarded `exec` namespace (imports of fs/net/subprocess modules — os, sys, socket, subprocess, shutil, tempfile, pathlib, urllib, … — and `open()` are blocked; json/re/math/collections/itertools stay available). Fine for trusted local models, **do not point it at untrusted remote endpoints**.
- Don't kill processes on the machine while benchmarking (llama-server / Unsloth Studio / mock servers may be running).
- Prompt sizes in perf are approximate (~1 token/word filler); actual counts come from server `usage`.

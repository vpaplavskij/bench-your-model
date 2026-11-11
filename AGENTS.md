# AGENTS.md — bench-your-model

Benchmark harness for local LLM servers exposing an OpenAI-compatible API.
Single Python package `src/bench/` (uv, Python ≥3.10, deps: httpx + pydantic).

## Commands

```bash
uv sync
uv run bench all --model <name> --endpoint http://localhost:8080/v1 \
    --quant <Q> --reasoning-level <lvl> --reasoning-budget <n> --preserve-thinking
```

- `bench all` runs **quality → perf → dev** back-to-back into ONE run folder. It is the only command you normally need.
- Subcommands: `quality` (default when no subcommand given), `perf`, `dev`, `all`, `mock`.
- Smoke test without a model: `uv run bench mock --port 8090 --model mock-model` (stdlib-only mock OpenAI server; streams checker-passing answers for every task). Point a `bench all` run at it to verify plumbing (SSE parsing, scoring, result folders).
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

## Docs hierarchy (read before acting)

- `RUN.md` — strict runbook for the *model* that runs a benchmark.
- `BENCHMARK_INSTRUCTIONS.md` — rules for the *model under test* (it is a subject, not an operator: no file edits, no commands, no re-runs).
- `REPORT_SPEC.md` — the reporting contract above, with legacy-run backfill procedure.

## Gotchas

- `bench all` / `bench dev` take ~5–10 min against a real model; run in background, don't poll.
- The dev harness executes model code in a plain `exec` namespace (no network, no filesystem sandbox) — fine for trusted local models, **do not point it at untrusted remote endpoints**.
- Don't kill processes on the machine while benchmarking (llama-server / Unsloth Studio / mock servers may be running).
- Prompt sizes in perf are approximate (~1 token/word filler); actual counts come from server `usage`.

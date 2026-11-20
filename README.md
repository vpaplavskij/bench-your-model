# bench-your-model

Benchmark harness for **local LLM servers** exposing an OpenAI-compatible API
(`POST /v1/chat/completions`, streaming). Three standards share one CLI — and a
single command runs all of them:

## Quick start

```bash
uv sync
uv run bench all --model <name> --endpoint http://localhost:8080/v1
```

That's it. It runs **quality → perf → dev** back-to-back into one run folder.
Re-run it any time — each run appends a new timestamped folder; nothing is
ever overwritten.

> **Required settings (see [REPORT_SPEC.md](REPORT_SPEC.md)):** every run must
> record how the *server* serves the model — pass `--quant` (the GGUF quant),
> `--reasoning-level`, `--reasoning-budget` (`-1` unlimited / `0` no thinking)
> and `--preserve-thinking`. Read them from the Unsloth Studio / llama.cpp
> startup of the server under test — **not** from the agent harness running the
> benchmark. Without `--quant` the results folder can't be compared to any
> other run, and the harness warns about it on stderr.

Individual standards:

```bash
uv run bench quality ...   # accuracy harness (structured + open/LLM-judge)
uv run bench perf ...      # speed benchmark (prefill/decode, TTFT, bottlenecks)
uv run bench dev ...       # developer/coding tasks (accuracy, errors, total time)
```

(`bench ...` with no subcommand defaults to `quality`.)

### Smoke-test without a model: `bench mock`

A stdlib-only mock OpenAI server is built in — handy to verify the harness
plumbing (SSE parsing, scoring, result folders) before pointing it at a real
model:

```bash
uv run bench mock --port 8090 --model mock-model &     # start mock in background
uv run bench all --model mock-model --endpoint http://127.0.0.1:8090/v1
```

The mock streams canned, checker-passing answers for every task (structured,
open/judge, dev) plus fixed-length token streams for the perf sweep.

> **Models under test:** read [`BENCHMARK_INSTRUCTIONS.md`](BENCHMARK_INSTRUCTIONS.md)
> before running. It is the strict, self-contained instruction set a model follows
> when it runs the benchmark against itself — and it forbids modifying anything.

---

## Where results go (one folder per model+quant)

Every run lands in a **per-model folder** under `--root` (default `./results`).
When you pass `--quant` (required by [REPORT_SPEC.md](REPORT_SPEC.md)), the
quant is part of the folder key — **one quant, one folder**:

```
results/
  <model>-<QUANT>/                 # one folder per model+quant pair
    2026-11-09T15-30-12Z/          # one subfolder per run (UTC timestamp)
      quality.json / perf.json / dev.json
      quality.md   / perf.md   / dev.md
```

So you can benchmark many models side by side: `results/model-a-IQ3_S/`,
`results/model-b-Q4_K_M/`, … The same model at two quants is **two folders** —
never mix quantizations into one folder (that's what made the old Qwen runs
uninterpretable). Re-running the same model+quant **appends** a new run —
nothing is ever overwritten.

---

## Standard 1 — Speed (`bench perf`)

Measures **where the time goes** so you can find the bottleneck.

### Protocol
For each prompt size in `--prompt-sizes` (default `128,512,1024,2048`):
1. warmup run(s) (discarded — warms server caches / KV pool),
2. a settle pause,
3. N measurement runs with a **fixed output length** (`--output-tokens`,
   default 256), so prefill and decode phases are cleanly separated.

### Metrics
| Metric | Meaning |
|---|---|
| **TTFT** (ms) | time to first **content/tool** chunk (`ttft_ms`); first token incl. thinking is also recorded as `ttft_first_token_ms` and used for prefill |
| **Prefill speed** (tok/s) | prompt_tokens / first-token time (incl. thinking) — how fast the prompt is processed (measured prompts get a cache-busting nonce); independent of whether any answer tokens arrived |
| **Decode speed** (tok/s) | (N−1) / (total − TTFT) over **content chunks only**; `decode_tps_all` is the thinking-inclusive speed over reasoning + content chunks (the real generation rate on thinking servers) |
| **ITL mean / p95 / max** (ms) | inter-token latency: stability of generation; `*_all` variants cover the reasoning phase too |
| **Stalls** (>500 ms gaps) | server hiccups (CPU offload, page faults, GC, OOM); `stalls_all` counts gaps incl. thinking |
| **Total wall time** (ms) | full request time per run |

### Bottleneck diagnosis (automatic)
- **prefill-dominated vs decode-dominated** — which phase owns the wall time.
- **KV/context pressure** — does decode speed drop as prompt grows?
- **ITL spikes / stalls** — steady generation or server-side hiccups.
- per-size detail line for every measured point.

### Run it
```bash
uv run bench perf --model <name> --endpoint http://localhost:8080/v1 \
    --prompt-sizes 128,512,1024,2048 --output-tokens 256 \
    --measure-runs 3 --settle-s 2
```

Output: `results/<model>-<QUANT>/<run-ts>/perf.json` + `perf.md`.

---

## Standard 2 — Quality (`bench quality`)

| Group | Metric | How |
|---|---|---|
| **Latency** | TTFT (time to first token) | streamed request, perf_counter at first non-empty content/tool `delta` |
| **Throughput** | Output speed (tokens/s) | `completion_tokens / total wall time`; when the server omits `usage`, chunk count is used as an approximate completion-token count so speed is never silently lost |
| **Request size** | prompt tokens, completion tokens | server-reported `usage` (completion falls back to stream-chunk count if absent) |
| **Reliability** | success rate, finish reasons, errors | per-task request record |
| **Quality — structured** | accuracy (exact match on JSON fields / tool calls) | deterministic checker per task |
| **Quality — open-ended** | 0-1 rubric scores per dimension | LLM judge (`--judge-model`, defaults to the model under test) |

### Task suites
- `structured` (deterministic): arithmetic, multi-step arithmetic, single &
  multi tool-calls, list extraction — scored exactly.
- `agent` (multi-turn): tool-loop tasks driven by the harness against a
  scripted fake environment — scored on the final answer and env state.
- `open` (LLM-judged): summarization faithfulness/brevity, strict formatting,
  tone/instruction following.

Add tasks in `src/bench/tasks.py` — a task is just a dict.

### Run it
```bash
uv run bench quality --model <name> --endpoint http://localhost:8080/v1 --suite all
```

Useful flags: `--timeout 300`, `--max-tokens 4096`,
`--temperature 0`, `--task s1 --task o2` (subset), `--root results`.

Output: `results/<model>-<QUANT>/<run-ts>/quality.json` + `quality.md`.

---

## Standard 3 — Developer / coding (`bench dev`)

Answers "which model is best for real developer work?" with **exact,
reproducible** scoring (no LLM judge, no prose flakiness). Each task is a
self-contained coding problem with a deterministic checker that runs the
model's code and tests its behavior.

### Task kinds (each maps to a real dev skill)
- `code` — implement a function that passes hidden unit tests
- `debug` — fix a buggy implementation so it passes
- `refactor` — rewrite under a behavior-preserving constraint
- `multi` — implement several related functions / a small module

### Metrics
| Metric | Meaning |
|---|---|
| **accuracy** (pass rate) | fraction of tasks whose code passes all checks |
| **runtime errors** | model code that failed to import/run/execute (with the error) |
| **truncated** | answers cut off at `max_tokens` (common with thinking models); detail says so, score unchanged |
| **request failures** | server-side failures during the task |
| **total execution time** | wall time per task (request + check), mean/median/max |
| **accuracy by skill** | pass rate broken down by task kind — shows where a model is strong/weak |
| **composite score** (0-100) | accuracy × 100, minus penalties for runtime errors & request failures → ranks models for dev suitability |

### Run it
```bash
uv run bench dev --model <name> --endpoint http://localhost:8080/v1
```
Flags: `--task d1` (subset), `--max-tokens`, `--timeout`. (`--temperature` is
quality-only — dev requests always use temperature 0.)

Output: `results/<model>-<QUANT>/<run-ts>/dev.json` + `dev.md` (includes a per-task
**error log** of exactly what went wrong during execution).

To compare models, run each with the same flags and compare `composite_score`
(and the by-skill accuracy) across their `dev.md` files.

Add tasks in `src/bench/devtasks.py` — a task is a dict with a `prompt` and a
`checker_src` string defining `check(code_src, run) -> {"pass": bool, "detail": str}`.

---

## Comparing models side by side

Benchmark each model with the same flags, and **record the server settings**
(`--quant` etc. — see [REPORT_SPEC.md](REPORT_SPEC.md)):

```bash
uv run bench all --model model-a --endpoint http://localhost:8080/v1 \
    --quant Q4_K_M --reasoning-level default --reasoning-budget -1 --preserve-thinking
uv run bench all --model model-b --endpoint http://localhost:8080/v1 \
    --quant IQ3_S --reasoning-level default --reasoning-budget 0 --no-preserve-thinking
```

Each model+quant gets its own folder (`results/model-a-Q4_K_M/`,
`results/model-b-IQ3_S/`). To compare, read the per-run `*.md` files — each is
fully human-readable (headline metrics + per-standard detail in plain language).
The exact server settings used for a run (quant, reasoning level/budget,
preserve-thinking, context length) are recorded in that run's `*.json` under
`meta.settings`. The dev standard's `composite_score` ranks which model best
suits developer work; the perf standard's diagnosis names the bottleneck.
Numbers from different quants or reasoning budgets are *not* directly
comparable — that's why they live in separate folders with their settings
recorded.

---

## Notes
- The speed harness requests **strict** output length via `max_tokens`; the
  quality harness requests strict JSON-schema output where applicable. If your
  server doesn't support `response_format`, structured tasks fall back to
  best-effort JSON extraction (still scored).
- Prompt sizes are approximate (1 token/word filler text); actual prompt token
  counts are taken from the server's `usage` and reported per point.
- The dev harness executes model code in a guarded `exec` namespace — imports
  of filesystem/network/subprocess modules (os, socket, subprocess, shutil,
  pathlib, …) and `open()` are blocked, json/re/collections stay available,
  and a SIGALRM watchdog kills hangs. Fine for benchmarking trusted local
  models; do not point it at untrusted remote endpoints.

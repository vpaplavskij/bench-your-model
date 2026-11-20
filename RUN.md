# How to run this benchmark (for the model — do exactly this)

You are testing a local LLM server with the `bench` harness in this directory.
Do not search, explore, or read source code. Execute one command, wait, then read the report.

## 0. Values you need (ask if missing — do not guess)

**Server identity:**

| Value | What it is |
|---|---|
| **MODEL** | The model name/alias of the server under test (e.g. from Unsloth Studio or the server's `/v1/models`). |
| **ENDPOINT** | Base URL of that server, e.g. `http://127.0.0.1:PORT/v1`. |
| **API KEY** | The key the server expects (from its auth file / Studio). |

**Server settings** — these must come from how the inference server
(Unsloth Studio / llama.cpp) is actually started, **not** from any agent or
harness on this machine. Read them from the server's startup command / Studio
config; ask if you can't find them (see `REPORT_SPEC.md`):

| Value | What it is | Example |
|---|---|---|
| **QUANT** | The GGUF quantization of the model under test. | `IQ3_S`, `Q4_K_M` |
| **REASONING LEVEL** | The reasoning/effort level the server runs at. | `default` |
| **REASONING BUDGET** | Thinking-token budget: `-1` = unlimited, `0` = no thinking. | `-1` |
| **PRESERVE THINKING** | Yes/no — whether the server preserves thinking content between turns. | `yes` |
| **CONTEXT LENGTH** | The context window (tokens) the server runs at, from llama.cpp `-c` / `--ctx-size`. | `84224` |

## 1. Run the benchmark (the only command you need)

From the project root:

```bash
uv run bench all --model <MODEL> --endpoint <ENDPOINT> --api-key <API_KEY> \
  --quant <QUANT> --reasoning-level <REASONING LEVEL> \
  --reasoning-budget <REASONING BUDGET> --preserve-thinking \
  --context-length <CONTEXT LENGTH> > /tmp/bench-run.log 2>&1
```

- Use `--no-preserve-thinking` if the server does not preserve thinking; omit a flag only if it is genuinely unknown.
- Results go to `results/<MODEL>-<QUANT>/`.
- Takes **~5–10 minutes** (depends on the model's speed). Run it in the background (or with a long timeout) and wait for it to finish. Do not interrupt or poll it while it runs.
- It runs three standards back-to-back: **quality → perf → dev**, then writes all result files itself.

## 2. Check the exit code

| Exit code | Action |
|---|---|
| **0** | Success — go to step 3. |
| **Non-zero** | Read the last ~30 lines of `/tmp/bench-run.log` and report them verbatim. Stop; do not try to "fix" anything unless asked. |

## 3. Read the results (only these files)

`<run-folder>` is the newest timestamped folder (e.g. `2026-11-09T22-44-29Z`) —
the harness prints its path when it finishes.

| File | What it is |
|---|---|
| `results/<MODEL>-<QUANT>/<run-folder>/quality.md` | **Read this one.** Per-question detail (which quiz items were right/wrong and why) + the standard's total execution time. |
| `results/<MODEL>-<QUANT>/<run-folder>/perf.md` | **Read this one.** Speed per prompt length + bottleneck diagnosis + total execution time. |
| `results/<MODEL>-<QUANT>/<run-folder>/dev.md` | **Read this one.** Which coding jobs passed/crashed and why + total execution time. |
| `results/<MODEL>-<QUANT>/<run-folder>/compare.md` | **Read this one.** This run vs every previously recorded run (same model, and best comparable run per model+quant). `†` rows used a different suite size; `⚠` rows ran with no thinking. |

The `.json` files next to each `.md` hold the same data in machine form; you need them only for the server settings (`meta.settings`, see step 4) or raw numbers.

## 4. Report back

First state **Model & settings**: read `meta.settings` from any of the run's
`*.json` files (quantization, reasoning level/budget, preserve thinking, context
length) and pair it with the model name and endpoint. Then give one line per
standard (quality / perf / dev) with the headline numbers from each `.md`, then
each standard's **Total execution time** line, plus the run folder path. Finish with
the comparison from `compare.md`: how this run ranks against the recorded runs (speed
and correctness), noting any `†`/`⚠` caveats.

## What NOT to do

- No codebase searching, no reading `src/`, no inspecting tasks or scorers.
- No extra benchmark commands — `bench all` already covers everything.
- Don't kill other processes on this machine (llama-server, Unsloth Studio, mock servers may be running).
- Don't modify any files unless the run fails and the user asks for a fix.

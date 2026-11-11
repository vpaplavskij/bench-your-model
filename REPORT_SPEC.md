# Report specification — model identity & settings

Every benchmark report must state **what exactly was tested and how the server
was serving it**. A run whose quantization or reasoning configuration is not
recorded is *not comparable* to any other run, and its numbers are nearly
worthless (see `ANALYSIS.md`: nine past runs formed a 2-quants ×
2-thinking-modes matrix that was indistinguishable in the reports).

This document is the contract. The harness enforces it mechanically:
`--quant` is required by convention and every run document stores the settings
block in its `meta`. If you can't fill a field in, **ask** — never guess.

## 1. Required settings (per model under test)

| Field | Flag | Example | Where to read it |
|---|---|---|---|
| Quantization | `--quant` | `IQ3_S`, `IQ3_XXS`, `Q4_K_M` | The GGUF filename / Unsloth Studio model card / llama.cpp `-m` path. **Not** from the agent harness. |
| Reasoning level | `--reasoning-level` | `default`, `low`, `medium`, `high` | The server's reasoning/effort setting (Unsloth Studio config or llama.cpp flag) |
| Reasoning budget | `--reasoning-budget` | `-1` (unlimited), `0` (no thinking), or a token count | The server's thinking-token budget flag (`--reasoning-budget` / equivalent). This is the knob that actually changes output length. |
| Preserve thinking | `--preserve-thinking` / `--no-preserve-thinking` | `true` / `false` | Whether the server keeps thinking content in context between turns. Omit the flag if you truly don't know — it then records `not recorded`. |
| Context length | `--context-length` | e.g. `8192`, `84224` (tokens) | The context window the server runs at (llama.cpp `-c` / `--ctx-size`). Omit if unknown — it then records `not recorded`. |

**Source of truth: the inference server, not the agent.** The model is served
by Unsloth Studio / llama.cpp; its quant and reasoning configuration come from
that startup (the command that launched the server, or the Studio config for
the running instance). Do **not** copy these values from OpenCode or any other
agent harness — those settings describe a different system.

`--model` remains the server's actual model id (e.g. `unsloth/Qwen3.8-27B-GGUF`)
because the harness sanity-checks it against `/v1/models`. The quant is a
separate reporting dimension; it is recorded in reports but never sent in
requests.

## 2. Results folder naming — one quant, one folder

```
results/<MODEL>-<QUANT>/
```

- `<MODEL>` is the model id as passed to `--model`, with `/` sanitized to `_`
  (e.g. model `unsloth/Qwen3.8-27B-GGUF`, quant `IQ3_S` →
  `results/unsloth_Qwen3.8-27B-GGUF-IQ3_S/`).
- **One folder per quant.** The same model at two different quants lives in
  two folders and must never be mixed into one — that is exactly the failure
  mode from `ANALYSIS.md`.
- If you don't know the quant, the harness still writes to `results/<MODEL>/`
  (no suffix) and warns on stderr; backfill per §6 afterwards.

Folder layout (unchanged):

```
results/unsloth_Qwen3.8-27B-GGUF-IQ3_S/
  2026-11-09T15-30-12Z/     # one subfolder per run (UTC timestamp)
    quality.json / perf.json / dev.json
    quality.md   / perf.md   / dev.md
```

## 3. Where the settings are recorded

Every new run records the settings block in its **run documents** —
`quality.json` / `perf.json` / `dev.json` each carry `meta.settings`:

```json
"settings": {
  "quant": "IQ3_S",
  "reasoning_level": "default",
  "reasoning_budget": -1,
  "preserve_thinking": true,
  "context_length": 84224
}
```

A field you omitted on the CLI is stored as `null` and rendered as
`not recorded`.

## 4. What each run document must state

Each standard's markdown (`quality.md`, `perf.md`, `dev.md`) starts with the
model (server id), endpoint and date, and states a **Total execution time**
line for its own standard:

- `quality.md` / `dev.md`: exact sum of the measured per-task `total_ms`
  values in that document.
- `perf.md`: points store only aggregates, so its share is estimated as
  `sum over points of total_ms.mean × runs_ok`.

The settings themselves live in each document's `meta.settings` (§3) — read
them from the `.json` files when you need to state what was tested.

## 5. Validation checklist (before calling a run "recorded")

- [ ] `--quant` was passed (or you deliberately backfilled it).
- [ ] The results folder name ends in `-<QUANT>` matching that flag.
- [ ] Each run's `*.json` has `meta.settings` with all five keys present
      (`null` allowed only for genuinely unknown values).
- [ ] Each standard's `*.md` states its own **Total execution time** line.
- [ ] Quant/reasoning values match the **server's** startup (Unsloth Studio /
      llama.cpp), not the agent harness that ran the benchmark.

## 6. Backfilling legacy runs (no settings recorded)

Old folders (e.g. `results/unsloth_Qwen3.8-27B-GGUF/`) predate this spec and
contain no settings. To bring them into compliance:

1. **Determine the truth** from the server logs / Studio history for each run
   folder (quant, reasoning budget, preserve_thinking). If a run's config can't
   be recovered, leave it — its `meta.settings` will show `null`, which is
   honest.
2. **Split the folder by quant** if runs mixed quants: copy each run's
   `<ts>/` subfolder into `results/<MODEL>-<QUANT>/<ts>/`.
3. **Backfill settings** in each run document: set `meta.settings` in every
   `<ts>/*.json` (all five keys, `null` only for genuinely unknown values).
4. Verify with the checklist in §5.

## 7. Command examples

```sh
# Full benchmark of one model+quant (settings read from the SERVER startup)
uv run bench all \
  --model unsloth/Qwen3.8-27B-GGUF \
  --endpoint http://127.0.0.1:8080/v1 \
  --quant IQ3_S \
  --reasoning-level default \
  --reasoning-budget -1 \
  --preserve-thinking
```

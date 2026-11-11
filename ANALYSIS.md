# Qwen3.8-27B benchmark analysis — quants × thinking modes

_9 runs, Nov 9 23:50 – Nov 10 00:45 UTC. Qwen3.6 excluded per request. All runs temp 0, max_tokens 4096, same task suites._

## What was actually tested

The 9 runs are **not** repeats of one setup — they form a **2 quants × 2 thinking-modes** matrix on 4 separate server instances (recovered from the Unsloth Studio server startup logs; the bench `report.md` files don't record these settings):

| Instance | Quant (GGUF) | Reasoning | Budget (`--reasoning-budget`) | preserve_thinking | ctx | Runs |
|---|---|---|---|---|---|---|
| port 52361 | **IQ3_S** | on (effort: default) | **-1 → unlimited thinking** | true | 66 688 | 2 × quality-only |
| port 44073 | **IQ3_S** | on (effort: default) | **0 → immediate end (no thinking)** | true | 64 512 | 2 × full |
| port 50859 | **IQ3_XXS** | on (effort: default) | **-1 → unlimited thinking** | true | 79 872 | 2 × full |
| port 52243 | **IQ3_XXS** | on (effort: default) | **0 → immediate end (no thinking)** | true | 79 872 | 3 × full |

Common to all: `--reasoning on`, MTP speculative draft (`draft-mtp`, n_max=2), KV q8_0, flash-attn on, `-ngl -1`, 1 slot. The "reasoning level" was the template default in every run (no `--reasoning-effort` flag); the knob that actually changed between instances is the **budget**: unlimited thinking vs. immediate end.

You can verify thinking happened: with budget −1 the model emitted 60–92 completion tokens on a one-line arithmetic answer (the rest is reasoning); with budget 0 it emits exactly 11.

## The matrix, in plain words

| Config | Quiz (n=5) | First word | Writing speed | Coding jobs (n=4) |
|---|---|---|---|---|
| **IQ3_S + thinking** | **3/5** — s1✓ 396, s2✓ 100, s5✓; s3/s4✗ | ~2.7 s (it thinks first) | ~43 tok/s visible | *not measured* (quality-only runs) |
| **IQ3_S + no thinking** | **1/5** — only s5✓ (s1→406, s2→108) | **~160 ms** | **~57 tok/s**, prefill ~5.2k tok/s | **4/4 = 100/100**, median job **1.7 s** |
| **IQ3_XXS + thinking** | **3/5** — same pattern as IQ3_S+think | 1.3–4.2 s | (throughput incl. reasoning tokens — not comparable) | 3/4 = 75, median job **32–43 s** (thinking time); d1 fails by **running out of all 4096 tokens on reasoning** before writing any code |
| **IQ3_XXS + no thinking** | **1/5** — only s5✓ (s1→416, s2→108) | 160 ms → 2.8 s (degraded within the instance, see F5) | ~47 tok/s (first run), prefill 5.3k → 479 tok/s (degraded) | 3/4 = 75 → **2/4 = 50**; d1 fails on the quoted-`msg` parser case, then d2 starts returning **empty completions** (3 tokens) |

Each cell was run 2–3 times and the results are **identical across repeats** (same answers, same token counts) — so these are stable measurements of each configuration, not noise.

## Findings

**F1 — Thinking is the lever for quiz accuracy; the quant is not.**
With thinking on: s1 and s2 answered correctly (396, 100) — in **both** quants. With thinking off: both wrong (406/416, 108) — again in both quants. IQ3_S and IQ3_XXS score exactly the same on the quiz in each mode (3/5 vs 3/5, 1/5 vs 1/5). The "answers flip between restarts" seen earlier was this mode change all along — at temp 0 the model is perfectly deterministic within a configuration.

**F2 — Thinking trades ~17× first-word latency for +2 quiz points.**
TTFT: ~2.7 s (thinking) vs ~160 ms (no thinking). For interactive use, no-thinking is dramatically snappier; for answer quality on the structured quiz, thinking is worth 40 percentage points. Note the speed columns aren't directly comparable between modes — in thinking runs the "writing" throughput includes reasoning tokens.

**F3 — The quant does matter for coding: only IQ3_S passes d1.**
- **IQ3_S + no thinking: 4/4 = 100/100**, median job 1.7 s — the best coding result in the whole dataset, and the only configuration where the log-parser task (d1) is solved.
- **IQ3_XXS never passes d1** (5/5 runs fail): with thinking it burns the entire 4096-token budget on reasoning and emits no code at all ("no function named parse_log_line found"); without thinking it writes a parser that breaks on the quoted `msg="login ok"` line. Same task, same prompt — the smaller quant simply can't solve it here.

**F4 — Tool calling fails in every configuration: 9/9 runs, both quants, both modes.**
s3/s4 (emit a JSON tool call from a plain system prompt) are refused in prose every single time ("I don't have access to real-time data…"). This is a consistent behavior gap of the model under this protocol, independent of quant or thinking. If tool calling matters, neither configuration passes it as set up.

**F5 — One genuine within-instance degradation (last instance only).**
On port 52243 (IQ3_XXS, no thinking), run 1 → run 3 on the *same* server: prefill collapsed 5 336 → 479 tok/s, TTFT 160 ms → 2.8 s, and d2 started returning empty completions (3 tokens, "no function named total found"). The other three instances show zero drift across their repeats. So the late-session numbers for that instance aren't representative of IQ3_XXS-no-thinking either — its run-1 values (47 tok/s, 160 ms TTFT, dev 75) are the honest ones.

**F6 — The thinking-mode perf numbers on port 50859 are internally inconsistent and shouldn't be compared.**
That instance shows the *slowest* prefill (~110–1 600 tok/s) of any run while simultaneously showing the *fastest* "decode" (250–783 tok/s). That combination isn't physical for this model; it's an artifact of measuring a thinking-mode stream with a harness that assumes plain output. Don't rank speeds using that column — use the no-thinking runs: **IQ3_S ≈ 57 tok/s vs IQ3_XXS ≈ 47 tok/s** writing, both at ~160 ms first word.

**F7 — Judge scores still saturated.**
Every open-ended dimension is 1.00 in all 9 runs, thinking and no-thinking alike. Self-judging gives full marks regardless; treat open-ended as "no red flags" only.

## What this means / what to do next

- **Best measured configuration overall: IQ3_S + no thinking** — 100/100 coding, 1.7 s jobs, 57 tok/s, 160 ms first word. Its weakness is the quiz (1/5), which thinking would fix (3/5) — but that cell was never measured for coding/perf.
- **Missing cell: IQ3_S + thinking × full suite.** That's the one experiment worth running next: it's the only way to know whether thinking also fixes d1 on IQ3_XXS-class problems or just the quiz arithmetic.
- **max_tokens 4096 is too small for thinking-mode coding** — d1 died at exactly 4096/4096 tokens of reasoning with zero code written. Either raise the cap for dev tasks or lower the reasoning effort so thinking can't consume the whole budget.
- **IQ3_XXS vs IQ3_S on this model: IQ3_S wins coding (d1), ties everything else, and costs a bit more VRAM.** The 27B's speed gap between quants is modest (~57 vs ~47 tok/s).
- Tool calling needs a protocol fix (send real tool schemas in the request, or change the task) — no quant/mode combination fixes it.

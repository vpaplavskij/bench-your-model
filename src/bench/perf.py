"""Speed benchmark: prefill / decode throughput, TTFT, stability, bottlenecks.

Protocol (per prompt size P in --prompt-sizes):

1. Warmup run at P (discarded) — warms caches/KV pool on the server.
2. Measurement runs (default 3) at P with a FIXED output length
   (--output-tokens), so prefill and decode phases are cleanly separated:

     * TTFT          = time to first token        -> prefill-bound phase
     * prefill tok/s = P / TTFT                   -> prompt-processing speed
     * decode tok/s  = (N-1) / (T_total - TTFT)   -> autoregressive speed
     * ITL mean/p95/max, stall count             -> stability of generation
     * total wall time per run

Bottleneck heuristics (see :func:`diagnose`):
     * prefill-dominated  -> prompt processing / memory bandwidth on long input
     * decode-dominated   -> KV-cache read / attention per generated token
     * ITL growth vs P    -> context/KV pressure (grows with prompt size)
     * stalls             -> server hiccup: CPU offload, page faults, GC, OOM
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from .client import BenchClient
from .humanize import fmt_ms, fmt_tps, glossary_md

# Repeated text that is short per word but long overall (tokenizes densely).
_FILLER = ("The quick brown fox jumps over the lazy dog near the river bank while "
           "the sun sets behind the hills and the wind carries leaves across the "
           "open field toward the distant town where people gather to share stories "
           "of old times and new beginnings under a wide and quiet sky. ")


def _prompt_of_size(target_tokens: int) -> tuple[str, int]:
    """Build a prompt of ~target_tokens tokens (approx: 1 token/word).

    The instruction asks for a long continuation so the model keeps generating
    until the max_tokens cap; a "one short sentence" prompt stops ~20 tokens in
    and decode speed would be measured over a tiny window.
    """
    words = target_tokens
    text = _FILLER * (words // len(_FILLER.split()) + 1)
    return text[: words * 7] + (" Continue the passage above with a long, detailed, "
                                "fictional narrative of many paragraphs."), words


def warmup(client: BenchClient, prompt: str, max_tokens: int) -> None:
    """One untimed request to warm server-side caches/KV pool."""
    client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


@dataclass
class PerfRun:
    """One timed request at a given prompt size."""

    ok: bool
    error: str | None = None
    ttft_ms: float | None = None
    ttft_first_token_ms: float | None = None  # incl. reasoning chunks; prefill basis
    total_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    itl_mean_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_max_ms: float | None = None
    stalls: int = 0
    decode_tps: float | None = None
    prefill_tps: float | None = None


@dataclass
class PerfPoint:
    """Aggregated results for one prompt size."""

    prompt_size: int                      # requested ~tokens
    actual_prompt_tokens: int | None = None
    runs: list[PerfRun] = field(default_factory=list)

    @property
    def ok_runs(self) -> list[PerfRun]:
        return [r for r in self.runs if r.ok]


def _time_request(client: BenchClient, prompt: str, max_tokens: int) -> PerfRun:
    # Nonce prefix defeats llama.cpp's prompt-prefix KV cache reuse: without it
    # every measured run replays the warmup's cached prefill and prefill tok/s
    # is measured on a cache hit, not on real prompt processing.
    res = client.chat([{"role": "user", "content": uuid4().hex[:12] + " " + prompt}],
                      max_tokens=max_tokens)
    run = PerfRun(ok=res.ok, error=res.error)
    if not res.ok:
        return run
    sm = res.stream_metrics
    run.ttft_ms = res.ttft_ms
    run.ttft_first_token_ms = res.ttft_first_token_ms
    run.total_ms = res.total_ms
    run.prompt_tokens = res.prompt_tokens
    run.completion_tokens = res.completion_tokens
    if sm is not None:
        run.itl_mean_ms = sm.inter_token_ms()
        run.itl_p95_ms = sm.p95_itl_ms()
        run.itl_max_ms = sm.max_itl_ms()
        run.stalls = sm.stall_count_ms()
    # Derived speeds (guard: need >=2 tokens for a decode span).
    # Decode counts CONTENT chunks only — usage completion_tokens include
    # thinking tokens and would inflate decode_tps on thinking servers.
    # Prefill uses time to the FIRST token (reasoning included): on thinking
    # servers ttft_ms is time to first ANSWER token, which understates prefill.
    prefill_ttft = res.ttft_first_token_ms or res.ttft_ms
    if res.ttft_ms and res.total_ms > res.ttft_ms:
        n = res.stream_chunks
        if n >= 2:
            run.decode_tps = (n - 1) / ((res.total_ms - res.ttft_ms) / 1000.0)
        if res.prompt_tokens and prefill_ttft:
            run.prefill_tps = res.prompt_tokens / (prefill_ttft / 1000.0)
    return run


def run_perf_sweep(
    cfg,
    prompt_sizes: list[int] | None = None,
    output_tokens: int = 256,
    warmup_runs: int = 1,
    measure_runs: int = 3,
    settle_s: float = 2.0,
) -> dict:
    """Run the full speed sweep and return the result document."""
    prompt_sizes = prompt_sizes or [128, 512, 1024, 2048]
    client = BenchClient(
        endpoint=cfg.endpoint,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        max_tokens=output_tokens,
        temperature=0.0,
    )

    server_info: dict = {"endpoint": cfg.endpoint, "model_available": False}
    try:
        available = client.list_models()
        server_info["available_models"] = available
        server_info["model_available"] = cfg.model in available
    except Exception as e:
        server_info["error"] = f"{type(e).__name__}: {e}"

    points: list[PerfPoint] = []
    for size in prompt_sizes:
        prompt, _ = _prompt_of_size(size)
        point = PerfPoint(prompt_size=size)
        print(f"  prompt ~{size} tok | warmup {warmup_runs}x + measure {measure_runs}x "
              f"(output capped at {output_tokens})")
        for _ in range(warmup_runs):
            warmup(client, prompt, output_tokens)
        time.sleep(settle_s)  # let server settle between sizes
        for i in range(measure_runs):
            r = _time_request(client, prompt, output_tokens)
            point.runs.append(r)
            status = "ok " if r.ok else "ERR"
            print(f"    [{status}] ttft={r.ttft_ms and round(r.ttft_ms)}ms "
                  f"total={round(r.total_ms)}ms prefill={r.prefill_tps and round(r.prefill_tps)}t/s "
                  f"decode={r.decode_tps and round(r.decode_tps)}t/s "
                  f"itl_mean={r.itl_mean_ms and round(r.itl_mean_ms)}ms")
            time.sleep(0.5)
        if point.ok_runs:
            # Some servers never report `usage` in streamed responses; don't
            # crash — just leave actual_prompt_tokens as None.
            vals = [r.prompt_tokens for r in point.ok_runs if r.prompt_tokens]
            point.actual_prompt_tokens = max(vals) if vals else None
        points.append(point)

    client.close()
    return _assemble(cfg, server_info, points, output_tokens)


def _med(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 1) if vals else None


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 1) if vals else None


def diagnose(points: list[PerfPoint]) -> list[str]:
    """Heuristic bottleneck diagnosis. Returns human-readable findings."""
    out: list[str] = []
    ok_pts = [p for p in points if p.ok_runs]
    if not ok_pts:
        return ["no successful runs — nothing to diagnose"]

    # 1) prefill vs decode share of total time (median run per point, averaged)
    shares: list[float] = []
    for p in ok_pts:
        totals = sorted((r.total_ms or 0) for r in p.ok_runs)
        r = p.ok_runs[totals.index(totals[len(totals) // 2])]  # median run by total_ms
        if r.ttft_ms and r.total_ms > r.ttft_ms:
            shares.append(r.ttft_ms / r.total_ms)
    if shares:
        avg_share = statistics.mean(shares)
        if avg_share > 0.6:
            out.append(
                f"PREFILL-DOMINATED ({avg_share:.0%} of wall time is first-token): "
                "the model spends most of its time processing the prompt — "
                "long-context memory bandwidth / KV write is the bottleneck. "
                "Shorter prompts or faster prompt processing will help most."
            )
        elif avg_share < 0.25:
            out.append(
                f"DECODE-DOMINATED ({1 - avg_share:.0%} of wall time is token generation): "
                "bottleneck is per-token cost (KV-cache reads / attention over context). "
                "This is the normal regime for long outputs."
            )
        else:
            out.append(
                f"MIXED ({avg_share:.0%} prefill / {1 - avg_share:.0%} decode): "
                "neither phase dominates strongly at these sizes."
            )

    # 2) decode speed vs prompt size -> KV/context pressure
    if len(ok_pts) >= 2:
        first, last = ok_pts[0], ok_pts[-1]
        d_first = _med([r.decode_tps for r in first.ok_runs])
        d_last = _med([r.decode_tps for r in last.ok_runs])
        if d_first and d_last and d_last < d_first * 0.85:
            out.append(
                f"KV/CONTEXT PRESSURE: decode speed drops "
                f"{d_first:.0f} -> {d_last:.0f} tok/s as prompt grows "
                f"({first.prompt_size} -> {last.prompt_size} tok). "
                "Attention over a growing KV cache is costing per-token latency — "
                "expect it to worsen with longer contexts."
            )
        elif d_first and d_last:
            out.append(
                f"decode speed stable across prompt sizes "
                f"({d_first:.0f} -> {d_last:.0f} tok/s) — no strong KV pressure at these lengths."
            )

    # 3) ITL spikes / stalls -> server-side hiccups
    worst_stalls = max((max((r.stalls for r in p.ok_runs), default=0) for p in ok_pts), default=0)
    p95s = [_med([r.itl_p95_ms for r in p.ok_runs]) for p in ok_pts]
    p95s = [v for v in p95s if v is not None]
    means = [_med([r.itl_mean_ms for r in p.ok_runs]) for p in ok_pts]
    means = [v for v in means if v is not None]
    if worst_stalls > 0:
        out.append(
            f"STALLS DETECTED: {worst_stalls} inter-token gap(s) > 500ms in at least one run — "
            "server hiccup (CPU offload, page fault, GC, or memory pressure). "
            "Check server logs / GPU utilization during the run."
        )
    elif p95s and means and statistics.mean(p95s) > 2.5 * statistics.mean(means):
        out.append(
            f"ITL SPIKES: p95 ITL ({statistics.mean(p95s):.0f}ms) is "
            f">2.5x the mean ITL ({statistics.mean(means):.0f}ms) — "
            "generation is mostly steady but has occasional spikes."
        )
    else:
        out.append("ITL stable: no stalls > 500ms and p95 within ~2.5x of the mean — smooth generation.")

    # Per-size detail is already in the "Per prompt size" table (render_perf_markdown),
    # so it is not repeated here.
    return out


def _assemble(cfg, server_info: dict, points: list[PerfPoint], output_tokens: int) -> dict:
    all_ok = [r for p in points for r in p.ok_runs]
    doc_points = []
    for p in points:
        ok = p.ok_runs
        doc_points.append({
            "prompt_size": p.prompt_size,
            "actual_prompt_tokens": p.actual_prompt_tokens,
            "runs_ok": len(ok),
            "runs_total": len(p.runs),
            "errors": [r.error for r in p.runs if not r.ok],
            "ttft_ms": {"mean": _mean([r.ttft_ms for r in ok]), "median": _med([r.ttft_ms for r in ok])},
            "ttft_first_token_ms": {"mean": _mean([r.ttft_first_token_ms for r in ok]), "median": _med([r.ttft_first_token_ms for r in ok])},
            "total_ms": {"mean": _mean([r.total_ms for r in ok]), "median": _med([r.total_ms for r in ok])},
            "prefill_tps": {"mean": _mean([r.prefill_tps for r in ok]), "median": _med([r.prefill_tps for r in ok])},
            "decode_tps": {"mean": _mean([r.decode_tps for r in ok]), "median": _med([r.decode_tps for r in ok])},
            "itl_mean_ms": {"mean": _mean([r.itl_mean_ms for r in ok]), "median": _med([r.itl_mean_ms for r in ok])},
            "itl_p95_ms": _med([r.itl_p95_ms for r in ok]),
            "itl_max_ms": max((r.itl_max_ms for r in ok if r.itl_max_ms is not None), default=None),
            "stalls_total": sum(r.stalls for r in ok),
        })

    ttft_all = _med([r.ttft_ms for r in all_ok])
    decode_all = _med([r.decode_tps for r in all_ok])
    prefill_all = _med([r.prefill_tps for r in all_ok])

    return {
        "meta": {
            "kind": "speed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": cfg.model,
            "endpoint": cfg.endpoint,
            "output_tokens": output_tokens,
            "measure_runs": len(points[0].runs) if points else 0,
            # How the SERVER serves the model (Unsloth Studio / llama.cpp), not
            # the agent harness. See REPORT_SPEC.md.
            "settings": cfg.settings(),
        },
        "server": server_info,
        "points": doc_points,
        "headline": {
            "ttft_ms_median": ttft_all,
            "decode_tps_median": decode_all,
            "prefill_tps_median": prefill_all,
        },
        "diagnosis": diagnose(points),
    }


def render_perf_markdown(doc: dict) -> str:
    m = doc["meta"]
    h = doc["headline"]

    plain: list[str] = []
    if h.get("prefill_tps_median") is not None:
        plain.append(f"- It **reads** your question at about **{fmt_tps(h['prefill_tps_median'])}** "
                     f"and **writes** the answer at about **{fmt_tps(h['decode_tps_median'])}**.")
    else:
        plain.append("- Read speed could not be measured (server did not report prompt size).")
    if h.get("ttft_ms_median") is not None:
        plain.append(f"- First word arrives after **{fmt_ms(h['ttft_ms_median'])}** "
                     f"(median across all test lengths).")
    plain.append(f"- Every answer was capped at {m['output_tokens']} tokens; "
                 f"each size tested {m['measure_runs']} times (plus 1 warm-up, discarded).")
    _p_total = sum(
        float((pt.get("total_ms") or {}).get("mean")) * int(pt.get("runs_ok") or 0)
        for pt in doc["points"] if (pt.get("total_ms") or {}).get("mean") is not None and pt.get("runs_ok")
    )
    if _p_total > 0:
        plain.append(f"- **Total execution time** (measured requests, all sizes): "
                      f"**{fmt_ms(_p_total)}**.")

    lines = [
        f"# Speed benchmark — `{m['model']}`",
        "",
        f"_endpoint: `{m['endpoint']}`_",
        "",
        "## In plain words",
        *plain,
        "",
        "## Numbers (median of runs)",
        "",
        "| question length | first word (TTFT) | read speed (prefill) | write speed (decode) "
        "| word gap avg / p95 | total time | pauses ≥ ½ s |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in doc["points"]:
        # The filler is ~1 token/word, so show the server's real prompt count when known.
        size = (f"{p['actual_prompt_tokens']} tok" if p["actual_prompt_tokens"]
                else f"~{p['prompt_size']} words")
        lines.append(
            f"| {size} | {fmt_ms(p['ttft_ms']['median'])} "
            f"| {fmt_tps(p['prefill_tps']['median'])} | {fmt_tps(p['decode_tps']['median'])} "
            f"| {fmt_ms(p['itl_mean_ms']['median'])} / {fmt_ms(p['itl_p95_ms'])} "
            f"| {fmt_ms(p['total_ms']['median'])} | {p['stalls_total']} |"
        )
    lines += ["", "## What's slowing it down (diagnosis)", ""]
    lines += [f"- {d}" for d in doc["diagnosis"]]
    lines += glossary_md()
    return "\n".join(lines) + "\n"

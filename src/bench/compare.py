"""Cross-run comparison, written into the run folder after `bench all`.

Scans every recorded run under <root> (the .json files already carry settings,
summaries, perf headline and diagnosis) and renders two tables: all runs of
this model+quant, and the best comparable run per model+quant folder.

Comparability rules:
* Suite sizes grew over history (quiz 9→11 items, dev 9→13 jobs). Rows whose
  suite differs from the current run are marked † and excluded from the ranking.
* Budget-0 runs (no thinking) are marked ⚠ — their speeds are not comparable
  with thinking runs, and self-inflated decode (usage counts reasoning tokens)
  is noted in the header.
"""

from __future__ import annotations

import json
from pathlib import Path

from .humanize import fmt_ms, fmt_tps

_HDR = "| run | settings | quiz | judge | dev | prefill / decode | TTFT | bottleneck | total |"
_SEP = "|---|---|---|---|---|---|---|---|---|"


def _load_run(run: Path) -> dict | None:
    docs: dict[str, dict] = {}
    for std in ("quality", "perf", "dev"):
        p = run / f"{std}.json"
        if p.exists():
            try:
                docs[std] = json.loads(p.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
    if not docs:
        return None
    first = docs.get("quality") or docs.get("perf") or docs.get("dev")
    st = first.get("meta", {}).get("settings") or {}
    q, p, d = docs.get("quality"), docs.get("perf"), docs.get("dev")
    qs = (q or {}).get("summary") or {}
    ds = (d or {}).get("summary") or {}
    h = (p or {}).get("headline") or {}
    st_sum = qs.get("structured") or {}

    total_ms = sum(t.get("total_ms") or 0.0 for t in (q or {}).get("tasks", []))
    total_ms += sum(t.get("total_ms") or 0.0 for t in (d or {}).get("tasks", []))
    total_ms += sum((pt.get("total_ms") or {}).get("mean") or 0.0
                    * (pt.get("runs_ok") or 0) for pt in (p or {}).get("points", []))

    judge = qs.get("judge_dimension_means") or {}
    diag = ((p or {}).get("diagnosis") or [""])[0].split(":")[0]
    return {
        "folder": run.parent.name,
        "ts": run.name,
        "model": first.get("meta", {}).get("model", ""),
        "settings": f"{st.get('quant') or '?'} · {st.get('reasoning_level') or '?'}"
                    f"/{st.get('reasoning_budget') if st.get('reasoning_budget') is not None else '?'}"
                    f" · {st.get('context_length') or '?'} tok",
        "quiz": f"{st_sum.get('correct', '—')}/{st_sum.get('n', '—')}",
        "n_quiz": st_sum.get("n"),
        "judge": f"{min(judge.values()):.2f}" if judge else "—",
        "dev": f"{ds.get('passed', '—')}/{ds.get('n_tasks', '—')} · {ds.get('composite_score', '—')}",
        "n_dev": ds.get("n_tasks"),
        "composite": ds.get("composite_score"),
        "perf": f"{fmt_tps(h.get('prefill_tps_median'))} / {fmt_tps(h.get('decode_tps_median'))}",
        "decode": h.get("decode_tps_median"),
        "ttft": fmt_ms(h.get("ttft_ms_median")),
        "bottleneck": diag or "—",
        "total": fmt_ms(total_ms) if total_ms else "—",
        "budget": st.get("reasoning_budget"),
    }


def _cells(r: dict) -> str:
    mark = ""
    if r.get("comparable") is False:
        mark += " †"
    if r["budget"] == 0:
        mark += " ⚠"
    return (f"| {r['folder']}/{r['ts']}{mark} | {r['settings']} | {r['quiz']} "
            f"| {r['judge']} | {r['dev']} | {r['perf']} | {r['ttft']} | {r['bottleneck']} "
            f"| {r['total']} |")


def _rank(r: dict) -> tuple:
    return (r["composite"] if r["composite"] is not None else -1,
            r["decode"] if r["decode"] is not None else -1)


def build_compare(root: str | Path, current_model_dir: Path,
                  n_quiz: int | None, n_dev: int | None) -> str:
    """Render compare.md. ``n_quiz``/``n_dev`` = suite sizes of the current run;
    rows with different sizes are marked † and excluded from the ranking."""
    rows: list[dict] = []
    for run in sorted(Path(root).glob("*/*/"), key=lambda r: r.name):
        r = _load_run(run)
        if not r or r["model"] == "mock-model":  # mock runs are plumbing tests, not model data
            continue
        r["comparable"] = (n_quiz is None or r["n_quiz"] == n_quiz) and \
                          (n_dev is None or r["n_dev"] == n_dev)
        rows.append(r)

    this_run = sorted((r for r in rows if run_matches(r, current_model_dir)),
                      key=lambda r: r["ts"], reverse=True)
    best: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r["ts"], reverse=True):
        rank = (r["composite"] if r["composite"] is not None else -1,
                r["decode"] if r["decode"] is not None else -1)
        cur = best.get(r["folder"])
        # Best comparable run wins; a comparable run always beats a † row.
        if cur is None or (r["comparable"] and (not cur["comparable"] or rank > _rank(cur))):
            best[r["folder"]] = r
    ranked = sorted(best.values(), key=_rank, reverse=True)

    lines = [
        "# Run comparison — all recorded runs",
        "",
        "_decode tok/s excludes thinking tokens from chunk timing but counts them in "
        "usage totals (see AGENTS.md), so treat it as an upper bound for thinking runs; "
        "⚠ = budget 0 (no thinking), † = different suite size, not comparable._",
        "",
        "## This model+quant — every recorded run (newest first)",
        "",
        _HDR, _SEP,
        *(_cells(r) for r in this_run),
        "",
        "## Best run per model+quant (comparable suite, ranked by dev score then decode)",
        "",
        _HDR, _SEP,
        *(_cells(r) for r in ranked),
    ]
    return "\n".join(lines) + "\n"


def run_matches(r: dict, current_model_dir: Path) -> bool:
    return r["folder"] == current_model_dir.name

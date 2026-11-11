"""Shared plain-language formatting helpers for result rendering.

Dependency-free on purpose: runner/perf/dev/report all import from here,
so it must not import any of them (no circulars).
"""

from __future__ import annotations

from typing import Any

# (term, plain-words explanation) — rendered as a footer in every result file.
GLOSSARY: list[tuple[str, str]] = [
    ("TTFT", "time to first word — how long until the model starts writing (the 'reading' phase)."),
    ("prefill tok/s", "how fast it reads your question (words per second)."),
    ("decode tok/s", "how fast it writes the answer, one word at a time."),
    ("ITL", "interval between two written words — how steady the writing is."),
    ("stall", "a pause of half a second or more mid-writing (a hiccup)."),
    ("tok/s", "words per second. One 'word' (token) is about 3/4 of an English word."),
]


def fmt_ms(v: Any) -> str:
    """Human-friendly duration: ms under 1 s, seconds under 1 min, then minutes."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v < 1000:
        return f"{v:.0f} ms"
    if v < 60000:
        return f"{v / 1000:.1f} s"
    m, s = divmod(int(v // 1000), 60)
    return f"{m}m {s:02d}s"


def fmt_tps(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.0f} tok/s"
    except (TypeError, ValueError):
        return str(v)


def pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.0%}"
    except (TypeError, ValueError):
        return str(x)


def glossary_md() -> list[str]:
    """Footer block explaining the jargon in plain words."""
    lines = ["", "---", "", "_Plain-words legend:_"]
    lines += [f"- **{term}** — {expl}" for term, expl in GLOSSARY]
    return lines

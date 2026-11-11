"""Per-model result folders and per-run output files.

Layout (one folder per model+quant; repeat runs append, never overwrite).
When ``--quant`` is given the quant is part of the folder key — one quant,
one folder:

    results/
      <model>-<QUANT>/              # one folder per model+quant pair
        2026-11-09T15-30-12Z/       # one subfolder per run (UTC timestamp)
          quality.json / perf.json / dev.json
          quality.md   / perf.md   / dev.md

``bench all`` runs the three standards back-to-back into one run folder.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_.") or "model"


def model_dir(root: str | Path, model: str, quant: str | None = None) -> Path:
    """Per-model folder. With ``quant`` given, the quant is part of the folder
    key: ``<root>/<model>-<QUANT>/`` — one quant per folder."""
    key = f"{_safe_name(model)}-{_safe_name(quant)}" if quant else _safe_name(model)
    return Path(root) / key


def model_display_name(model: str, quant: str | None = None) -> str:
    """The name used in reports for a model+quant pair."""
    return f"{model}-{quant}" if quant else model


def run_folder(model_dir: Path, ts: str) -> Path:
    return model_dir / ts


# --------------------------------------------------------------------------- #
# Orchestration: write one standard's outputs into the run folder
# --------------------------------------------------------------------------- #
_RUN_TS_FMT = "%Y-%m-%dT%H-%M-%SZ"


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime(_RUN_TS_FMT)


def write_standard(
    root: str | Path,
    model: str,
    folder_ts: str,
    std: str,
    doc: dict,
    md: str,
    status: str = "ok",
    quant: str | None = None,
) -> tuple[Path, Path]:
    """Write one standard's json+md into the run folder."""
    mdir = model_dir(root, model, quant=quant)
    rdir = run_folder(mdir, folder_ts)
    rdir.mkdir(parents=True, exist_ok=True)
    jp = rdir / f"{std}.json"
    mp = rdir / f"{std}.md"
    jp.write_text(json.dumps(doc, indent=2))
    mp.write_text(md)
    return jp, mp

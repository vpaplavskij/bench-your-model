"""C3: harness revision decides comparability (unstamped runs are rev 1)."""
import json

from bench.compare import build_compare
from bench.version import HARNESS_REV


def make_run(tmp_path, folder, ts, meta_extra):
    run = tmp_path / folder / ts
    run.mkdir(parents=True)
    meta = {"model": "X", "settings": {"quant": "Q4"}} | meta_extra
    (run / "quality.json").write_text(json.dumps(
        {"meta": meta, "summary": {"structured": {"correct": 11, "n": 11}}}))


def test_unstamped_run_is_marked_not_comparable(tmp_path):
    make_run(tmp_path, "X-Q4", "2026-01-01T00:00:00", {})
    make_run(tmp_path, "X-Q4", "2026-01-02T00:00:00", {"harness_rev": HARNESS_REV})
    md = build_compare(tmp_path, tmp_path / "X-Q4", 11, None)  # dev size irrelevant here
    rows = {line.split("|")[1].strip().split(" †")[0]: " †" in line
            for line in md.splitlines() if line.startswith("| X-Q4")}
    assert rows["X-Q4/2026-01-01T00:00:00"] is True   # rev 1 ≠ 2 → †
    assert rows["X-Q4/2026-01-02T00:00:00"] is False  # current rev → no †

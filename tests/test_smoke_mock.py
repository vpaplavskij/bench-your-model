"""Full `bench all` against the mock: the end-to-end plumbing check the
README's smoke test describes, asserted instead of eyeballed."""
import json

from bench.cli import main
from bench.version import HARNESS_REV
from conftest import MockServer

SETTINGS_KEYS = {"quant", "reasoning_level", "reasoning_budget",
                 "preserve_thinking", "context_length"}


def test_bench_all_against_mock(tmp_path):
    srv = MockServer()
    # Trimmed perf sweep: the point of this test is plumbing, not speed numbers.
    rc = main(["all", "--model", "mock-model", "--endpoint", srv.endpoint,
               "--quant", "MOCK", "--root", str(tmp_path),
               "--prompt-sizes", "128", "--output-tokens", "64",
               "--measure-runs", "1", "--settle-s", "0"])
    srv.stop()
    assert rc == 0

    folder = next((tmp_path / "mock-model-MOCK").iterdir())
    files = sorted(p.name for p in folder.iterdir())
    assert files == ["compare.md", "dev.json", "dev.md", "perf.json",
                     "perf.md", "quality.json", "quality.md"]

    q = json.loads((folder / "quality.json").read_text())
    d = json.loads((folder / "dev.json").read_text())
    assert q["summary"]["structured"]["correct"] == 11
    assert d["summary"]["passed"] == 13 and d["summary"]["n_tasks"] == 13
    assert d["summary"]["composite_score"] == 100
    assert q["summary"]["open_unjudged"] == 0
    assert q["summary"]["n_ok_requests"] == q["summary"]["n_tasks"]

    for doc in (q, d, json.loads((folder / "perf.json").read_text())):
        assert set(doc["meta"]["settings"]) == SETTINGS_KEYS
        assert doc["meta"]["harness_rev"] == HARNESS_REV

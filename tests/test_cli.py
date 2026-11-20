"""CLI dispatch regressions: no-subcommand defaults to quality; an unknown
--task ID is an empty run, not a crash (exit 0)."""
import json

from bench.cli import main
from conftest import MockServer


def run(argv, srv, tmp_path):
    return main(argv + ["--model", "mock-model", "--endpoint", srv.endpoint,
                        "--root", str(tmp_path)])


def test_no_subcommand_runs_quality(mock_server, tmp_path):
    assert run(["--suite", "structured"], mock_server, tmp_path) == 0
    folder = next((tmp_path / "mock-model").iterdir())
    doc = json.loads((folder / "quality.json").read_text())
    assert doc["summary"]["structured"]["correct"] == doc["summary"]["structured"]["n"]


def test_explicit_quality_subcommand(mock_server, tmp_path):
    assert run(["quality", "--suite", "structured"], mock_server, tmp_path) == 0


def test_unknown_task_exits_zero(mock_server, tmp_path):
    assert run(["dev", "--task", "zzz"], mock_server, tmp_path) == 0
    assert run(["--task", "zzz"], mock_server, tmp_path) == 0

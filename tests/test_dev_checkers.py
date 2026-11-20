"""Dev-task checker tests: reference solutions pass, wrong answers FAIL.

This is the harness-bug detector that makes fix A1 safe: if a checker ever
mis-scores (crashes on a valid solution, or a checker-load bug hides a wrong
answer), these tests catch it.
"""
import time

import pytest

from bench import dev
from bench.devtasks import DEV_TASKS
from conftest import (answer_client, checker_required_symbols, code_symbol_in_prompt,
                      dev_task, reference_answer)


@pytest.mark.parametrize("task_id", [t["id"] for t in DEV_TASKS])
def test_reference_solution_passes(task_id):
    task = dev_task(task_id)
    ref = reference_answer(task)
    assert ref, f"mock has no reference solution for {task_id}"
    rec = dev._run_dev_task(answer_client(ref), task)
    assert rec.checker_error is None, rec.checker_error
    assert rec.status == "PASS", rec.checker_detail


@pytest.mark.parametrize("task_id", [t["id"] for t in DEV_TASKS])
def test_wrong_answer_is_scored_fail_not_checker_error(task_id):
    task = dev_task(task_id)
    wrong = "no code here, just prose" if task["kind"] == "thinking" else "x = 1"
    rec = dev._run_dev_task(answer_client(wrong), task)
    assert rec.checker_error is None, rec.checker_error  # harness fault, not a FAIL
    assert not rec.passed
    assert rec.status in ("FAIL", "RUNTIME-ERROR")


@pytest.mark.parametrize("task_id", ["d4", "d5", "d8"])
def test_none_return_counts_as_fail(task_id):
    """A missing `return` (function returns None) broke the checks: A1 made
    that a plain FAIL instead of a CHECKER-ERROR exclusion."""
    task = dev_task(task_id)
    names = checker_required_symbols(task)
    assert names, f"no symbols in {task_id} checker"
    snippet = "\n\n".join(f"def {n}(*a, **k):\n    pass" for n in names)
    rec = dev._run_dev_task(answer_client(snippet), task)
    assert rec.checker_error is None, rec.checker_error
    assert rec.status == "FAIL"
    # The None return was actually evaluated (crashed the checker or was
    # compared against), not excused — that's what A1 fixed.
    assert "broke the checks" in rec.checker_detail or "None" in rec.checker_detail


@pytest.mark.slow
def test_infinite_loop_is_a_runtime_error_in_about_15s():
    task = dev_task("d10")
    name = code_symbol_in_prompt(task["prompt"])
    t0 = time.perf_counter()
    rec = dev._run_dev_task(
        answer_client(f"def {name}(*a, **k):\n    while True:\n        pass"), task)
    elapsed = time.perf_counter() - t0
    assert rec.checker_error is None
    assert rec.status == "RUNTIME-ERROR"
    assert "timed out" in rec.runtime_error
    assert 14.0 < elapsed < 30.0, elapsed

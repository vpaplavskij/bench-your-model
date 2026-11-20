"""Shared test helpers: a mock server subprocess and a scripted fake client."""
import re
import socket
import subprocess
import sys
import time

import httpx
import pytest

from bench.client import ChatResult
from bench.devtasks import DEV_TASKS
from bench.mockserver import DEV_CODE


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockServer:
    """Starts ``python -m bench.cli mock`` as a child process.

    Kills only the PID it started — never anything else on the machine.
    """

    def __init__(self, **flags):
        self.port = free_port()
        args = [sys.executable, "-m", "bench.cli", "mock", "--port", str(self.port)]
        for k, v in flags.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                httpx.get(self.endpoint + "/models", timeout=1.0).raise_for_status()
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError("mock server exited before it was ready")
                time.sleep(0.1)
        self.stop()
        raise RuntimeError("mock server never came up")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture()
def mock_server():
    srv = MockServer()
    yield srv
    srv.stop()


class FakeClient:
    """Returns canned ChatResults in order (one chat() call each)."""

    def __init__(self, results):
        self.results = list(results)
        self.i = 0

    def chat(self, messages, **kw):
        r = self.results[self.i]
        self.i += 1
        return r


def answer_client(content: str) -> FakeClient:
    return FakeClient([ChatResult(ok=True, content=content, raw_finish_reason="stop",
                                  stream_chunks=1, ttft_ms=1.0, total_ms=1.0,
                                  prompt_tokens=10, completion_tokens=10)])


def dev_task(task_id: str) -> dict:
    return next(t for t in DEV_TASKS if t["id"] == task_id)


def reference_answer(task: dict) -> str:
    """The mock's canned answer for a task prompt — its reference solution."""
    return _detect_answer(task["prompt"])


def _detect_answer(prompt: str) -> str:
    from bench.mockserver import _detect
    return _detect([{"role": "user", "content": prompt}], {}) or ""


def checker_required_symbols(task: dict) -> list[str]:
    """Names the checker looks up in the executed namespace (all checkers use
    ``r["ns"].get("<name>")``), i.e. every symbol a snippet must define."""
    return re.findall(r'\["ns"\]\.get\("(\w+)"\)', task["checker_src"])


def code_symbol_in_prompt(prompt: str) -> str | None:
    """The DEV_CODE symbol the mock (and checkers) key on for this prompt."""
    for key in DEV_CODE:
        if re.search(rf"\b{re.escape(key)}\b", prompt):
            return key
    return None

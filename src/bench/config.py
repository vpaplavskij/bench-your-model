"""Run configuration for the benchmark harness."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """Everything a run needs to know.

    ``endpoint``/``model`` are required; everything else has sane defaults
    that can be overridden via CLI flags (see :func:`build_parser`).

    The ``quant`` / ``reasoning_*`` / ``preserve_thinking`` fields record how
    the SERVER serves the model (Unsloth Studio / llama.cpp startup settings).
    They are reported verbatim in every run document — see ``REPORT_SPEC.md``.
    They describe the server under test, NOT the
    agent/harness that invokes the benchmark.
    """

    endpoint: str = "http://localhost:8080/v1"
    model: str = ""
    api_key: str = "none"
    timeout_s: float = 300.0
    max_tokens: int = 4096
    temperature: float = 0.0
    judge_model: str | None = None
    suite: str = "all"  # "structured" | "open" | "all"
    root: str = "results"  # results live under <root>/<model>/
    task_ids: list[str] = field(default_factory=list)  # empty = all tasks

    # How the server serves the model (from Unsloth Studio / llama.cpp, NOT
    # from the agent harness). None = "not recorded". See REPORT_SPEC.md.
    quant: str | None = None
    reasoning_level: str | None = None
    reasoning_budget: int | None = None
    preserve_thinking: bool | None = None
    context_length: int | None = None

    def resolve_judge(self) -> str:
        """Judge model defaults to the model under test."""
        return self.judge_model or self.model

    def settings(self) -> dict:
        """The server-side settings block stored in every run document's meta."""
        return {
            "quant": self.quant,
            "reasoning_level": self.reasoning_level,
            "reasoning_budget": self.reasoning_budget,
            "preserve_thinking": self.preserve_thinking,
            "context_length": self.context_length,
        }


def add_settings_args(p: argparse.ArgumentParser) -> None:
    """Flags recording how the SERVER serves the model (REPORT_SPEC.md).

    These describe the Unsloth Studio / llama.cpp instance under test — its
    GGUF quant and reasoning configuration. They must be read from the
    server's startup command / Studio config, never from the agent harness
    (e.g. OpenCode) that runs the benchmark.
    """
    p.add_argument("--quant", default=None,
                   help="GGUF quantization of the model under test, e.g. IQ3_S, Q4_K_M "
                        "(required by REPORT_SPEC.md; results folder becomes <model>-<QUANT>)")
    p.add_argument("--reasoning-level", dest="reasoning_level", default=None,
                   help="reasoning effort/level the server runs at (e.g. default, low, medium, high)")
    p.add_argument("--reasoning-budget", dest="reasoning_budget", type=int, default=None,
                   help="token budget for reasoning: -1 = unlimited thinking, 0 = no thinking")
    p.add_argument("--preserve-thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="whether the server preserves thinking content "
                        "(--preserve-thinking / --no-preserve-thinking; omit if unknown)")
    p.add_argument("--context-length", dest="context_length", type=int, default=None,
                   help="context length (tokens) the server runs at, e.g. from -c/--ctx-size "
                        "(omit if unknown)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench",
        description="Benchmark a local LLM server (OpenAI-compatible API).",
    )
    p.add_argument("--endpoint", default=os.environ.get("BENCH_ENDPOINT", "http://localhost:8080/v1"),
                   help="Base URL of an OpenAI-compatible server, e.g. http://localhost:8080/v1")
    p.add_argument("--model", required=True,
                   help="Model name to send in requests (see /v1/models)")
    p.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", "none"))
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-request timeout in seconds (default: 300)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--judge-model", default=None,
                   help="Model used for LLM-judge scoring (default: model under test)")
    p.add_argument("--suite", choices=["structured", "agent", "open", "all"], default="all")
    p.add_argument("--task", action="append", default=[], metavar="ID",
                   help="Run only specific task IDs (repeatable), e.g. --task s1 --task o2")
    p.add_argument("--root", default="results",
                   help="Results root; results live under <root>/<model>/ (default: ./results)")
    add_settings_args(p)
    return p


def parse_args(argv: list[str] | None = None) -> RunConfig:
    args = build_parser().parse_args(argv)
    return RunConfig(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        judge_model=args.judge_model,
        suite=args.suite,
        root=args.root,
        task_ids=list(args.task),
        quant=args.quant,
        reasoning_level=args.reasoning_level,
        reasoning_budget=args.reasoning_budget,
        preserve_thinking=args.preserve_thinking,
        context_length=args.context_length,
    )

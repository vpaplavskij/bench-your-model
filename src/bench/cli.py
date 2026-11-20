"""CLI entry point.

Six subcommands share one executable:

* ``bench quality ...``  — accuracy harness (structured + open/LLM-judge)
* ``bench perf ...``     — speed benchmark (prefill/decode, TTFT, bottlenecks)
* ``bench dev ...``      — developer/coding tasks (accuracy, errors, total time)
* ``bench all ...``      — run all three back-to-back into ONE run folder
* ``bench mock ...``     — start a stdlib-only mock OpenAI server (smoke tests)

Running ``bench ...`` with no subcommand defaults to ``quality``.

Every run lands in a per-model folder:  ``<root>/<model>-<QUANT>/<run-ts>/``
(quant from ``--quant``, see REPORT_SPEC.md).
"""

from __future__ import annotations

import os
import sys

from .config import RunConfig, add_settings_args, build_parser, parse_args
from .runner import run_benchmark, render_markdown


def _warn_no_quant(quant: str | None) -> None:
    if not quant:
        print("WARNING: --quant not given — the report will record 'not recorded' for "
              "the quantization. REPORT_SPEC.md requires the GGUF quant of the model "
              "under test (read it from the Unsloth Studio / llama.cpp startup).",
              file=sys.stderr)


def _perf_parser() -> "argparse.ArgumentParser":  # noqa: F821
    import argparse

    p = argparse.ArgumentParser(
        prog="bench perf",
        description="Speed benchmark: prefill/decode throughput, TTFT, stability, bottlenecks.",
    )
    p.add_argument("--endpoint", default=os.environ.get("BENCH_ENDPOINT", "http://localhost:8080/v1"))
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", "none"))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--prompt-sizes", default="128,512,1024,2048",
                   help="comma-separated prompt sizes in tokens (default: 128,512,1024,2048)")
    p.add_argument("--output-tokens", type=int, default=256,
                   help="fixed output length per run (default: 256)")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measure-runs", type=int, default=3)
    p.add_argument("--settle-s", type=float, default=2.0,
                   help="pause between prompt sizes to let the server settle")
    p.add_argument("--root", default="results",
                   help="results root; results live under <root>/<model>/ (default: ./results)")
    add_settings_args(p)
    return p


def _run_perf(argv: list[str]) -> int:
    from . import perf as perf_mod
    from . import report

    args = _perf_parser().parse_args(argv)
    _warn_no_quant(args.quant)
    cfg = RunConfig(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.timeout,
        quant=args.quant,
        reasoning_level=args.reasoning_level,
        reasoning_budget=args.reasoning_budget,
        preserve_thinking=args.preserve_thinking,
        context_length=args.context_length,
    )
    sizes = [int(x) for x in args.prompt_sizes.split(",") if x.strip()]
    doc = perf_mod.run_perf_sweep(
        cfg,
        prompt_sizes=sizes,
        output_tokens=args.output_tokens,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        settle_s=args.settle_s,
    )

    # Per-model folder: <root>/<model>-<QUANT>/<run-ts>/perf.{json,md}
    ts = report.run_ts(args.root, args.model, args.quant)
    jp, mp = report.write_standard(
        args.root, args.model, ts, "perf", doc, perf_mod.render_perf_markdown(doc),
        quant=args.quant)

    n_err = sum(len(pt["errors"]) for pt in doc["points"])
    if n_err:
        print(f"WARNING: {n_err} perf run(s) errored — read perf.json, exit code is 0 "
              "by design (task failures are data, not aborts).", file=sys.stderr)

    print("\nDiagnosis:")
    for d in doc["diagnosis"]:
        print(f"  {d}")
    print(f"\nResults written to:\n  {jp}\n  {mp}")
    return 0


def _dev_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="bench dev",
        description="Developer/coding benchmark: accuracy, runtime errors, total time, ranking.",
    )
    p.add_argument("--endpoint", default=os.environ.get("BENCH_ENDPOINT", "http://localhost:8080/v1"))
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", "none"))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--task", action="append", default=[], metavar="ID",
                   help="run only specific dev task IDs (repeatable), e.g. --task d1")
    p.add_argument("--root", default="results",
                   help="results root; results live under <root>/<model>/ (default: ./results)")
    add_settings_args(p)
    return p


def _run_dev(argv: list[str]) -> int:
    from . import dev as dev_mod
    from . import report

    args = _dev_parser().parse_args(argv)
    _warn_no_quant(args.quant)
    cfg = RunConfig(
        endpoint=args.endpoint, model=args.model, api_key=args.api_key,
        timeout_s=args.timeout, max_tokens=args.max_tokens,
        quant=args.quant, reasoning_level=args.reasoning_level,
        reasoning_budget=args.reasoning_budget, preserve_thinking=args.preserve_thinking,
        context_length=args.context_length,
    )
    doc = dev_mod.run_dev_benchmark(cfg, task_ids=list(args.task))

    ts = report.run_ts(args.root, args.model, args.quant)
    jp, mp = report.write_standard(
        args.root, args.model, ts, "dev", doc, dev_mod.render_dev_markdown(doc),
        quant=args.quant)

    s = doc["summary"]
    print(f"\nAccuracy: {s['passed']}/{s['n_tasks']} ({s['accuracy']:.0%})  |  "
          f"runtime errors: {s['runtime_errors']}  |  composite score: {s['composite_score']}/100")
    if doc["error_log"]:
        print(f"WARNING: {len(doc['error_log'])} dev task(s) failed/errored — read dev.json, "
              "exit code is 0 by design (task failures are data, not aborts).", file=sys.stderr)
        for e in doc["error_log"]:
            print(f"  - {e['task_id']}: {e['error']}")
    print(f"\nResults written to:\n  {jp}\n  {mp}")
    return 0


def _run_quality(argv: list[str]) -> int:
    from . import report

    cfg = parse_args(argv)
    _warn_no_quant(cfg.quant)
    doc = run_benchmark(cfg)

    ts = report.run_ts(cfg.root, cfg.model, cfg.quant)
    jp, mp = report.write_standard(
        cfg.root, cfg.model, ts, "quality", doc, render_markdown(doc), quant=cfg.quant)

    s = doc["summary"]
    print(f"\nTasks: {s['n_tasks']} (ok {s['n_ok_requests']})  |  "
          f"structured accuracy: {s['structured']['correct']}/{s['structured']['n']}")
    if s["n_ok_requests"] < s["n_tasks"] or s["structured"]["correct"] < s["structured"]["n"] or s["checker_errors"]:
        print("WARNING: some tasks failed/errored — read the .json; exit code is 0 "
              "by design (task failures are data, not aborts).", file=sys.stderr)
    print(f"\nResults written to:\n  {jp}\n  {mp}")
    return 0


def _all_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="bench all",
        description="Run quality + perf + dev back-to-back into ONE run folder.",
    )
    p.add_argument("--endpoint", default=os.environ.get("BENCH_ENDPOINT", "http://localhost:8080/v1"))
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", "none"))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature for quality tasks (perf/dev run at 0.0)")
    p.add_argument("--judge-model", default=None,
                   help="Model used for LLM-judge scoring (default: model under test)")
    p.add_argument("--suite", choices=["structured", "agent", "open", "all"], default="all")
    p.add_argument("--prompt-sizes", default="128,512,1024,2048")
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--measure-runs", type=int, default=3)
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--root", default="results")
    add_settings_args(p)
    return p


def _run_all(argv: list[str]) -> int:
    from . import dev as dev_mod
    from . import perf as perf_mod
    from . import report

    args = _all_parser().parse_args(argv)
    _warn_no_quant(args.quant)
    cfg = RunConfig(
        endpoint=args.endpoint, model=args.model, api_key=args.api_key,
        timeout_s=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature,
        judge_model=args.judge_model, suite=args.suite, root=args.root,
        quant=args.quant, reasoning_level=args.reasoning_level,
        reasoning_budget=args.reasoning_budget, preserve_thinking=args.preserve_thinking,
        context_length=args.context_length,
    )
    ts = report.run_ts(args.root, args.model, args.quant)

    print(f"=== quality ===")
    qdoc = run_benchmark(cfg)
    jp, mp = report.write_standard(args.root, args.model, ts, "quality",
                                   qdoc, render_markdown(qdoc), quant=args.quant)
    print(f"  -> {mp}")

    print(f"\n=== perf ===")
    sizes = [int(x) for x in args.prompt_sizes.split(",") if x.strip()]
    pdoc = perf_mod.run_perf_sweep(
        cfg, prompt_sizes=sizes, output_tokens=args.output_tokens,
        warmup_runs=1, measure_runs=args.measure_runs, settle_s=args.settle_s)
    jp, mp = report.write_standard(args.root, args.model, ts, "perf",
                                   pdoc, perf_mod.render_perf_markdown(pdoc), quant=args.quant)
    print(f"  -> {mp}")

    print(f"\n=== dev ===")
    ddoc = dev_mod.run_dev_benchmark(cfg)
    jp, mp = report.write_standard(args.root, args.model, ts, "dev",
                                   ddoc, dev_mod.render_dev_markdown(ddoc), quant=args.quant)
    print(f"  -> {mp}")

    if (qdoc["summary"]["n_ok_requests"] < qdoc["summary"]["n_tasks"]
            or qdoc["summary"]["checker_errors"] or ddoc["error_log"]):
        print("WARNING: some tasks failed/errored — read the .json files; exit code is 0 "
              "by design (task failures are data, not aborts).", file=sys.stderr)

    mdir = report.model_dir(args.root, args.model, quant=args.quant)
    from .compare import build_compare
    cp = report.run_folder(mdir, ts) / "compare.md"
    cp.write_text(build_compare(args.root, mdir,
                                qdoc["summary"]["structured"]["n"], ddoc["summary"]["n_tasks"]))
    print(f"  -> {cp}")

    print(f"\nAll three standards complete for `{args.model}`.")
    print(f"  run folder : {mdir / ts}")
    return 0


def _run_mock(argv: list[str]) -> int:
    from .mockserver import main as mock_main

    return mock_main(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Dispatch on the first token BEFORE parsing, so `bench <mode> --help` works.
    mode = argv[0] if argv and argv[0] in ("quality", "perf", "dev", "all", "mock") else "quality"
    rest = argv[1:] if mode != "quality" else argv

    if mode == "perf":
        return _run_perf(rest)
    if mode == "dev":
        return _run_dev(rest)
    if mode == "all":
        return _run_all(rest)
    if mode == "mock":
        return _run_mock(rest)

    return _run_quality(rest)


if __name__ == "__main__":
    sys.exit(main())

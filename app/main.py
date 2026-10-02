"""CLI entry point: python -m app.main <command>."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from app.config import Config
from app.controller import Controller, PreflightError, setup_logging


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.main", description="Local autonomous self-improver for deepseek-r1:1.5b via Ollama")
    p.add_argument("--model", help="override model tag (default deepseek-r1:1.5b)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check Python deps, Ollama, model, database")
    sub.add_parser("ping", help="exit 0 if Ollama is reachable (used by the .bat launcher)")
    si = sub.add_parser("self-improve", help="autonomous Self Improve: learning starts right after preflight, no benchmarks by default")
    si.add_argument("--minutes", type=float, default=30, help="actual learning time")
    si.add_argument("--with-benchmarks", action="store_true", help="optionally run before/after benchmarks around learning")
    si.add_argument("--quick-benchmark", action="store_true", help="use the fixed 12-task stratified subset (with --with-benchmarks)")
    r = sub.add_parser("run", help="run an autonomous self-improvement session")
    r.add_argument("--minutes", type=float, default=30)
    r.add_argument("--skip-benchmark", action="store_true", help="skip before/after benchmark (no improvement claim)")
    r.add_argument("--quick-benchmark", action="store_true", help="use the fixed 12-task stratified subset")
    b = sub.add_parser("benchmark", help="run the fixed benchmark once")
    b.add_argument("--quick", action="store_true")
    sub.add_parser("stats", help="show database statistics")
    l = sub.add_parser("lessons", help="list verified lessons")
    l.add_argument("--domain")
    l.add_argument("--limit", type=int, default=50)
    sub.add_parser("prepare-training", help="export the verified-only training dataset")
    t = sub.add_parser("train-adapter", help="(optional) train a LoRA adapter; never overwrites the base model")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--hf-base")
    t.add_argument("--force-cpu", action="store_true")
    sub.add_parser("gui", help="open the Tkinter GUI")
    return p


def main(argv: Sequence[str] | None = None, controller: Controller | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.model:
        cfg.model = args.model
    setup_logging(cfg)
    c = controller or Controller(cfg)
    try:
        return _dispatch(args, c)
    except PreflightError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        c.stop()
        print("Interrupted.")
        return 130


def _dispatch(args: argparse.Namespace, c: Controller) -> int:
    if args.cmd == "doctor":
        checks = c.doctor()
        for ch in checks:
            print(f"[{'OK' if ch['ok'] else ('FAIL' if ch['required'] else 'WARN')}] {ch['name']}: {ch['detail']}")
        return 0 if all(ch["ok"] or not ch["required"] for ch in checks) else 1
    if args.cmd == "ping":
        try:
            print(c.client.check_server())
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"Ollama not reachable: {e}", file=sys.stderr)
            return 1
    if args.cmd in ("run", "self-improve"):
        c.add_listener(_progress_printer())
        if args.cmd == "self-improve":
            report = c.self_improve(args.minutes, args.with_benchmarks, args.quick_benchmark)
        else:
            report = c.run_session(args.minutes, not args.skip_benchmark, args.quick_benchmark)
        print()
        print(f"Session {report['session_id']} {report['status']}: attempted={report['attempted']} "
              f"verified={report['verified']} failed={report['failed']} retries={report['retries']} "
              f"lessons={report['lessons_learned']}")
        if report["benchmark_delta"] is not None:
            print(f"Benchmark before={report['benchmark_before']['overall']:.3f} after={report['benchmark_after']['overall']:.3f} "
                  f"delta={report['benchmark_delta']:+.3f} improved={report['improved']}")
        else:
            print(f"Benchmark comparison: {report['improvement_note']}")
        print("Reports:", *c.snapshot()["report_paths"])
        return 0 if report["status"] in ("completed", "stopped") else 1
    if args.cmd == "benchmark":
        res = c.run_benchmark(args.quick, lambda i, n, row: print(f"  [{i}/{n}] {row['task_id']} {'PASS' if row['correct'] else 'fail'}"))
        print(f"Overall: {res.overall * 100:.1f}%  ({res.n_tasks} tasks, {res.version})")
        for k, v in res.scores.items():
            print(f"  {k}: {v * 100:.1f}%")
        return 0
    if args.cmd == "stats":
        _print_json(c.stats())
        return 0
    if args.cmd == "lessons":
        rows = c.lessons(args.domain, args.limit)
        if not rows:
            print("No verified lessons yet.")
        for r in rows:
            print(f"#{r['lesson_id']} [{r['domain']}] {r['created_at']}\n  pattern: {r['failure_pattern']}\n  method:  {r['correct_method']}\n  source:  {r['source']}")
        return 0
    if args.cmd == "prepare-training":
        _print_json(c.prepare_training())
        return 0
    if args.cmd == "train-adapter":
        from app.training.lora import TrainingUnavailable, train_adapter
        m = c.prepare_training()
        try:
            out = train_adapter(c.config.training_dir / "train.jsonl", c.config.adapters_dir, c.client.model,
                                args.hf_base, args.epochs, args.force_cpu)
        except TrainingUnavailable as e:
            print(f"Training not started: {e}")
            print(f"(Dataset ready: {m['examples']} verified examples at {m['path']})")
            return 3
        print(f"Adapter saved to {out} (base model untouched)")
        return 0
    if args.cmd == "gui":
        from app.gui.app import launch_gui
        return launch_gui(c)
    return 1


def _progress_printer():
    last = {"msg": None}

    def cb(s: dict) -> None:
        line = (f"[{s['status']}] session={int(s['session_elapsed'])}s learning={int(s['elapsed'])}s attempted={s['attempted']} verified={s['verified']} "
                f"failed={s['failed']} retries={s['retries']} lessons={s['lessons']} | {s['message']}")
        if line != last["msg"]:
            last["msg"] = line
            print(line, flush=True)
    return cb


if __name__ == "__main__":
    sys.exit(main())

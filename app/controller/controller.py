"""Shared controller used by both the CLI and the GUI."""

from __future__ import annotations

import copy
import logging
import logging.handlers
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.benchmark.runner import BenchmarkAborted, BenchmarkResult, BenchmarkRunner
from app.benchmark.tasks import (
    BENCHMARK_TASKS,
    benchmark_hashes,
    benchmark_version,
    select_subset,
)
from app.config import Config
from app.discovery.adapters import default_adapters
from app.discovery.chain import CachingHttp, DiscoveryChain, NoTaskAvailable
from app.discovery.http import HttpClient, UrllibHttp
from app.memory.lessons import LessonMemory, learn_from_outcome
from app.ollama.client import (
    ModelNotAvailable,
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
)
from app.reporting.report import build_report, write_reports
from app.solver.solver import FATAL, CallBudget, Solver
from app.storage.db import Storage, now_iso
from app.training import dataset as training_dataset
from app.training.hardware import detect_hardware
from app.verifier import Verifier

log = logging.getLogger("self_improver")


class PreflightError(Exception):
    """Ollama/model not ready. The message is user-facing and actionable."""


class SessionBusy(Exception):
    pass


def setup_logging(config: Config) -> None:
    config.ensure_dirs()
    root = logging.getLogger()
    if any(getattr(h, "_selfimp", False) for h in root.handlers):
        return
    h = logging.handlers.RotatingFileHandler(
        config.logs_dir / "self_improver.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    h._selfimp = True  # type: ignore[attr-defined]
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)


@dataclass
class SessionState:
    status: str = "idle"  # idle|preflight|benchmark_before|running|benchmark_after|reporting|completed|stopped|error
    message: str = "Idle"
    mode: str = "benchmarked"  # benchmarked|self_improve
    session_id: int | None = None
    planned_minutes: float = 0.0
    elapsed: float = (
        0.0  # learning phase only (excludes preflight, benchmarks, reporting)
    )
    session_elapsed: float = (
        0.0  # whole session; live via snapshot() while running, frozen at the end
    )
    attempted: int = 0
    verified: int = 0
    failed: int = 0
    invalid: int = 0
    pending: int = 0  # handed to the model but cut off by deadline/stop: counted as attempted, not verified/failed
    retries: int = 0
    lessons: int = 0
    current_task: str = ""
    current_domain: str = ""
    benchmark_before: dict | None = None
    benchmark_after: dict | None = None
    benchmark_progress: str = ""
    domains: dict[str, dict[str, int]] = field(default_factory=dict)
    failure_categories: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    report_paths: list[str] = field(default_factory=list)


class Controller:
    def __init__(
        self,
        config: Config | None = None,
        client: OllamaClient | None = None,
        http: HttpClient | None = None,
        storage: Storage | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
        adapters: list | None = None,
    ):
        self.config = config or Config()
        self.config.ensure_dirs()
        self.storage = storage or Storage(self.config.db_path)
        self.client = client or OllamaClient(
            self.config.ollama_host,
            self.config.model,
            request_timeout=self.config.request_timeout,
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
            temperature=self.config.temperature,
        )
        self.http = http or UrllibHttp(self.config.user_agent, self.config.http_timeout)
        self.verifier = Verifier(self.http, self.config.code_timeout)
        self.memory = LessonMemory(self.storage)
        self.solver = Solver(
            self.client,
            self.verifier,
            self.storage,
            self.memory,
            self.config.max_retries,
            self.config.lessons_per_prompt,
            self.config.request_timeout,
            CallBudget(
                self.config.call_share,
                self.config.min_call_seconds,
                self.config.min_tokens,
                self.config.assumed_tokens_per_second,
            ),
        )
        self.bench = BenchmarkRunner(
            self.solver, self.memory, self.storage, self.client.model
        )
        self.clock = clock
        self._rng = rng or random.Random()
        self._custom_adapters = adapters
        self._stop = threading.Event()
        self._busy = threading.Lock()
        self._state = SessionState()
        self._state_lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self._thread: threading.Thread | None = None
        self._wait = sleep or (lambda s: self._stop.wait(s))
        self.last_summary: dict | None = None
        self._loop_end_reason = "deadline"
        self._loop_started: float | None = (
            None  # clock() when the learning phase began; None outside it
        )
        self._session_started: float | None = (
            None  # clock() when run_session began; None when no session runs
        )

    # ------------------------------------------------------------------ state / events
    def add_listener(self, cb: Callable[[dict], None]) -> None:
        self._listeners.append(cb)

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            snap = copy.deepcopy(self._state.__dict__)
            if self._loop_started is not None:
                # Live learning-phase time (the loop only refreshes `elapsed` between tasks, so it would freeze
                # during a long model call). Never reported above the planned limit.
                snap["elapsed"] = max(
                    0.0,
                    min(
                        self.clock() - self._loop_started,
                        self._state.planned_minutes * 60,
                    ),
                )
            if self._session_started is not None:
                snap["session_elapsed"] = max(0.0, self.clock() - self._session_started)
            return snap

    def _set(self, **kw: Any) -> None:
        with self._state_lock:
            for k, v in kw.items():
                setattr(self._state, k, v)
        snap = self.snapshot()
        for cb in list(self._listeners):
            try:
                cb(snap)
            except Exception:  # noqa: BLE001 - UI listeners must never break the loop
                log.exception("listener failed")

    @property
    def is_running(self) -> bool:
        return self._busy.locked()

    # ------------------------------------------------------------------ preflight / doctor
    def preflight(self) -> str:
        """Check Ollama + model. Returns the Ollama version, raises PreflightError otherwise."""
        try:
            return self.client.ensure_ready()
        except OllamaUnavailable as e:
            raise PreflightError(
                f"Ollama is not reachable at {self.client.host}. Start it with `ollama serve` "
                f"(START_SELF_IMPROVER.bat does this). Details: {e}"
            ) from e
        except ModelNotAvailable as e:
            raise PreflightError(str(e)) from e
        except OllamaError as e:
            raise PreflightError(f"Ollama check failed: {e}") from e

    def doctor(self) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
            checks.append(
                {"name": name, "ok": ok, "detail": detail, "required": required}
            )

        add(
            "python",
            sys.version_info >= (3, 10),
            f"Python {sys.version.split()[0]} (need >= 3.10)",
        )
        try:
            import sympy

            add("sympy", True, f"SymPy {sympy.__version__}")
        except ImportError:
            add("sympy", False, "missing: pip install -r requirements.txt")
        try:
            import tkinter  # noqa: F401

            add("tkinter", True, "available (GUI)", required=False)
        except ImportError:
            add(
                "tkinter",
                False,
                "missing: GUI unavailable (CLI still works)",
                required=False,
            )
        try:
            self.storage.stats()
            add("database", True, str(self.config.db_path))
        except Exception as e:  # noqa: BLE001
            add("database", False, str(e))
        try:
            v = self.client.check_server()
            add("ollama_server", True, f"Ollama {v} at {self.client.host}")
            try:
                have = self.client.has_model()
                add(
                    "model",
                    have,
                    f"{self.client.model} is "
                    + (
                        "installed"
                        if have
                        else f"NOT installed - run: ollama pull {self.client.model}"
                    ),
                )
            except OllamaError as e:
                add("model", False, str(e))
        except OllamaError as e:
            add("ollama_server", False, f"not reachable at {self.client.host}: {e}")
            add("model", False, "cannot check model while Ollama is down")
        hw = detect_hardware()
        add(
            "training_hardware",
            True,
            f"{hw.kind}: {hw.detail}"
            + (
                ""
                if hw.accelerated
                else " -> automatic fine-tuning disabled; lessons + dataset only"
            ),
            required=False,
        )
        return checks

    # ------------------------------------------------------------------ recovery
    def recover(self) -> list[int]:
        """Mark sessions left 'running' by a crash/kill as interrupted, requeue their tasks, write reports."""
        recovered = []
        for s in self.storage.running_sessions():
            sid = s["session_id"]
            self.storage.update_session(
                sid,
                status="interrupted",
                ended_at=now_iso(),
                stop_reason="process ended unexpectedly (recovered on next start)",
            )
            recovered.append(sid)
            counts = self.storage._all(
                "SELECT status, COUNT(*) c FROM tasks WHERE session_id=? GROUP BY status",
                (sid,),
            )
            c = {r["status"]: r["c"] for r in counts}
            state = {
                "attempted": c.get("verified", 0) + c.get("failed", 0),
                "verified": c.get("verified", 0),
                "failed": c.get("failed", 0),
                "retries": 0,
                "lessons": len(
                    [
                        l
                        for l in self.storage.list_lessons()
                        if self.storage.get_task_row(l["task_id"])
                        and self.storage.get_task_row(l["task_id"])["session_id"] == sid
                    ]
                ),
                "elapsed": 0.0,
            }
            rep = build_report(
                self.storage.get_session(sid),
                state,
                None,
                None,
                [],
                self.storage.list_sources(s["started_at"]),
                ["session was interrupted; counts reconstructed from the database"],
                skipped_reason="session interrupted; no benchmark comparison",
            )
            write_reports(rep, self.config.reports_dir)
        n = self.storage.requeue_unfinished_tasks()
        if recovered or n:
            log.info("recovered sessions=%s requeued_tasks=%s", recovered, n)
        return recovered

    # ------------------------------------------------------------------ benchmark only
    def run_benchmark(
        self,
        quick: bool = False,
        progress: Callable[[int, int, dict], None] | None = None,
    ) -> BenchmarkResult:
        self.preflight()
        tasks = select_subset(self.config.benchmark_quick_size if quick else None)
        self._stop.clear()
        return self.bench.run(tasks, "standalone", None, self._stop.is_set, progress)

    # ------------------------------------------------------------------ session
    def start(
        self,
        minutes: float,
        run_benchmarks: bool = True,
        quick_benchmark: bool = False,
        mode: str = "benchmarked",
    ) -> threading.Thread:
        """Run a session in a background thread (GUI). Errors are reported via state."""
        if self.is_running:
            raise SessionBusy("A session is already running")

        def target() -> None:
            try:
                self.run_session(minutes, run_benchmarks, quick_benchmark, mode)
            except (PreflightError, SessionBusy) as e:
                self._set(status="error", message=str(e), errors=[str(e)])
            except Exception as e:  # noqa: BLE001
                log.exception("session crashed")
                self._set(status="error", message=f"Unexpected error: {e}")

        self._thread = threading.Thread(target=target, name="session", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if self.is_running:
            self._set(message="Stopping...")

    def self_improve(
        self,
        minutes: float,
        run_benchmarks: bool = False,
        quick_benchmark: bool = False,
    ) -> dict[str, Any]:
        """Autonomous Self Improve mode: the learning loop starts right after preflight and the whole
        selected duration is learning time. Benchmarks run only when explicitly enabled."""
        return self.run_session(
            minutes, run_benchmarks, quick_benchmark, mode="self_improve"
        )

    def run_session(
        self,
        minutes: float,
        run_benchmarks: bool = True,
        quick_benchmark: bool = False,
        mode: str = "benchmarked",
    ) -> dict[str, Any]:
        if not self._busy.acquire(blocking=False):
            raise SessionBusy("A session is already running")
        self._session_started = self.clock()
        try:
            return self._run_session(minutes, run_benchmarks, quick_benchmark, mode)
        finally:
            with self._state_lock:  # freeze the whole-session time in the same step that stops the live clock
                self._state.session_elapsed = max(
                    0.0, self.clock() - self._session_started
                )
                self._session_started = None
            self._busy.release()

    def _run_session(
        self,
        minutes: float,
        run_benchmarks: bool,
        quick: bool,
        mode: str = "benchmarked",
    ) -> dict[str, Any]:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        self._stop.clear()
        with self._state_lock:
            self._state = SessionState(
                status="preflight",
                message="Checking Ollama and model...",
                planned_minutes=minutes,
                mode=mode,
            )
        self._set()
        self.recover()
        try:
            version = self.preflight()
        except PreflightError as e:
            self._set(status="error", message=str(e), errors=[str(e)])
            raise
        errors: list[str] = []
        sid = self.storage.create_session(minutes, self.client.model)
        self._set(
            session_id=sid, message=f"Ollama {version}, model {self.client.model} ready"
        )
        bench_tasks = select_subset(self.config.benchmark_quick_size if quick else None)
        before = after = None
        stop_reason = "deadline"
        skip_note: str | None = None
        t_loop = 0.0
        try:
            # --- benchmark BEFORE (outside the learning deadline)
            if run_benchmarks:
                before = self._bench_phase(
                    "benchmark_before", "before", sid, bench_tasks, allow_reuse=True
                )
                if before is None:
                    stop_reason = "stopped"
            # --- hard deadline starts here
            if stop_reason != "stopped":
                t_loop = self._learning_loop(sid, minutes, errors)
                stop_reason = self._loop_end_reason
            # --- benchmark AFTER
            if run_benchmarks and stop_reason == "deadline":
                after = self._bench_phase(
                    "benchmark_after", "after", sid, bench_tasks, allow_reuse=False
                )
                if after is None:
                    skip_note = (
                        "after-benchmark did not complete; no improvement claimed"
                    )
            elif run_benchmarks:
                skip_note = f"session ended early ({stop_reason}); after-benchmark skipped, no improvement claimed"
            elif not run_benchmarks:
                skip_note = "benchmarks disabled for this run"
        except OllamaError as e:
            errors.append(f"Ollama failure: {e}")
            stop_reason = "error"
            skip_note = "Ollama failed during the session; no improvement claimed"
        except Exception as e:  # noqa: BLE001
            log.exception("session failed")
            errors.append(f"{type(e).__name__}: {e}")
            stop_reason = "error"
            skip_note = "internal error; no improvement claimed"
        return self._finish(sid, stop_reason, before, after, errors, skip_note, t_loop)

    # -- phases ---------------------------------------------------------------
    def _bench_phase(
        self, status: str, label: str, sid: int, tasks: list, allow_reuse: bool
    ) -> dict | None:
        self._set(
            status=status,
            message=f"Running {label} benchmark ({len(tasks)} fixed tasks)...",
            benchmark_progress=f"0/{len(tasks)}",
        )

        def prog(i: int, n: int, row: dict) -> None:
            self._set(
                benchmark_progress=f"{i}/{n}",
                current_task=row["task_id"],
                current_domain=row["domain"],
            )

        try:
            res = self.bench.run(
                tasks, label, sid, self._stop.is_set, prog, allow_reuse=allow_reuse
            )
        except BenchmarkAborted:
            self._set(message=f"{label} benchmark aborted")
            return None
        d = res.summary()
        self._set(**{f"benchmark_{label}": d})
        return d

    def _learning_loop(self, sid: int, minutes: float, errors: list[str]) -> float:
        start = self.clock()
        self._loop_started = start
        try:
            return self._learning_loop_body(sid, minutes, errors, start)
        finally:
            self._loop_started = None

    def _learning_loop_body(
        self, sid: int, minutes: float, errors: list[str], start: float
    ) -> float:
        deadline = start + minutes * 60
        time_left = lambda: deadline - self.clock()
        should_abort = lambda: self._stop.is_set() or self.clock() >= deadline
        blocked = benchmark_hashes()
        adapters = self._custom_adapters
        if adapters is None:
            adapters = default_adapters(
                CachingHttp(self.http), lambda **kw: self.storage.record_source(**kw)
            )
        chain = DiscoveryChain(adapters, self.storage, self._rng, blocked)
        self._set(status="running", message="Learning loop running")
        self._loop_end_reason = "deadline"
        consecutive_no_task = 0
        while True:
            self._set(elapsed=min(self.clock() - start, minutes * 60))
            if self._stop.is_set():
                self._loop_end_reason = "stopped"
                break
            if time_left() <= 0:
                self._loop_end_reason = "deadline"
                break
            try:
                task = chain.next_task(sid)
                consecutive_no_task = 0
            except NoTaskAvailable as e:
                consecutive_no_task += 1
                msg = str(e)
                if msg not in errors:
                    errors.append(msg)
                self._set(
                    message="No task available; retrying shortly", errors=list(errors)
                )
                if (
                    consecutive_no_task
                    >= self.config.max_consecutive_discovery_failures
                ):
                    errors.append(
                        "giving up: discovery kept failing (check internet connection)"
                    )
                    self._loop_end_reason = "error"
                    break
                self._wait(min(5.0, max(0.0, time_left())))
                continue
            self.storage.set_task_status(task.task_id, "in_progress", sid)
            self._set(
                current_task=task.question.splitlines()[-1][:120]
                if task.question
                else task.task_id,
                current_domain=task.domain,
                message=f"Solving [{task.domain}] from {task.source_title}",
            )
            try:
                outcome = self.solver.solve(
                    task, sid, should_abort, time_left, self._emit
                )
            except FATAL as e:
                self.storage.set_task_status(task.task_id, "pending")
                errors.append(f"Ollama failure: {e}")
                self._loop_end_reason = "error"
                break
            self._record_outcome(outcome, errors)
        final = min(
            self.clock() - start, minutes * 60
        )  # learning phase only, never above the selected duration
        self._set(elapsed=final)
        return final

    def _emit(self, kind: str, **kw: Any) -> None:
        if kind == "retry":
            self._set(
                message=f"Retry {kw['attempt_no']}/{self.config.max_retries} on current task"
            )

    def _record_outcome(self, outcome, errors: list[str]) -> None:
        task, st = outcome.task, outcome.status
        with self._state_lock:
            s = self._state
            dom = s.domains.setdefault(task.domain, {"verified": 0, "failed": 0})
            for cat in outcome.failure_categories:
                s.failure_categories[cat] = s.failure_categories.get(cat, 0) + 1
            s.retries += outcome.retries
            if st == "aborted":
                # Invariant: attempted == verified + failed + invalid + pending. A task is attempted once a
                # model call was handed out for it (outcome.started), even if that call was still in flight
                # when the deadline/stop hit and so produced no recorded attempt. It stays pending: it is
                # neither verified, failed nor invalid. A task aborted before any model call is not counted.
                if outcome.started:
                    s.attempted += 1
                    s.pending += 1
            elif st == "verified":
                s.attempted += 1
                s.verified += 1
                dom["verified"] += 1
            elif st == "failed":
                s.attempted += 1
                s.failed += 1
                dom["failed"] += 1
                if not outcome.failure_categories:
                    s.failure_categories["other"] = (
                        s.failure_categories.get("other", 0) + 1
                    )
            elif st == "invalid":
                s.attempted += 1
                s.invalid += 1
        status_map = {
            "verified": "verified",
            "failed": "failed",
            "invalid": "abandoned",
            "aborted": "pending",
        }
        self.storage.set_task_status(task.task_id, status_map[st])
        if st == "aborted" and outcome.started:
            why = "stopped by the user" if self._stop.is_set() else "deadline reached"
            note = (
                f"task {task.task_id} [{task.domain}] pending: model call started but not finished "
                f"({why}); completed attempts before the cut-off: {len(outcome.attempts)}, "
                f"retries started: {outcome.retries}"
            )
            log.info(note)
            if note not in errors:
                errors.append(note)
        if st == "verified":
            lid = learn_from_outcome(self.memory, outcome)
            if lid:
                with self._state_lock:
                    self._state.lessons += 1
        if st == "invalid" and outcome.final and outcome.final.verification:
            errors.append(
                f"task {task.task_id} skipped: {outcome.final.verification.message}"
            )
        self._set(errors=list(errors))

    # -- finish ---------------------------------------------------------------
    def _finish(
        self,
        sid: int,
        stop_reason: str,
        before: dict | None,
        after: dict | None,
        errors: list[str],
        skip_note: str | None,
        elapsed: float,
    ) -> dict[str, Any]:
        self._set(status="reporting", message="Generating report...")
        status = {"deadline": "completed", "stopped": "stopped", "error": "failed"}[
            stop_reason
        ]
        bid = lambda b: b["benchmark_id"] if b else None
        self.storage.update_session(
            sid,
            status=status,
            ended_at=now_iso(),
            stop_reason=stop_reason,
            benchmark_before_id=bid(before),
            benchmark_after_id=bid(after),
            errors=errors,
        )
        snap = self.snapshot()
        session = self.storage.get_session(sid)
        b_before = self._full_bench(before)
        b_after = self._full_bench(after)
        lessons = [
            l
            for l in self.storage.list_lessons()
            if (self.storage.get_task_row(l["task_id"]) or {}).get("session_id") == sid
        ]
        report = build_report(
            session,
            snap,
            b_before,
            b_after,
            lessons,
            self.storage.list_sources(session["started_at"]),
            errors,
            skip_note,
        )
        jp, mp = write_reports(report, self.config.reports_dir)
        self.last_summary = report
        self._set(
            status=status if status != "completed" else "completed",
            message=f"Session {status}. Report: {mp.name}",
            report_paths=[str(jp), str(mp)],
            errors=list(errors),
        )
        return report

    def _full_bench(self, summary: dict | None) -> dict | None:
        if not summary:
            return None
        row = self.storage.get_benchmark(summary["benchmark_id"])
        return {
            "benchmark_id": row["benchmark_id"],
            "overall": row["overall"],
            "scores": row["scores"],
            "version": row["benchmark_version"],
            "n_tasks": row["n_tasks"],
            "reused_from": row["reused_from"],
            "created_at": row["created_at"],
            "results": row["results"],
        }

    # ------------------------------------------------------------------ queries / training
    def stats(self) -> dict[str, Any]:
        out = self.storage.stats()
        out["recent_sessions"] = [
            {k: s[k] for k in ("session_id", "started_at", "status", "planned_minutes")}
            for s in self.storage.list_sessions(5)
        ]
        out["recent_benchmarks"] = [
            {
                "benchmark_id": b["benchmark_id"],
                "label": b["label"],
                "overall": b["overall"],
                "n_tasks": b["n_tasks"],
                "created_at": b["created_at"],
            }
            for b in self.storage.list_benchmarks(5)
        ]
        return out

    def lessons(self, domain: str | None = None, limit: int = 50) -> list[dict]:
        return self.storage.list_lessons(domain, limit)

    def prepare_training(self) -> dict[str, Any]:
        manifest = training_dataset.prepare_training(
            self.storage, self.config.training_dir
        )
        hw = detect_hardware()
        manifest["hardware"] = hw.kind
        manifest["auto_finetune"] = False
        manifest["note"] = (
            "CPU-only: no fine-tuning started; dataset and lessons are maintained."
            if not hw.accelerated
            else "Accelerator detected; run `train-adapter` to train a LoRA adapter (never automatic)."
        )
        return manifest

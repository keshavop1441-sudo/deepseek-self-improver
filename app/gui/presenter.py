"""Toolkit-independent GUI logic (testable without Tk). The Tk window only renders what this returns."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.config import ALLOWED_DURATIONS
from app.controller import Controller, SessionBusy
from app.gui.worker import SessionWorker


def fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_bench(b: dict | None) -> str:
    if not b:
        return "-"
    parts = [f"overall {b['overall'] * 100:.1f}%"]
    parts += [f"{k} {v * 100:.0f}%" for k, v in b.get("scores", {}).items()]
    return " | ".join(parts)


class Presenter:
    def __init__(self, controller: Controller, clock: Callable[[], float] = time.monotonic):
        self.c = controller
        self.worker = SessionWorker(controller)
        self._clock = clock
        self._started_at: float | None = None
        self.last_error: str | None = None
        self.finished = False

    def start(self, minutes: int, quick_benchmark: bool = False, run_benchmarks: bool = True) -> str | None:
        """Returns an error string for the UI or None on success."""
        if minutes not in ALLOWED_DURATIONS:
            return f"Duration must be one of {ALLOWED_DURATIONS}"
        try:
            self.worker.start(minutes, run_benchmarks, quick_benchmark)  # returns immediately; work is off-thread
        except SessionBusy as e:
            return str(e)
        self.last_error, self.finished = None, False
        self._started_at = self._clock()
        return None

    def stop(self) -> None:
        self.worker.stop()

    def poll(self) -> None:
        """Call from the UI thread: apply worker results (completion / errors) to presenter state."""
        for kind, payload in self.worker.drain():
            if kind == "error":
                self.last_error = payload
            else:
                self.finished = True
            self._started_at = None

    @property
    def busy(self) -> bool:
        return self.worker.is_alive or self.c.is_running

    def view(self) -> dict[str, Any]:
        s = self.c.snapshot()
        before, after = s["benchmark_before"], s["benchmark_after"]
        delta = ""
        if before and after:
            d = (after["overall"] - before["overall"]) * 100
            delta = f"{d:+.1f} pts ({'IMPROVED' if d > 0 else 'no improvement'})"
        self.poll()
        busy = self.busy
        errors = list(s["errors"][-3:])
        status, message = s["status"], s["message"]
        if self.last_error:
            status = "error"
            if self.last_error not in errors:
                errors.append(self.last_error)
            message = self.last_error
        # Controller elapsed only advances between tasks; while a session runs, show live wall-clock time.
        elapsed = s["elapsed"] if self._started_at is None else max(s["elapsed"], self._clock() - self._started_at)
        return {
            "model": self.c.client.model, "status": status, "message": message,
            "elapsed": fmt_elapsed(elapsed), "attempted": s["attempted"], "verified": s["verified"],
            "failed": s["failed"], "retries": s["retries"], "lessons": s["lessons"],
            "task": s["current_task"], "domain": s["current_domain"],
            "bench_before": fmt_bench(before), "bench_after": fmt_bench(after), "bench_delta": delta,
            "bench_progress": s["benchmark_progress"], "start_enabled": not busy, "stop_enabled": busy,
            "report": s["report_paths"][-1] if s["report_paths"] else "", "errors": errors,
        }

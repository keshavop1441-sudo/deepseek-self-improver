"""Toolkit-independent GUI logic (testable without Tk). The Tk window only renders what this returns."""
from __future__ import annotations

from typing import Any

from app.config import ALLOWED_DURATIONS
from app.controller import Controller, SessionBusy


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
    def __init__(self, controller: Controller):
        self.c = controller

    def start(self, minutes: int, quick_benchmark: bool = False, run_benchmarks: bool = True) -> str | None:
        """Returns an error string for the UI or None on success."""
        if minutes not in ALLOWED_DURATIONS:
            return f"Duration must be one of {ALLOWED_DURATIONS}"
        try:
            self.c.start(minutes, run_benchmarks, quick_benchmark)
        except SessionBusy as e:
            return str(e)
        return None

    def stop(self) -> None:
        self.c.stop()

    def view(self) -> dict[str, Any]:
        s = self.c.snapshot()
        before, after = s["benchmark_before"], s["benchmark_after"]
        delta = ""
        if before and after:
            d = (after["overall"] - before["overall"]) * 100
            delta = f"{d:+.1f} pts ({'IMPROVED' if d > 0 else 'no improvement'})"
        busy = self.c.is_running
        return {
            "model": self.c.client.model, "status": s["status"], "message": s["message"],
            "elapsed": fmt_elapsed(s["elapsed"]), "attempted": s["attempted"], "verified": s["verified"],
            "failed": s["failed"], "retries": s["retries"], "lessons": s["lessons"],
            "task": s["current_task"], "domain": s["current_domain"],
            "bench_before": fmt_bench(before), "bench_after": fmt_bench(after), "bench_delta": delta,
            "bench_progress": s["benchmark_progress"], "start_enabled": not busy, "stop_enabled": busy,
            "report": s["report_paths"][-1] if s["report_paths"] else "", "errors": s["errors"][-3:],
        }

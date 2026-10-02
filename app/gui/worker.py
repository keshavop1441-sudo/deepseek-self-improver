"""Background worker for the GUI: runs the Controller session off the Tk thread.

The worker thread never touches Tk. It only calls the controller and posts results to a thread-safe
queue; the Tk main thread drains that queue (see Presenter.poll) and owns every widget.
"""
from __future__ import annotations

import queue
import threading
from typing import Any

from app.controller import Controller, SessionBusy


class SessionWorker:
    def __init__(self, controller: Controller):
        self.c = controller
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self, minutes: float, run_benchmarks: bool = True, quick_benchmark: bool = False) -> None:
        with self._lock:
            if self.is_alive or self.c.is_running:
                raise SessionBusy("A session is already running")
            self._thread = threading.Thread(target=self._run, args=(minutes, run_benchmarks, quick_benchmark),
                                            name="gui-session-worker", daemon=True)
            self._thread.start()

    def _run(self, minutes: float, run_benchmarks: bool, quick: bool) -> None:
        try:
            report = self.c.run_session(minutes, run_benchmarks, quick)
        except Exception as e:  # noqa: BLE001 - every failure must reach the UI
            self.events.put(("error", str(e) or type(e).__name__))
        else:
            self.events.put(("finished", report))

    def stop(self) -> None:
        """Signal the controller to stop cooperatively; never kills the thread or touches the DB."""
        self.c.stop()

    def drain(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def join(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

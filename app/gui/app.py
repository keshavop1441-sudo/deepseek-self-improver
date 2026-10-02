"""Tkinter front-end. All logic lives in Presenter/Controller; this file only draws widgets."""

from __future__ import annotations

import gc

from app.config import ALLOWED_DURATIONS
from app.controller import Controller
from app.gui.presenter import Presenter

LABELS = [
    ("Model", "model"),
    ("Mode", "mode"),
    ("Status", "status"),
    ("Session elapsed", "session_elapsed"),
    ("Learning elapsed", "elapsed"),
    ("Current domain", "domain"),
    ("Current task", "task"),
    ("Tasks attempted", "attempted"),
    ("Tasks verified", "verified"),
    ("Failed", "failed"),
    ("Retries", "retries"),
    ("Lessons learned", "lessons"),
    ("Benchmark before", "bench_before"),
    ("Benchmark after", "bench_after"),
    ("Benchmark delta", "bench_delta"),
    ("Benchmark progress", "bench_progress"),
]


REFRESH_MS = 200
CLOSE_JOIN_SECONDS = 10.0


class GuiApp:
    """Builds the widgets and refreshes them. Runs only on the Tk main thread; the session runs in a worker."""

    def __init__(self, root, pres: Presenter):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.pres = root, pres
        root.title("DeepSeek Self-Improver (local, Ollama)")
        root.geometry("760x560")
        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")
        ttk.Label(top, text="Duration (minutes):").pack(side="left")
        self.minutes = tk.IntVar(value=ALLOWED_DURATIONS[0])
        self.duration_buttons = []
        for m in ALLOWED_DURATIONS:
            rb = ttk.Radiobutton(top, text=str(m), value=m, variable=self.minutes)
            rb.pack(side="left", padx=4)
            self.duration_buttons.append(rb)
        self.stop_btn = ttk.Button(top, text="STOP", command=pres.stop)
        self.stop_btn.pack(side="right")
        self.self_improve_btn = ttk.Button(
            top, text="SELF IMPROVE", command=self.on_self_improve
        )
        self.self_improve_btn.pack(side="right", padx=6)

        bench = ttk.Frame(frm)
        bench.pack(fill="x", pady=(6, 0))
        ttk.Label(bench, text="Optional benchmarks:").pack(side="left")
        self.measure = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bench,
            text="measure before/after during Self Improve",
            variable=self.measure,
        ).pack(side="left", padx=6)
        self.quick = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bench, text="quick benchmark (12 tasks)", variable=self.quick
        ).pack(side="left", padx=6)
        self.start_btn = ttk.Button(
            bench, text="BENCHMARKED RUN", command=self.on_start
        )
        self.start_btn.pack(side="right")

        grid = ttk.Frame(frm)
        grid.pack(fill="x", pady=10)
        self.vars: dict[str, tk.StringVar] = {}
        for i, (label, key) in enumerate(LABELS):
            ttk.Label(grid, text=label + ":", width=20).grid(
                row=i, column=0, sticky="w"
            )
            self.vars[key] = tk.StringVar(value="-")
            ttk.Label(
                grid, textvariable=self.vars[key], wraplength=520, justify="left"
            ).grid(row=i, column=1, sticky="w")
        self.msg = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.msg, wraplength=720, foreground="#444").pack(
            fill="x"
        )
        self.err = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.err, wraplength=720, foreground="#a00").pack(
            fill="x", pady=4
        )
        self.report = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.report, wraplength=720).pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_self_improve(self) -> None:
        problem = self.pres.self_improve(
            self.minutes.get(), self.measure.get(), self.quick.get()
        )
        self.err.set(problem or "")
        self.refresh_once()

    def on_start(self) -> None:
        problem = self.pres.start(
            self.minutes.get(), self.quick.get()
        )  # non-blocking: session runs in a worker
        self.err.set(problem or "")
        self.refresh_once()

    def refresh_once(self) -> None:
        v = self.pres.view()
        for _, key in LABELS:
            self.vars[key].set(str(v[key]))
        self.msg.set(v["message"])
        self.err.set(
            "\n".join(v["errors"])
            if v["status"] == "error" or v["errors"]
            else self.err.get()
        )
        self.report.set(f"Report: {v['report']}" if v["report"] else "")
        self.start_btn.state(["!disabled"] if v["start_enabled"] else ["disabled"])
        self.self_improve_btn.state(
            ["!disabled"] if v["start_enabled"] else ["disabled"]
        )
        self.stop_btn.state(["!disabled"] if v["stop_enabled"] else ["disabled"])

    def refresh(self) -> None:
        self.refresh_once()
        self.root.after(REFRESH_MS, self.refresh)

    def on_close(self) -> None:
        self.pres.stop()  # cooperative: the controller finishes/persists its state
        self.root.destroy()


def launch_gui(controller: Controller | None = None) -> int:
    try:
        import tkinter as tk
    except ImportError:
        print(
            "Tkinter is not available in this Python installation. Reinstall Python with the 'tcl/tk' option "
            "or use the CLI: python -m app.main self-improve --minutes 30"
        )
        return 2
    controller = controller or Controller()
    pres = Presenter(controller)
    root = tk.Tk()
    app = GuiApp(root, pres)
    app.refresh()
    root.mainloop()
    del app, root
    gc.collect()  # finalise Tk objects on the main thread, never inside the worker thread
    pres.worker.join(
        CLOSE_JOIN_SECONDS
    )  # let a stopped session persist before the process exits
    return 0

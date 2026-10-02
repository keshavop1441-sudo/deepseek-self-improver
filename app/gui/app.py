"""Tkinter front-end. All logic lives in Presenter/Controller; this file only draws widgets."""
from __future__ import annotations

from app.config import ALLOWED_DURATIONS
from app.controller import Controller
from app.gui.presenter import Presenter

LABELS = [("Model", "model"), ("Status", "status"), ("Elapsed", "elapsed"), ("Current domain", "domain"),
          ("Current task", "task"), ("Tasks attempted", "attempted"), ("Verified correct", "verified"),
          ("Failed", "failed"), ("Retries", "retries"), ("Lessons learned", "lessons"),
          ("Benchmark before", "bench_before"), ("Benchmark after", "bench_after"), ("Benchmark delta", "bench_delta"),
          ("Benchmark progress", "bench_progress")]


def launch_gui(controller: Controller | None = None) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter is not available in this Python installation. Reinstall Python with the 'tcl/tk' option "
              "or use the CLI: python -m app.main run --minutes 30")
        return 2
    controller = controller or Controller()
    pres = Presenter(controller)
    root = tk.Tk()
    root.title("DeepSeek Self-Improver (local, Ollama)")
    root.geometry("760x560")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    top = ttk.Frame(frm)
    top.pack(fill="x")
    ttk.Label(top, text="Duration (minutes):").pack(side="left")
    minutes = tk.IntVar(value=ALLOWED_DURATIONS[0])
    for m in ALLOWED_DURATIONS:
        ttk.Radiobutton(top, text=str(m), value=m, variable=minutes).pack(side="left", padx=4)
    quick = tk.BooleanVar(value=False)
    ttk.Checkbutton(top, text="quick benchmark (12 tasks)", variable=quick).pack(side="left", padx=12)
    start_btn = ttk.Button(top, text="START")
    stop_btn = ttk.Button(top, text="STOP")
    stop_btn.pack(side="right")
    start_btn.pack(side="right", padx=6)

    grid = ttk.Frame(frm)
    grid.pack(fill="x", pady=10)
    vars_: dict[str, tk.StringVar] = {}
    for i, (label, key) in enumerate(LABELS):
        ttk.Label(grid, text=label + ":", width=20).grid(row=i, column=0, sticky="w")
        vars_[key] = tk.StringVar(value="-")
        ttk.Label(grid, textvariable=vars_[key], wraplength=520, justify="left").grid(row=i, column=1, sticky="w")
    msg = tk.StringVar(value="")
    ttk.Label(frm, textvariable=msg, wraplength=720, foreground="#444").pack(fill="x")
    err = tk.StringVar(value="")
    ttk.Label(frm, textvariable=err, wraplength=720, foreground="#a00").pack(fill="x", pady=4)
    report = tk.StringVar(value="")
    ttk.Label(frm, textvariable=report, wraplength=720).pack(fill="x")

    def on_start() -> None:
        problem = pres.start(minutes.get(), quick.get())
        err.set(problem or "")

    start_btn.config(command=on_start)
    stop_btn.config(command=pres.stop)

    def refresh() -> None:
        v = pres.view()
        for _, key in LABELS:
            vars_[key].set(str(v[key]))
        msg.set(v["message"])
        err.set("\n".join(v["errors"]) if v["status"] == "error" or v["errors"] else err.get())
        report.set(f"Report: {v['report']}" if v["report"] else "")
        start_btn.state(["!disabled"] if v["start_enabled"] else ["disabled"])
        stop_btn.state(["!disabled"] if v["stop_enabled"] else ["disabled"])
        root.after(500, refresh)

    def on_close() -> None:
        pres.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    root.mainloop()
    return 0

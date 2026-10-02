"""Real Tk smoke test; skipped automatically where Tk or a display is unavailable."""
import pytest

tk = pytest.importorskip("tkinter")

from app.gui import app as gapp  # noqa: E402
from tests.helpers import FakeClock, FakeOllama, make_oracle  # noqa: E402


def test_tk_window_start_button_runs_session(make_controller, monkeypatch):
    try:
        probe = tk.Tk()
        probe.destroy()
    except tk.TclError:
        pytest.skip("no display")
    from tkinter import ttk
    clock = FakeClock()
    c, _, _ = make_controller(FakeOllama(make_oracle(), on_chat=lambda n: clock.advance(10)), clock=clock)
    real = tk.Tk.mainloop

    def find(w, text):
        for ch in w.winfo_children():
            if isinstance(ch, ttk.Button) and ch.cget("text") == text:
                return ch
            r = find(ch, text)
            if r:
                return r

    def fake_mainloop(self, n=0):
        find(self, "BENCHMARKED RUN").invoke()
        self.after(2500, self.destroy)
        real(self)
    monkeypatch.setattr(tk.Tk, "mainloop", fake_mainloop)
    assert gapp.launch_gui(c) == 0
    assert c.snapshot()["status"] == "completed" and c.snapshot()["verified"] >= 1


def test_tk_self_improve_button_runs_without_benchmarks(make_controller, monkeypatch):
    try:
        probe = tk.Tk()
        probe.destroy()
    except tk.TclError:
        pytest.skip("no display")
    from tkinter import ttk
    clock = FakeClock()
    c, _, _ = make_controller(FakeOllama(make_oracle(), on_chat=lambda n: clock.advance(10)), clock=clock)
    real = tk.Tk.mainloop

    def find(w, text):
        for ch in w.winfo_children():
            if isinstance(ch, ttk.Button) and ch.cget("text") == text:
                return ch
            r = find(ch, text)
            if r:
                return r

    def fake_mainloop(self, n=0):
        find(self, "SELF IMPROVE").invoke()
        self.after(2500, self.destroy)
        real(self)
    monkeypatch.setattr(tk.Tk, "mainloop", fake_mainloop)
    assert gapp.launch_gui(c) == 0
    s = c.snapshot()
    assert s["status"] == "completed" and s["mode"] == "self_improve"
    assert s["benchmark_before"] is None and s["benchmark_after"] is None and s["verified"] >= 1

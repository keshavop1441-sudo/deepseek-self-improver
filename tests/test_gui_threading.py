"""GUI duration options + background-worker behaviour. Uses a fake controller; no Ollama needed.
Tk-dependent tests skip automatically without Tk or a display (run under xvfb-run to exercise them)."""

import gc
import threading
import time
from types import SimpleNamespace

import pytest

from app.config import ALLOWED_DURATIONS
from app.controller import PreflightError, SessionBusy
from app.gui.presenter import Presenter


class FakeController:
    """Blocks inside run_session until released, like a long DeepSeek call; records the calling thread."""

    def __init__(self, fail: Exception | None = None, clock=None):
        self.clock = clock
        self._t0 = None
        self.client = SimpleNamespace(model="fake-model")
        self.release = threading.Event()
        self.entered = threading.Event()
        self.stop_requested = threading.Event()
        self.fail = fail
        self.run_thread = None
        self.calls = []
        self._running = threading.Lock()
        self._state = {
            "status": "idle",
            "message": "",
            "elapsed": 0.0,
            "attempted": 0,
            "verified": 0,
            "failed": 0,
            "retries": 0,
            "lessons": 0,
            "current_task": "",
            "current_domain": "",
            "benchmark_before": None,
            "benchmark_after": None,
            "benchmark_progress": "",
            "report_paths": [],
            "errors": [],
        }

    @property
    def is_running(self):
        return self._running.locked()

    def snapshot(self):
        snap = dict(self._state)
        if self.clock is not None and self._t0 is not None and self.is_running:
            snap["elapsed"] = (
                self.clock() - self._t0
            )  # like Controller.snapshot(): live, not per-task
        return snap

    def run_session(self, minutes, run_benchmarks=True, quick=False):
        if not self._running.acquire(blocking=False):
            raise SessionBusy("A session is already running")
        try:
            self.run_thread = threading.current_thread()
            self._t0 = self.clock() if self.clock else None
            self.calls.append((minutes, run_benchmarks, quick))
            self._state.update(status="running", message="thinking")
            self.entered.set()
            while not (self.release.is_set() or self.stop_requested.is_set()):
                time.sleep(0.01)
            if self.fail:
                self._state.update(status="error")
                raise self.fail
            self._state.update(
                status="stopped" if self.stop_requested.is_set() else "completed",
                report_paths=["reports/r.md"],
                verified=1,
            )
            return {"ok": True}
        finally:
            self._running.release()

    def stop(self):
        self.stop_requested.set()


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------- durations (no Tk needed)
def test_two_minutes_is_allowed_and_ordered():
    assert ALLOWED_DURATIONS == (2, 20, 30, 45)


@pytest.mark.parametrize("minutes", [2, 20, 30, 45])
def test_presenter_accepts_every_allowed_duration(minutes):
    c = FakeController()
    c.release.set()
    p = Presenter(c)
    assert p.start(minutes) is None
    p.worker.join(5)
    assert c.calls == [(minutes, True, False)]


@pytest.mark.parametrize("minutes", [1, 3, 7, 60])
def test_presenter_rejects_other_durations(minutes):
    c = FakeController()
    assert Presenter(c).start(minutes) is not None and c.calls == []


def test_cli_accepts_two_minutes():
    from app.main import build_parser

    assert build_parser().parse_args(["run", "--minutes", "2"]).minutes == 2


# ---------------------------------------------------------------- worker / presenter (no Tk needed)
def test_start_returns_immediately_and_runs_off_main_thread():
    c = FakeController()
    p = Presenter(c)
    t0 = time.monotonic()
    assert p.start(2) is None
    assert time.monotonic() - t0 < 0.5
    assert c.entered.wait(5) and c.run_thread is not threading.main_thread()
    assert p.view()["stop_enabled"] and not p.view()["start_enabled"]
    assert p.start(2) is not None  # second start refused while busy
    c.release.set()
    p.worker.join(5)


def test_elapsed_advances_while_worker_blocked():
    clock = Clock()
    c = FakeController(clock=clock)
    p = Presenter(c)
    p.start(2)
    assert c.entered.wait(5)
    seen = []
    for _ in range(3):
        clock.t += 7
        seen.append(p.view()["elapsed"])
    assert seen == ["00:00:07", "00:00:14", "00:00:21"]
    c.release.set()
    p.worker.join(5)


def test_stop_while_worker_running_ends_session_safely():
    c = FakeController()
    p = Presenter(c)
    p.start(2)
    assert c.entered.wait(5)
    p.stop()
    assert c.stop_requested.is_set()
    p.worker.join(5)
    v = p.view()
    assert v["status"] == "stopped" and v["start_enabled"] and not v["stop_enabled"]


def test_completion_reaches_ui_state():
    c = FakeController()
    c.release.set()
    p = Presenter(c)
    p.start(2)
    p.worker.join(5)
    v = p.view()
    assert (
        p.finished
        and v["status"] == "completed"
        and v["report"] == "reports/r.md"
        and v["start_enabled"]
    )


@pytest.mark.parametrize(
    "exc", [RuntimeError("boom"), PreflightError("Ollama not reachable")]
)
def test_worker_errors_reach_ui(exc):
    c = FakeController(fail=exc)
    c.release.set()
    p = Presenter(c)
    p.start(2)
    p.worker.join(5)
    v = p.view()
    assert (
        v["status"] == "error" and str(exc) in v["errors"] and str(exc) in v["message"]
    )
    assert v["start_enabled"]


# ---------------------------------------------------------------- real Tk window (xvfb)
tk = pytest.importorskip("tkinter")


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass
    # Tk variables must be finalised on the main thread; a GC pass in a worker thread would abort Tcl.
    gc.collect()


def pump(root, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update()
        time.sleep(0.005)


def make_app(root, controller):
    from app.gui.app import GuiApp

    app = GuiApp(root, Presenter(controller))
    root.update()
    return app


def test_gui_shows_two_as_selectable_duration(root):
    app = make_app(root, FakeController())
    assert [b.cget("text") for b in app.duration_buttons] == ["2", "20", "30", "45"]
    assert app.minutes.get() == 2
    for b, m in zip(app.duration_buttons, ALLOWED_DURATIONS, strict=True):
        b.invoke()
        assert app.minutes.get() == m


def test_gui_start_does_not_block_event_loop_and_elapsed_updates(root):
    clock = Clock()
    c = FakeController(clock=clock)
    app = make_app(root, c)
    ticks = []
    root.after(20, lambda: ticks.append(1))
    t0 = time.monotonic()
    app.start_btn.invoke()
    assert time.monotonic() - t0 < 0.5
    assert c.entered.wait(5)
    # Event loop still serves timers while the "model" is blocked
    repeat = []

    def tick():
        repeat.append(1)
        clock.t += 1
        root.after(10, tick)

    tick()
    app.refresh()
    pump(root, 0.5)
    assert ticks and len(repeat) > 5
    first = app.vars["elapsed"].get()
    pump(root, 0.3)
    assert app.vars["elapsed"].get() != first and app.vars["elapsed"].get() != "-"
    assert app.vars["status"].get() == "running"
    c.release.set()
    app.pres.worker.join(5)


def test_gui_stop_button_responsive_while_running(root):
    c = FakeController()
    app = make_app(root, c)
    app.refresh()
    app.start_btn.invoke()
    assert c.entered.wait(5)
    pump(root, 0.3)
    assert "disabled" not in app.stop_btn.state()
    app.stop_btn.invoke()
    assert c.stop_requested.is_set()
    assert wait_for(
        lambda: (pump(root, 0.05), app.vars["status"].get() == "stopped")[1]
    )
    assert "disabled" not in app.start_btn.state()


def test_gui_completion_and_errors_update_widgets(root):
    ok = FakeController()
    ok.release.set()
    app = make_app(root, ok)
    app.refresh()
    app.start_btn.invoke()
    assert wait_for(
        lambda: (pump(root, 0.05), app.vars["status"].get() == "completed")[1]
    )
    assert "r.md" in app.report.get()

    bad = FakeController(fail=RuntimeError("model exploded"))
    bad.release.set()
    r2 = tk.Tk()
    try:
        app2 = make_app(r2, bad)
        app2.refresh()
        app2.start_btn.invoke()
        assert wait_for(lambda: (pump(r2, 0.05), "model exploded" in app2.err.get())[1])
        assert app2.vars["status"].get() == "error"
    finally:
        r2.destroy()
        app2 = None
        gc.collect()


# ---- real Controller + Presenter (+ GuiApp) end-to-end: the value the GUI shows while solver.solve() is blocked


def _blocked_session(make_controller, minutes=2, run_benchmarks=False):
    from tests.helpers import FakeClock
    from tests.test_deadline import HangingOllama

    clock = FakeClock()
    fake = HangingOllama(ok_calls=0)
    c, _, _ = make_controller(fake, clock=clock)
    return c, fake, clock


def _finish_session(pres, fake):
    fake.unblock.set()
    pres.worker.join(5)


def test_presenter_elapsed_advances_second_by_second_while_solve_is_blocked(
    make_controller,
):
    c, fake, clock = _blocked_session(make_controller)
    pres = Presenter(c)
    assert pres.view()["elapsed"] == "00:00:00" and pres.view()["status"] == "idle"
    assert pres.start(2, run_benchmarks=False) is None
    assert fake.hanging.wait(5)  # worker is now inside solver.solve -> client.chat
    shown = [pres.view()["elapsed"]]
    for _ in range(4):
        clock.advance(1)
        shown.append(pres.view()["elapsed"])
    assert shown == ["00:00:00", "00:00:01", "00:00:02", "00:00:03", "00:00:04"]
    assert (
        pres.worker.is_alive and pres.view()["status"] == "running"
    )  # still blocked the whole time
    clock.advance(10_000)
    assert pres.view()["elapsed"] == "00:02:00"  # capped at the selected duration
    _finish_session(pres, fake)
    assert pres.view()["elapsed"] == "00:02:00"


def test_presenter_stop_freezes_elapsed_and_excludes_nothing_after(make_controller):
    c, fake, clock = _blocked_session(make_controller)
    pres = Presenter(c)
    pres.start(2, run_benchmarks=False)
    assert fake.hanging.wait(5)
    clock.advance(9)
    pres.stop()
    _finish_session(pres, fake)
    clock.advance(500)
    v = pres.view()
    assert v["elapsed"] == "00:00:09" and v["status"] == "stopped"


def test_presenter_elapsed_excludes_benchmark_phases(make_controller):
    from tests.helpers import FakeClock, FakeOllama, make_oracle

    clock = FakeClock()
    seen = []
    holder = {}

    def on_chat(n):
        clock.advance(10)
        v = holder["p"].view()
        seen.append((v["status"], v["elapsed"]))

    c, _, _ = make_controller(
        FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=on_chat), clock=clock
    )
    holder["p"] = pres = Presenter(c)
    pres.start(2, quick_benchmark=True, run_benchmarks=True)
    pres.worker.join(10)
    by_status = {}
    for st, el in seen:
        by_status.setdefault(st, []).append(el)
    assert set(by_status["benchmark_before"]) == {
        "00:00:00"
    }  # clock moved, timer did not
    assert (
        max(by_status["running"]) <= "00:02:00"
        and by_status["running"][-1] > "00:00:00"
    )
    assert pres.view()["elapsed"] == "00:02:00"  # not 2:00 + benchmark time


def test_gui_elapsed_stringvar_advances_with_real_controller_while_blocked(
    root, make_controller
):
    c, fake, clock = _blocked_session(make_controller)
    app = make_app(root, c)
    assert app.vars["elapsed"].get() == "-" and (app.refresh_once() or True)
    assert app.vars["elapsed"].get() == "00:00:00"
    assert app.pres.start(2, run_benchmarks=False) is None
    assert fake.hanging.wait(5)
    shown = []
    for _ in range(4):
        clock.advance(1)
        app.refresh_once()  # the exact call GuiApp.refresh() makes every tick
        root.update()
        shown.append(app.vars["elapsed"].get())
    assert shown == ["00:00:01", "00:00:02", "00:00:03", "00:00:04"]
    assert app.pres.worker.is_alive and app.vars["status"].get() == "running"
    _finish_session(app.pres, fake)
    app.refresh_once()

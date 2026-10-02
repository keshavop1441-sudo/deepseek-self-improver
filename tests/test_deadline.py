"""Hard learning-deadline and STOP behaviour while a model call is in flight (fake clock, fake Ollama)."""
import threading
import time

from app.gui.presenter import Presenter
from tests.helpers import FakeClock, FakeOllama, make_oracle

MINUTES = 2
LIMIT = MINUTES * 60


class HangingOllama(FakeOllama):
    """Answers normally for the first `ok_calls` chats, then blocks like a very long generation."""

    def __init__(self, ok_calls=0, **kw):
        super().__init__(make_oracle(wrong_first_gcd=False), **kw)
        self.ok_calls = ok_calls
        self.hanging = threading.Event()
        self.unblock = threading.Event()

    def __call__(self, method, url, payload, timeout):
        if url.endswith("/api/chat") and len(self.chat_calls) >= self.ok_calls:
            self.chat_calls.append(payload)
            self.hanging.set()
            self.unblock.wait(30)          # far longer than any test waits
            return 500, {"error": "released after abort"}
        return super().__call__(method, url, payload, timeout)


def run_in_thread(c, **kw):
    box = {}

    def target():
        box["rep"] = c.run_session(MINUTES, run_benchmarks=False, **kw)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, box


def finish(t, box, fake, timeout=5):
    t.join(timeout)
    fake.unblock.set()
    assert not t.is_alive(), "session did not terminate promptly (hung in model call)"
    return box["rep"]


def test_deadline_aborts_inflight_request_and_reports_completed_deadline(make_controller):
    clock = FakeClock()
    fake = HangingOllama(ok_calls=1, on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    t, box = run_in_thread(c)
    assert fake.hanging.wait(5)
    clock.advance(LIMIT)                                  # deadline arrives mid-generation
    rep = finish(t, box, fake)
    assert rep["status"] == "completed" and rep["stop_reason"] == "deadline"
    assert rep["verified"] == 1 and rep["failed"] == 0 and rep["attempted"] == 1
    assert rep["duration_seconds"] <= LIMIT + 1


def test_inflight_task_is_pending_not_failed_and_no_lesson(make_controller):
    clock = FakeClock()
    fake = HangingOllama(ok_calls=1, on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    t, box = run_in_thread(c)
    assert fake.hanging.wait(5)
    clock.advance(LIMIT)
    rep = finish(t, box, fake)
    rows = c.storage._conn.execute("SELECT status FROM tasks ORDER BY rowid").fetchall()
    statuses = [r[0] for r in rows]
    assert statuses == ["verified", "pending"], statuses
    assert "failed" not in statuses and "in_progress" not in statuses
    assert rep["lessons_learned"] == len(rep["lessons"]) == len(c.storage.list_lessons())
    assert c.storage._conn.execute("SELECT COUNT(*) FROM attempts WHERE parse_ok=1 AND task_id IN "
                                  "(SELECT task_id FROM tasks WHERE status='pending')").fetchone()[0] == 0


def test_stop_aborts_inflight_request_safely(make_controller):
    clock = FakeClock()
    fake = HangingOllama(ok_calls=1, on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    t, box = run_in_thread(c)
    assert fake.hanging.wait(5)
    c.stop()
    rep = finish(t, box, fake)
    assert rep["status"] == "stopped" and rep["stop_reason"] == "stopped"
    assert rep["verified"] == 1 and rep["failed"] == 0
    statuses = [r[0] for r in c.storage._conn.execute("SELECT status FROM tasks ORDER BY rowid")]
    assert statuses == ["verified", "pending"]
    assert not c.is_running


def test_request_budget_never_exceeds_time_left(make_controller):
    """The per-request timeout handed to Ollama is capped by the remaining session time, not the 300 s default."""
    clock = FakeClock()
    seen = []
    fake = HangingOllama(ok_calls=0)
    real = fake.__call__

    def spy(method, url, payload, timeout):
        if url.endswith("/api/chat"):
            seen.append(timeout)
        return real(method, url, payload, timeout)
    c, _, _ = make_controller(spy, clock=clock)
    t, box = run_in_thread(c)
    assert fake.hanging.wait(5)
    clock.advance(LIMIT)
    finish(t, box, fake)
    assert seen and max(seen) <= LIMIT


def test_before_benchmark_not_counted_against_learning_deadline(make_controller):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    rep = c.run_session(MINUTES, run_benchmarks=True, quick_benchmark=True)
    assert rep["benchmark_before"] is not None and rep["status"] == "completed" and rep["stop_reason"] == "deadline"
    assert rep["verified"] >= 1                                       # learning still got its full 120 s
    assert rep["duration_seconds"] <= LIMIT + 1


def test_normal_tasks_complete_without_deadline_interference(make_controller):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    rep = c.run_session(MINUTES, run_benchmarks=False)
    assert rep["status"] == "completed" and rep["stop_reason"] == "deadline"
    assert rep["verified"] >= 5 and rep["failed"] == 0 and rep["lessons_learned"] >= 1


def test_gui_two_minute_mode_elapsed_bounded_during_blocked_call(make_controller):
    clock = FakeClock()
    fake = HangingOllama(ok_calls=0)
    c, _, _ = make_controller(fake, clock=clock)
    p = Presenter(c)
    assert p.start(2, run_benchmarks=False) is None
    assert fake.hanging.wait(5)
    clock.advance(30)
    assert p.view()["elapsed"] == "00:00:30"
    clock.advance(500)                                    # blocked call, session limit long gone
    deadline = time.monotonic() + 5
    while p.view()["status"] not in ("completed", "stopped", "error") and time.monotonic() < deadline:
        time.sleep(0.02)
    fake.unblock.set()
    p.worker.join(5)
    v = p.view()
    assert v["status"] == "completed" and v["elapsed"] <= "00:02:01" and v["start_enabled"]


def test_real_clock_deadline_cuts_off_hung_request(make_controller):
    """No fake clock: a 3 s learning window must end ~3 s in even though the model call hangs (300 s default)."""
    fake = HangingOllama(ok_calls=0)
    c, _, _ = make_controller(fake, clock=time.monotonic)
    box = {}
    t0 = time.monotonic()
    t = threading.Thread(target=lambda: box.update(rep=c.run_session(0.05, run_benchmarks=False)), daemon=True)
    t.start()
    t.join(10)
    fake.unblock.set()
    assert not t.is_alive()
    assert time.monotonic() - t0 < 6
    rep = box["rep"]
    assert rep["status"] == "completed" and rep["stop_reason"] == "deadline" and rep["failed"] == 0
    assert [r[0] for r in c.storage._conn.execute("SELECT status FROM tasks")] == ["pending"]

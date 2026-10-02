"""Session elapsed (whole session, preflight..report) vs learning elapsed (learning phase only), via Presenter."""

from app.gui.presenter import Presenter
from tests.helpers import FakeClock, FakeOllama, make_oracle
from tests.test_deadline import HangingOllama

PREFLIGHT_COST = 3
LIMIT = 120


class PreflightCost:
    """Mixin: the first /api/version call (preflight) costs PREFLIGHT_COST fake seconds."""

    clock: FakeClock
    on_preflight = None

    def __call__(self, method, url, payload, timeout):
        if url.endswith("/api/version") and not getattr(self, "_pf_done", False):
            self._pf_done = True
            self.clock.advance(PREFLIGHT_COST)
            if self.on_preflight:
                self.on_preflight()
        return super().__call__(method, url, payload, timeout)


class SlowPreflightHanging(PreflightCost, HangingOllama):
    pass


class SlowPreflightOllama(PreflightCost, FakeOllama):
    pass


def test_session_elapsed_through_every_phase_and_learning_excludes_the_rest(
    make_controller,
):
    clock = FakeClock()
    t0 = clock()
    seen = []
    holder = {}

    def record():
        v = holder["p"].view()
        seen.append((v["status"], v["session_elapsed"], v["elapsed"]))

    def on_chat(n):
        clock.advance(10)
        record()

    fake = SlowPreflightOllama(make_oracle(wrong_first_gcd=False), on_chat=on_chat)
    fake.clock, fake.on_preflight = clock, record
    c, _, _ = make_controller(fake, clock=clock)
    holder["p"] = pres = Presenter(c)

    def on_state(snap):
        if snap["status"] == "reporting" and not holder.get("reported"):
            holder["reported"] = True
            record()
            clock.advance(5)  # report generation takes 5 s
            record()

    c.add_listener(on_state)
    assert pres.view()["session_elapsed"] == "00:00:00"
    pres.start(2, quick_benchmark=True, run_benchmarks=True)
    pres.worker.join(10)
    by = {}
    for status, sess, learn in seen:
        by.setdefault(status, []).append((sess, learn))

    assert by["preflight"] == [
        ("00:00:03", "00:00:00")
    ]  # session clock runs in preflight; learning does not
    before = by["benchmark_before"]
    assert [s for s, _ in before] == sorted(s for s, _ in before) and before[0][
        0
    ] > "00:00:03"
    assert {learn for _, learn in before} == {
        "00:00:00"
    }  # learning elapsed stays 0 during benchmark_before
    running = by["running"]
    assert (
        running[0][0] > before[-1][0]
    )  # session keeps counting into the learning phase
    assert [lrn for _, lrn in running] == sorted(lrn for _, lrn in running) and running[
        -1
    ][1] > "00:00:00"
    assert max(lrn for _, lrn in running) <= "00:02:00"
    after = by["benchmark_after"]
    assert {learn for _, learn in after} == {
        "00:02:00"
    }  # learning frozen at the limit during benchmark_after
    assert (
        after[-1][0] > after[0][0] > running[-1][0]
    )  # ...while session elapsed keeps advancing
    reporting = by["reporting"]
    assert reporting[1][0] > reporting[0][0] and {learn for _, learn in reporting} == {
        "00:02:00"
    }

    total = clock() - t0
    v = pres.view()
    assert v["status"] == "completed"
    assert v["elapsed"] == "00:02:00"
    assert (
        v["session_elapsed"]
        == f"{int(total) // 3600:02d}:{int(total) % 3600 // 60:02d}:{int(total) % 60:02d}"
    )
    assert (
        total > LIMIT + PREFLIGHT_COST + 5
    )  # benchmarks really cost clock time, none of it in learning elapsed
    clock.advance(500)
    assert (
        pres.view()["session_elapsed"] == v["session_elapsed"]
    )  # frozen once the session finished


def test_session_elapsed_live_while_solve_blocked_and_learning_separate(
    make_controller,
):
    clock = FakeClock()
    fake = SlowPreflightHanging(ok_calls=0)
    fake.clock = clock
    c, _, _ = make_controller(fake, clock=clock)
    pres = Presenter(c)
    assert pres.start(2, run_benchmarks=False) is None
    assert fake.hanging.wait(5)  # worker is inside solver.solve()
    shown = []
    for _ in range(3):
        v = pres.view()
        shown.append((v["session_elapsed"], v["elapsed"]))
        clock.advance(1)
    assert shown == [
        ("00:00:03", "00:00:00"),
        ("00:00:04", "00:00:01"),
        ("00:00:05", "00:00:02"),
    ]
    assert pres.worker.is_alive and pres.view()["status"] == "running"
    clock.advance(10_000)  # far past the limit while still blocked
    v = pres.view()
    assert v["elapsed"] == "00:02:00" and v["session_elapsed"] == "02:46:46"
    fake.unblock.set()
    pres.worker.join(5)
    v = pres.view()
    assert v["elapsed"] == "00:02:00" and v["session_elapsed"] == "02:46:46"


def test_stop_freezes_session_and_learning_elapsed(make_controller):
    clock = FakeClock()
    fake = SlowPreflightHanging(ok_calls=0)
    fake.clock = clock
    c, _, _ = make_controller(fake, clock=clock)
    pres = Presenter(c)
    pres.start(2, run_benchmarks=False)
    assert fake.hanging.wait(5)
    clock.advance(9)
    pres.stop()
    fake.unblock.set()
    pres.worker.join(5)
    clock.advance(500)
    v = pres.view()
    assert v["status"] == "stopped"
    assert v["session_elapsed"] == "00:00:12" and v["elapsed"] == "00:00:09"
    clock.advance(500)
    assert pres.view()["session_elapsed"] == "00:00:12"

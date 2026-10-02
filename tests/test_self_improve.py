"""Self Improve mode: learning starts right after preflight; benchmarks only when explicitly enabled."""

from unittest import mock

from app.gui.presenter import Presenter
from app.main import build_parser, main
from tests.helpers import FakeClock, FakeOllama, make_oracle
from tests.test_session_elapsed import SlowPreflightOllama, PREFLIGHT_COST


def _controller(make_controller):
    clock = FakeClock()
    statuses: list[tuple[str, float, float]] = []
    fake = SlowPreflightOllama(make_oracle(), on_chat=lambda n: clock.advance(10))
    fake.clock = clock
    c, _, _ = make_controller(fake, clock=clock)
    c.add_listener(
        lambda s: statuses.append((s["status"], s["elapsed"], s["session_elapsed"]))
    )
    return c, clock, statuses


def _order(statuses):
    out: list[str] = []
    for st, _, _ in statuses:
        if not out or out[-1] != st:
            out.append(st)
    return out


def test_self_improve_skips_both_benchmarks_by_default(make_controller):
    c, _, statuses = _controller(make_controller)
    report = c.self_improve(2)
    order = _order(statuses)
    assert "benchmark_before" not in order and "benchmark_after" not in order
    assert order[:2] == ["preflight", "running"]
    assert report["benchmark_before"] is None and report["benchmark_after"] is None
    assert report["benchmark_delta"] is None
    assert c.snapshot()["mode"] == "self_improve"
    assert report["verified"] >= 1


def test_learning_starts_immediately_after_preflight(make_controller):
    c, _, statuses = _controller(make_controller)
    c.self_improve(2)
    first_running = next(s for s in statuses if s[0] == "running")
    assert (
        first_running[1] == 0.0
    )  # learning elapsed starts at zero when the loop begins
    assert (
        first_running[2] == PREFLIGHT_COST
    )  # session clock already covered preflight only


def test_learning_elapsed_and_session_elapsed(make_controller):
    c, clock, _ = _controller(make_controller)
    t0 = clock()
    c.self_improve(2)
    s = c.snapshot()
    assert (
        s["elapsed"] == 120
    )  # the whole selected duration is learning time (no benchmark time)
    assert s["session_elapsed"] >= s["elapsed"] + PREFLIGHT_COST
    assert s["session_elapsed"] == clock() - t0


def test_benchmarks_run_only_when_explicitly_enabled(make_controller):
    c, _, statuses = _controller(make_controller)
    c.self_improve(2, run_benchmarks=True, quick_benchmark=True)
    order = _order(statuses)
    assert (
        order.index("benchmark_before")
        < order.index("running")
        < order.index("benchmark_after")
    )


def test_benchmarked_mode_unchanged(make_controller):
    c, _, statuses = _controller(make_controller)
    report = c.run_session(2, True, True)
    order = _order(statuses)
    assert (
        order.index("benchmark_before")
        < order.index("running")
        < order.index("benchmark_after")
    )
    assert c.snapshot()["mode"] == "benchmarked"
    assert (
        report["benchmark_before"] is not None and report["benchmark_after"] is not None
    )


def test_cli_self_improve_parses():
    a = build_parser().parse_args(["self-improve", "--minutes", "7"])
    assert (a.cmd, a.minutes, a.with_benchmarks) == ("self-improve", 7.0, False)
    a = build_parser().parse_args(
        ["self-improve", "--with-benchmarks", "--quick-benchmark"]
    )
    assert a.with_benchmarks and a.quick_benchmark and a.minutes == 30


def test_cli_self_improve_dispatches(make_controller, capsys):
    clock = FakeClock()
    c, _, _ = make_controller(
        FakeOllama(make_oracle(), on_chat=lambda n: clock.advance(10)), clock=clock
    )
    with (
        mock.patch.object(c, "self_improve", wraps=c.self_improve) as si,
        mock.patch.object(c, "run_session", wraps=c.run_session) as rs,
    ):
        assert main(["self-improve", "--minutes", "0.05"], c) == 0
    si.assert_called_once_with(0.05, False, False)
    rs.assert_called_once()
    assert (
        rs.call_args.kwargs["mode"] == "self_improve" and rs.call_args.args[1] is False
    )
    assert "Session" in capsys.readouterr().out


def test_cli_run_still_dispatches_benchmarked(make_controller):
    c, _, _ = make_controller(FakeOllama(make_oracle(), on_chat=lambda n: None))
    with mock.patch.object(
        c,
        "run_session",
        return_value={
            "session_id": 1,
            "status": "completed",
            "attempted": 0,
            "verified": 0,
            "failed": 0,
            "retries": 0,
            "lessons_learned": 0,
            "benchmark_delta": None,
            "improvement_note": "n/a",
        },
    ) as rs:
        assert main(["run", "--minutes", "3", "--skip-benchmark"], c) == 0
    rs.assert_called_once_with(3.0, False, False)


def test_gui_self_improve_action_dispatches(make_controller):
    c, _, _ = _controller(make_controller)
    pres = Presenter(c)
    assert pres.self_improve(2) is None
    pres.worker.join(10)
    v = pres.view()
    assert v["status"] == "completed" and v["mode"] == "self_improve"
    assert v["bench_before"] == "-" and v["bench_after"] == "-"
    assert v["attempted"] >= 1 and v["verified"] >= 1 and v["lessons"] >= 0
    assert v["elapsed"] == "00:02:00"


def test_gui_self_improve_rejects_bad_duration(make_controller):
    c, _, _ = _controller(make_controller)
    assert Presenter(c).self_improve(7) is not None

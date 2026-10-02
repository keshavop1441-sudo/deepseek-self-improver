import json
import time
from pathlib import Path

import pytest

from app.controller import Controller, PreflightError, SessionBusy
from app.gui.presenter import Presenter, fmt_elapsed
from app.ollama.client import OllamaClient
from app.tasks.model import Task
from tests.helpers import FakeClock, FakeOllama, StaticAdapter, make_oracle, model_json


def reports(cfg):
    return sorted(cfg.reports_dir.glob("session_*.json")), sorted(cfg.reports_dir.glob("session_*.md"))


def test_end_to_end_full_pipeline_with_fake_ollama(make_controller, cfg):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(bench_needs_lessons=True), on_chat=lambda n: clock.advance(5))
    c, _, _ = make_controller(fake, clock=clock)
    rep = c.run_session(2, run_benchmarks=True, quick_benchmark=True)

    assert rep["status"] == "completed" and rep["stop_reason"] == "deadline"
    assert rep["attempted"] >= 2 and rep["verified"] == rep["attempted"] and rep["failed"] == 0
    assert rep["retries"] >= 1                                  # gcd tasks are wrong on first attempt
    assert "arithmetic" in rep["failure_categories"] or "formula" in rep["failure_categories"]
    assert rep["lessons_learned"] == rep["verified"] and rep["lessons"]
    assert rep["domains"]["numerical_reasoning"]["verified"] == rep["verified"]
    # objective benchmark: after (with verified lessons in context) beats before
    assert rep["benchmark_after"]["overall"] > rep["benchmark_before"]["overall"]
    assert rep["benchmark_delta"] > 0 and rep["improved"] is True
    assert rep["sources"] == [] or isinstance(rep["sources"], list)
    jp, mp = reports(cfg)
    assert len(jp) == 1 and len(mp) == 1
    data = json.loads(jp[0].read_text())
    for key in ("duration_seconds", "attempted", "verified", "failed", "retries", "domains", "failure_categories",
                "lessons", "benchmark_before", "benchmark_after", "benchmark_delta", "sources", "errors"):
        assert key in data
    md = mp[0].read_text()
    assert "Benchmark" in md and "Lessons" in md and "Delta" in md
    # DB persisted everything
    st = c.stats()
    assert st["tasks_verified"] == rep["verified"] and st["lessons"] == rep["lessons_learned"] and st["benchmarks"] == 2
    assert st["verifications"] >= st["attempts"] - 0 and st["sessions"] == 1


def test_no_improvement_claimed_when_score_does_not_increase(make_controller):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(), on_chat=lambda n: clock.advance(5))   # benchmark answers identical before/after
    c, _, _ = make_controller(fake, clock=clock)
    rep = c.run_session(1, quick_benchmark=True)
    assert rep["benchmark_delta"] == 0 and rep["improved"] is False and "no objective improvement" in rep["improvement_note"]


def test_regression_is_reported_as_not_improved(make_controller):
    clock = FakeClock()
    state = {"phase": 0}
    base = make_oracle(bench_always_correct=True)
    bad = make_oracle(always_wrong=True)
    fake = FakeOllama(lambda m, p: (bad if state["phase"] else base)(m, p), on_chat=lambda n: clock.advance(5))
    c, _, _ = make_controller(fake, clock=clock)
    c.add_listener(lambda s: state.__setitem__("phase", 1 if s["status"] == "running" else state["phase"]))
    rep = c.run_session(1, quick_benchmark=True)
    assert rep["benchmark_delta"] < 0 and rep["improved"] is False


def test_time_limit_enforced(make_controller, cfg):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(25))
    c, _, _ = make_controller(fake, clock=clock)
    start = clock.t
    rep = c.run_session(1, run_benchmarks=False)           # 60 s budget, 25 s per model call
    assert rep["stop_reason"] == "deadline" and rep["status"] == "completed"
    assert len(fake.chat_calls) <= 3                         # no model call is started after the deadline
    assert clock.t - start < 60 + 25 + 1
    assert rep["benchmark_after"] is None and "benchmarks disabled" in rep["improvement_note"]


def test_stop_button_behavior(make_controller, cfg):
    clock = FakeClock()
    holder = {}
    fake = FakeOllama(make_oracle(wrong_first_gcd=False),
                      on_chat=lambda n: (clock.advance(5), holder["c"].stop() if n == 3 else None))
    c, _, _ = make_controller(fake, clock=clock)
    holder["c"] = c
    rep = c.run_session(30, run_benchmarks=True, quick_benchmark=True)
    assert rep["status"] == "stopped" and rep["stop_reason"] == "stopped"
    assert rep["benchmark_after"] is None and rep["improved"] is None and "no improvement claimed" in rep["improvement_note"]
    assert reports(cfg)[0]                                      # report still written
    assert c.snapshot()["status"] == "stopped" and not c.is_running


def test_stop_during_model_call_is_prompt(make_controller):
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), delay=3.0)
    c, _, _ = make_controller(fake)
    t = c.start(30, run_benchmarks=False)
    time.sleep(0.4)
    t0 = time.time()
    c.stop()
    t.join(5)
    assert not t.is_alive() and time.time() - t0 < 2.0
    assert c.snapshot()["status"] == "stopped"


@pytest.mark.parametrize("fake,match", [(FakeOllama(up=False), "not reachable"),
                                        (FakeOllama(models=["llama3"]), "ollama pull deepseek-r1:1.5b")])
def test_preflight_failures(make_controller, fake, match):
    c, _, _ = make_controller(fake)
    with pytest.raises(PreflightError, match=match):
        c.run_session(1)
    snap = c.snapshot()
    assert snap["status"] == "error" and match in snap["message"]
    assert c.stats()["sessions"] == 0


def test_ollama_dies_mid_session(make_controller, cfg):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(5))
    c, _, _ = make_controller(fake, clock=clock)
    orig = fake.__call__

    def dying(method, url, payload, timeout):
        if url.endswith("/api/chat") and len(fake.chat_calls) >= 2:
            fake.up = False
        return orig(method, url, payload, timeout)
    c.client.transport = dying
    rep = c.run_session(5, run_benchmarks=False)
    assert rep["status"] == "failed" and any("Ollama failure" in e for e in rep["errors"])
    assert reports(cfg)[0]


def test_source_failure_fallback_and_total_failure(make_controller):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(5))
    c, _, _ = make_controller(fake, clock=clock, adapters=[StaticAdapter(fail=True), StaticAdapter()])
    rep = c.run_session(1, run_benchmarks=False)
    assert rep["verified"] >= 1 and rep["status"] == "completed"
    c2, _, _ = make_controller(FakeOllama(), clock=FakeClock(), adapters=[StaticAdapter(fail=True)])
    rep2 = c2.run_session(1, run_benchmarks=False)
    assert rep2["verified"] == 0 and rep2["status"] == "failed"
    assert any("No new task" in e for e in rep2["errors"]) and any("giving up" in e for e in rep2["errors"])


def test_recovery_of_interrupted_session_and_resume(make_controller, cfg):
    c, _, _ = make_controller()
    s = c.storage
    sid = s.create_session(20, "deepseek-r1:1.5b")
    t = Task("numerical_reasoning", 1, "What is the gcd of 8 and 12?", "https://example.org", "t", "numeric_computation",
             {"kind": "gcd", "a": 8, "b": 12})
    s.insert_task(t, sid)
    s.set_task_status(t.task_id, "in_progress", sid)
    c2, _, clock = make_controller(FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: None), clock=FakeClock())
    assert c2.recover() == [sid]
    assert s.get_session(sid)["status"] == "interrupted"
    assert s.get_task_row(t.task_id)["status"] == "pending"
    assert reports(cfg)[0] and "interrupted" in json.loads(reports(cfg)[0][0].read_text())["status"]
    # next session resumes the pending task before discovering new ones
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: c2.clock.advance(20))
    c2.client.transport = fake
    rep = c2.run_session(1, run_benchmarks=False)
    assert s.get_task_row(t.task_id)["status"] == "verified"
    assert rep["verified"] >= 1


def test_benchmark_command_path(make_controller):
    c, _, _ = make_controller(FakeOllama(make_oracle(bench_always_correct=True)))
    res = c.run_benchmark(quick=True)
    assert res.overall == 1.0 and res.n_tasks == 12 and c.stats()["tasks"] == 0


def test_busy_guard(make_controller):
    c, _, _ = make_controller(FakeOllama(make_oracle(wrong_first_gcd=False), delay=0.5))
    t = c.start(30, run_benchmarks=False)
    time.sleep(0.2)
    with pytest.raises(SessionBusy):
        c.start(30)
    c.stop()
    t.join(5)


# ------------------------------------------------------------------ GUI <-> controller
def test_gui_presenter_drives_shared_controller(make_controller):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    p = Presenter(c)
    assert p.start(7) is not None                        # only 2/20/30/45 are valid in the GUI
    v0 = p.view()
    assert v0["model"] == "deepseek-r1:1.5b" and v0["start_enabled"] and not v0["stop_enabled"]
    assert p.start(20, quick_benchmark=True, run_benchmarks=False) is None
    p.worker.join(10)
    v = p.view()
    assert v["status"] == "completed" and v["verified"] >= 1 and v["lessons"] >= 1 and v["report"].endswith(".md")
    assert v["start_enabled"] and not v["stop_enabled"] and v["failed"] == 0
    assert fmt_elapsed(3725) == "01:02:05"


def test_gui_start_error_reports_preflight(make_controller):
    c, _, _ = make_controller(FakeOllama(up=False))
    p = Presenter(c)
    assert p.start(20) is None
    p.worker.join(5)
    v = p.view()
    assert v["status"] == "error" and "not reachable" in v["message"]


def test_doctor_reports_missing_ollama(make_controller):
    c, _, _ = make_controller(FakeOllama(up=False))
    checks = {x["name"]: x for x in c.doctor()}
    assert not checks["ollama_server"]["ok"] and checks["database"]["ok"] and checks["sympy"]["ok"]
    c2, _, _ = make_controller(FakeOllama(models=[]))
    assert not {x["name"]: x for x in c2.doctor()}["model"]["ok"]

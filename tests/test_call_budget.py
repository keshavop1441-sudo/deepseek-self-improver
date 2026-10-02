"""Adaptive Self Improve per-call budget: wall share of remaining time + generation-token cap. Benchmarks unchanged."""

import math

import pytest

from app.benchmark.runner import BenchmarkRunner
from app.benchmark.tasks import select_subset
from app.config import Config
from app.memory.lessons import LessonMemory
from app.ollama.client import OllamaClient
from app.solver.solver import CallBudget, Solver, measured_tokens_per_second
from app.storage.db import Storage
from app.tasks.model import Task
from app.verifier import Verifier
from tests.helpers import FakeOllama, model_json


def task():
    return Task(
        "numerical_reasoning",
        3,
        "What is the gcd of 12 and 18?",
        "https://example.org/s",
        "src",
        "numeric_computation",
        {"kind": "gcd", "a": 12, "b": 18},
    )


class MeteredOllama(FakeOllama):
    """FakeOllama whose chat replies carry Ollama eval metadata; `meta` is consumed one entry per call."""

    def __init__(self, answers, meta=None, on_call=None):
        super().__init__(
            lambda m, p: model_json(
                answers[min(len(self.chat_calls), len(answers)) - 1]
            )
        )
        self.meta = list(meta or [])
        self.on_call = on_call

    def __call__(self, method, url, payload, timeout):
        status, body = super().__call__(method, url, payload, timeout)
        if url.endswith("/api/chat") and status == 200:
            self.timeouts = getattr(self, "timeouts", []) + [timeout]
            if self.meta:
                body.update(self.meta.pop(0))
            if self.on_call:
                self.on_call(len(self.chat_calls))
        return status, body

    @property
    def predicts(self):
        return [c["options"]["num_predict"] for c in self.chat_calls]


def make(tmp_path, fake, budget=None, request_timeout=300.0, num_predict=3072):
    s = Storage(tmp_path / "s.db")
    client = OllamaClient(
        "http://x",
        "deepseek-r1:1.5b",
        transport=fake,
        request_timeout=request_timeout,
        num_predict=num_predict,
    )
    solver = Solver(
        client,
        Verifier(),
        s,
        LessonMemory(s),
        request_timeout=request_timeout,
        budget=budget,
    )
    t = task()
    s.insert_task(t)
    return s, solver, t


NEVER = lambda: False


def eval_meta(tokens, seconds):
    return {"eval_count": tokens, "eval_duration": int(seconds * 1e9)}


# ---------------------------------------------------------------- A. client
def test_client_num_predict_default_and_override():
    fake = FakeOllama()
    c = OllamaClient("http://x", "deepseek-r1:1.5b", transport=fake, num_predict=3072)
    c.chat([{"role": "user", "content": "x"}])
    c.chat([{"role": "user", "content": "x"}], num_predict=500)
    assert [p["options"]["num_predict"] for p in fake.chat_calls] == [3072, 500]
    with pytest.raises(ValueError):
        c.chat([{"role": "user", "content": "x"}], num_predict=0)


# ---------------------------------------------------------------- B. wall budget
@pytest.mark.parametrize(
    "remaining,timeout,expected",
    [
        (120.0, 300.0, 96.0),  # 2-minute session: 80% share
        (30.0, 300.0, 24.0),  # share above the floor
        (1000.0, 300.0, 300.0),  # request_timeout cap
        (25.0, 300.0, 20.0),  # floor (20 > 25*0.8), still below remaining
        (15.0, 300.0, 15.0),  # below min_call_seconds: use what remains
        (0.0, 300.0, 0.0),
        (-5.0, 300.0, 0.0),
        (120.0, 40.0, 40.0),
        (100.0, 300.0, 80.0),
    ],
)
def test_wall_budget(tmp_path, remaining, timeout, expected):
    _, solver, _ = make(tmp_path, FakeOllama(), request_timeout=timeout)
    got = solver.call_wall_budget(remaining)
    assert got == expected and got <= max(0.0, remaining) and got <= timeout


def test_solve_passes_budget_as_request_timeout(tmp_path):
    fake = MeteredOllama(["6"])
    _, solver, t = make(tmp_path, fake)
    assert solver.solve(t, 1, NEVER, lambda: 120.0).status == "verified"
    assert fake.timeouts == [96.0]


# ---------------------------------------------------------------- C. token cap
def test_token_cap_formula_and_clamps(tmp_path):
    _, solver, _ = make(tmp_path, FakeOllama())
    assert (
        solver.call_token_cap(96.0, 1) == math.floor(96 * 23.0) == 2208
    )  # first call: assumed rate (2-minute session)
    assert solver.call_token_cap(1000.0, 1) == 3072  # clamped to configured num_predict
    assert solver.call_token_cap(1.0, 1) == 256  # clamped up to min_tokens
    assert solver.call_token_cap(0.0, 1) == 256 and solver.call_token_cap(-3, 1) == 256
    solver._rate = (1, 23.5)
    assert (
        solver.call_token_cap(60.0, 1) == math.floor(60 * 23.5) == 1410
    )  # measured rate
    assert solver.call_token_cap(60.0, 2) == 1380  # another session: back to assumed


def test_token_cap_with_num_predict_below_min_tokens(tmp_path):
    _, solver, _ = make(tmp_path, FakeOllama(), num_predict=100)
    assert solver.call_token_cap(60.0) == 100


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"eval_count": 230, "eval_duration": 10_000_000_000}, 23.0),
        ({}, None),
        (None, None),
        ("x", None),
        ({"eval_count": 0, "eval_duration": 1e9}, None),
        ({"eval_count": 10, "eval_duration": 0}, None),
        ({"eval_count": -5, "eval_duration": 1e9}, None),
        ({"eval_count": "10", "eval_duration": 1e9}, None),
        ({"eval_count": True, "eval_duration": 1e9}, None),
        ({"eval_count": float("nan"), "eval_duration": 1e9}, None),
        ({"eval_count": 10, "eval_duration": float("inf")}, None),
        ({"eval_count": 10**9, "eval_duration": 1}, None),  # implausibly fast
        ({"eval_count": 1, "eval_duration": 10**12}, None),  # implausibly slow
    ],
)
def test_measured_rate_guards(raw, expected):
    assert measured_tokens_per_second(raw) == expected


def test_rate_measured_from_returned_call_only_and_invalid_falls_back(tmp_path):
    # remaining is constant 40 s -> wall 32 s. Call 1: no measurement yet (assumed 23 tok/s).
    # Call 2 follows a usable 15 tok/s reading; call 3 follows invalid metadata, so it keeps 15; call 4 follows 10.
    fake = MeteredOllama(
        ["7", "7", "7", "6"],
        meta=[eval_meta(1500, 100), {"eval_count": 0}, eval_meta(100, 10), {}],
    )
    _, solver, t = make(tmp_path, fake)
    out = solver.solve(t, 1, NEVER, lambda: 40.0)
    assert out.status == "verified"
    assert fake.predicts == [
        math.floor(32 * 23.0),
        math.floor(32 * 15.0),
        math.floor(32 * 15.0),
        math.floor(32 * 10.0),
    ]
    assert solver._rate == (1, 10.0)


def test_aborted_call_does_not_set_rate(tmp_path):
    from tests.test_deadline import HangingOllama

    fake = HangingOllama(ok_calls=0)
    _, solver, t = make(tmp_path, fake)
    out = solver.solve(
        t, 1, lambda: len(fake.chat_calls) > 0 and fake.hanging.is_set(), lambda: 120.0
    )
    fake.unblock.set()
    assert out.status == "aborted" and solver._rate is None


# ---------------------------------------------------------------- D. retry adaptation
def test_retry_budget_and_tokens_shrink_with_remaining_time(tmp_path):
    clock = {"t": 0.0}
    fake = MeteredOllama(
        ["7", "7", "7", "6"],
        meta=[eval_meta(2300, 100)] * 4,
        on_call=lambda n: clock.__setitem__("t", clock["t"] + 30),
    )
    _, solver, t = make(tmp_path, fake)
    out = solver.solve(t, 1, NEVER, lambda: 120.0 - clock["t"])
    assert out.status == "verified" and out.retries == 3
    # remaining 120, 90, 60, 30 -> wall 96, 72, 48, 24 (80%); rate 23 tok/s assumed, then measured 23
    assert fake.timeouts == [96.0, 72.0, 48.0, 24.0]
    assert fake.predicts == [2208, 1656, 1104, 552]
    assert fake.predicts == sorted(fake.predicts, reverse=True)


def test_no_call_started_when_nothing_remains(tmp_path):
    fake = MeteredOllama(["6"])
    _, solver, t = make(tmp_path, fake)
    out = solver.solve(t, 1, NEVER, lambda: 0.0)
    assert out.status == "aborted" and out.started == 0 and not fake.chat_calls


def test_budget_never_exceeds_remaining_for_tiny_windows(tmp_path):
    fake = MeteredOllama(["6"])
    _, solver, t = make(tmp_path, fake)
    solver.solve(t, 1, NEVER, lambda: 3.0)
    assert fake.timeouts == [3.0] and fake.predicts == [256]


# ---------------------------------------------------------------- E. benchmark path unchanged
def test_benchmark_ask_path_uses_configured_cap_and_no_adaptive_logic(tmp_path):
    fake = MeteredOllama(["6"], meta=[eval_meta(100, 1)])
    _, solver, t = make(tmp_path, fake, num_predict=3072)
    solver.call_wall_budget = solver.call_token_cap = (
        None  # would blow up if the adaptive path were touched
    )
    solver.ask(
        t, [{"role": "user", "content": "x"}], NEVER, None, temperature=0.0, seed=1
    )
    assert fake.predicts == [3072]
    assert fake.timeouts == [300.0]
    assert solver._rate is None


def test_benchmark_runner_payload_unchanged(tmp_path):
    fake = MeteredOllama(["0"])
    s, solver, _ = make(tmp_path, fake, num_predict=3072)
    runner = BenchmarkRunner(solver, LessonMemory(s), s, "deepseek-r1:1.5b")
    runner.run(select_subset(2), "before", None, NEVER, None, allow_reuse=False)
    assert fake.chat_calls and set(fake.predicts) == {3072}


def test_config_defaults():
    c = Config()
    assert (
        c.call_share,
        c.min_call_seconds,
        c.min_tokens,
        c.assumed_tokens_per_second,
    ) == (0.8, 20.0, 256, 23.0)
    assert (c.request_timeout, c.num_predict) == (300.0, 3072)
    assert CallBudget() == CallBudget(
        c.call_share, c.min_call_seconds, c.min_tokens, c.assumed_tokens_per_second
    )


# ---------------------------------------------------------------- controller plumbing
def test_controller_passes_config_budget_and_session_calls_are_capped(
    make_controller, cfg
):
    from tests.helpers import FakeClock, make_oracle

    clock = FakeClock()
    fake = FakeOllama(
        make_oracle(wrong_first_gcd=False), on_chat=lambda n: clock.advance(10)
    )
    c, _, _ = make_controller(fake, clock=clock)
    assert c.solver.budget == CallBudget(
        cfg.call_share,
        cfg.min_call_seconds,
        cfg.min_tokens,
        cfg.assumed_tokens_per_second,
    )
    c.solver.request_timeout = (
        300.0  # the test Config uses 5 s, which would dominate the share
    )
    rep = c.run_session(2, run_benchmarks=False, mode="self_improve")
    preds = [p["options"]["num_predict"] for p in fake.chat_calls]
    assert rep["stop_reason"] == "deadline" and preds
    assert preds[0] == math.floor(
        96 * 23.0
    )  # 120 s window -> 96 s (80%) at the assumed 23 tok/s = 2208
    assert all(1 <= p <= 3072 for p in preds) and preds == sorted(preds, reverse=True)
    assert (
        rep["attempted"]
        == rep["verified"] + rep["failed"] + rep["invalid_tasks"] + rep["pending_tasks"]
    )

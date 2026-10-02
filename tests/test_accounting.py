"""Session accounting for the Self Improve loop: attempted / failed / retries / pending vs failure_categories.

Regression for a real run (probability task, verifier recorded `arithmetic: 1`, deadline hit during the retry)
that reported attempted=0, failed=0, retries=0.

Invariant: attempted == verified + failed + invalid + pending. A task counts as attempted once at least one model
attempt completed and was recorded. A task whose first model call was still in flight when the deadline hit has no
verifier result, so it is not counted and adds nothing to failure_categories. Invalid (unusable) tasks are the one
case where failure_categories gets a `missing_data` entry for a task that is *not* a model failure; they are still
counted as attempted + invalid, so failure_categories never shows a failure for a task the report calls unattempted.
"""

import threading

import pytest

from app.discovery.adapters import Adapter
from app.solver.solver import SolveOutcome
from app.tasks.model import Task
from tests.helpers import FakeClock, FakeOllama, model_json

MINUTES = 2
LIMIT = MINUTES * 60
WRONG = "0.1640"  # within 10% of the true 0.15625 for n=5, so the verifier says `arithmetic`


class ProbabilityAdapter(Adapter):
    name = "prob_test"
    domains = ("probability",)

    def __init__(self):
        from tests.helpers import FakeHttp

        super().__init__(FakeHttp())
        self.n = 0

    def discover(self, rng):
        self.n += 1
        return [
            Task(
                "probability",
                3,
                f"X ~ Binomial(n={4 + self.n}, p=1/2). Compute P(X = 1) to 4 decimal places.",
                "https://example.org/p",
                "Prob test",
                "probability_exact",
                {
                    "kind": "binomial_pmf",
                    "n": 4 + self.n,
                    "k": 1,
                    "p": "1/2",
                    "decimals": 4,
                },
            )
        ]


class WrongThenHang(FakeOllama):
    """Chat call 1 -> wrong answer (verifier: arithmetic). Call 2 -> blocks like a long generation (the retry)."""

    def __init__(self):
        super().__init__(lambda m, p: model_json(WRONG))
        self.hanging = threading.Event()
        self.unblock = threading.Event()

    def __call__(self, method, url, payload, timeout):
        if url.endswith("/api/chat") and len(self.chat_calls) >= 1:
            self.chat_calls.append(payload)
            self.hanging.set()
            self.unblock.wait(30)
            return 500, {"error": "released after abort"}
        return super().__call__(method, url, payload, timeout)


def run_until_deadline(make_controller, stop=False):
    clock = FakeClock()
    fake = WrongThenHang()
    c, _, _ = make_controller(fake, clock=clock, adapters=[ProbabilityAdapter()])
    box = {}
    t = threading.Thread(
        target=lambda: box.update(rep=c.run_session(MINUTES, run_benchmarks=False)),
        daemon=True,
    )
    t.start()
    assert fake.hanging.wait(5)
    if stop:
        c.stop()
    else:
        clock.advance(LIMIT)  # deadline arrives while the retry is in flight
    t.join(5)
    fake.unblock.set()
    assert not t.is_alive()
    return c, box["rep"]


def assert_consistent(rep):
    assert (
        rep["attempted"]
        == rep["verified"] + rep["failed"] + rep["invalid_tasks"] + rep["pending_tasks"]
    )
    if sum(rep["failure_categories"].values()):
        assert rep["attempted"] >= 1, (
            "failure category recorded for a task the report says was never attempted"
        )


@pytest.mark.parametrize("stop", [False, True])
def test_real_run_pattern_deadline_during_retry_is_pending_not_failed(
    make_controller, stop
):
    c, rep = run_until_deadline(make_controller, stop)
    # attempt 0 was verified wrong (arithmetic); retry 1 was handed to the model and cut off
    assert rep["failure_categories"] == {"arithmetic": 1}
    assert rep["attempted"] == 1 and rep["pending_tasks"] == 1
    assert rep["failed"] == 0 and rep["verified"] == 0 and rep["invalid_tasks"] == 0
    assert rep["retries"] == 1
    assert rep["lessons_learned"] == 0 and rep["domains"] == {
        "probability": {"verified": 0, "failed": 0}
    }
    assert_consistent(rep)
    assert [r[0] for r in c.storage._conn.execute("SELECT status FROM tasks")] == [
        "pending"
    ]
    snap = c.snapshot()
    for k in ("attempted", "verified", "failed", "retries", "pending", "invalid"):
        assert snap[k] == rep[k if k not in ("pending", "invalid") else k + "_tasks"], k
    assert snap["failure_categories"] == rep["failure_categories"]
    assert c.storage.get_session(rep["session_id"])["stop_reason"] == (
        "stopped" if stop else "deadline"
    )


def test_first_call_cut_off_is_not_attempted_and_has_no_failure_category(
    make_controller,
):
    clock = FakeClock()
    fake = WrongThenHang()
    fake.chat_calls.append({})  # every real chat call hangs (first one included)
    c, _, _ = make_controller(fake, clock=clock, adapters=[ProbabilityAdapter()])
    box = {}
    t = threading.Thread(
        target=lambda: box.update(rep=c.run_session(MINUTES, run_benchmarks=False)),
        daemon=True,
    )
    t.start()
    assert fake.hanging.wait(5)
    clock.advance(LIMIT)
    t.join(5)
    fake.unblock.set()
    rep = box["rep"]
    assert (
        rep["attempted"] == rep["pending_tasks"] == rep["retries"] == rep["failed"] == 0
    )
    assert rep["failure_categories"] == {}
    assert_consistent(rep)


def solve_scripted(make_controller, answers):
    """Run the real Solver on one probability task, feed the outcome to the real Controller accounting."""
    calls = []

    def respond(m, p):
        calls.append(1)
        return model_json(answers[min(len(calls), len(answers)) - 1])

    fake = FakeOllama(respond)
    c, _, _ = make_controller(fake, adapters=[ProbabilityAdapter()])
    task = ProbabilityAdapter().discover(None)[0]
    c.storage.insert_task(task)
    out = c.solver.solve(task, None, lambda: False, lambda: 1000.0)
    c._record_outcome(out, [])
    return c, out


def test_verifier_failure_then_retry_success_counts_retry_and_verified(make_controller):
    c, out = solve_scripted(
        make_controller, [WRONG, "0.1563"]
    )  # n=5: C(5,1)/32 = 0.15625
    s = c.snapshot()
    assert out.status == "verified" and s["verified"] == 1 and s["attempted"] == 1
    assert s["failed"] == 0 and s["retries"] == 1
    assert s["failure_categories"] == {
        "arithmetic": 1
    }  # corrected failure stays recorded, consistent with attempted
    assert s["domains"]["probability"] == {"verified": 1, "failed": 0}


def test_exhausted_retries_counts_failed_and_all_retries(make_controller):
    c, out = solve_scripted(make_controller, [WRONG])
    s = c.snapshot()
    assert (
        out.status == "failed"
        and s["failed"] == 1
        and s["attempted"] == 1
        and s["verified"] == 0
    )
    assert s["retries"] == 3 and s["failure_categories"] == {"arithmetic": 4}
    assert s["domains"]["probability"] == {"verified": 0, "failed": 1}


def test_invalid_task_is_not_a_model_failure(make_controller):
    fake = FakeOllama(lambda m, p: model_json("1"))
    c, _, _ = make_controller(fake, adapters=[ProbabilityAdapter()])
    bad = Task(
        "probability", 3, "bad", "u", "t", "probability_exact", {"kind": "no_such_kind"}
    )
    c.storage.insert_task(bad)
    out = c.solver.solve(bad, None, lambda: False, lambda: 1000.0)
    c._record_outcome(out, [])
    s = c.snapshot()
    assert (
        out.status == "invalid"
        and s["invalid"] == 1
        and s["failed"] == 0
        and s["retries"] == 0
    )
    assert s["attempted"] == 1 and s["domains"]["probability"] == {
        "verified": 0,
        "failed": 0,
    }
    assert s["failure_categories"] == {
        "missing_data": 1
    }  # data-validation entry, attempted counted: consistent


def test_aborted_outcome_without_attempts_changes_nothing(make_controller):
    c, _, _ = make_controller(FakeOllama(), adapters=[ProbabilityAdapter()])
    task = ProbabilityAdapter().discover(None)[0]
    c.storage.insert_task(task)
    c._record_outcome(SolveOutcome(task=task, status="aborted"), [])
    s = c.snapshot()
    assert (s["attempted"], s["pending"], s["retries"], s["failed"]) == (0, 0, 0, 0)
    assert s["failure_categories"] == {}

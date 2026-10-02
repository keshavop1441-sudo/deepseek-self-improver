import pytest

from app.memory.lessons import LessonMemory, UnverifiedLessonError, learn_from_outcome
from app.ollama.client import ModelNotAvailable, OllamaClient, OllamaUnavailable
from app.solver.solver import Solver
from app.storage.db import Storage
from app.tasks.model import Task
from app.verifier import Verifier
from app.verifier.base import incorrect, verified
from tests.helpers import FakeOllama, model_json, user_text


def task(a=12, b=18, q=None):
    return Task("numerical_reasoning", 1, q or f"What is the gcd of {a} and {b}?", "https://example.org/s", "src",
                "numeric_computation", {"kind": "gcd", "a": a, "b": b})


def setup(tmp_path, script, max_retries=3, **fake_kw):
    s = Storage(tmp_path / "s.db")
    calls = []

    def responder(messages, payload):
        calls.append(messages)
        item = script[min(len(calls) - 1, len(script) - 1)]
        return item if isinstance(item, str) else item(messages)
    fake = FakeOllama(responder, **fake_kw)
    client = OllamaClient("http://x", "deepseek-r1:1.5b", transport=fake, request_timeout=5)
    mem = LessonMemory(s)
    solver = Solver(client, Verifier(), s, mem, max_retries=max_retries)
    t = task()
    s.insert_task(t)
    return s, solver, mem, t, calls


NEVER = lambda: False
LEFT = lambda: 1000.0


def test_correct_first_try(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("6")])
    out = solver.solve(t, 1, NEVER, LEFT)
    assert out.status == "verified" and out.retries == 0 and len(calls) == 1


def test_wrong_then_corrected_with_feedback_and_preserved_attempts(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("7"), model_json("6", "fixed method")])
    out = solver.solve(t, 1, NEVER, LEFT)
    assert out.status == "verified" and out.retries == 1
    rows = s.attempts_for_task(t.task_id)
    assert [r["attempt_no"] for r in rows] == [0, 1] and rows[0]["answer"] == "7" and rows[1]["answer"] == "6"
    vers = s.verifications_for_task(t.task_id)
    assert [v["verified"] for v in vers] == [0, 1]
    retry_prompt = calls[1][-1]["content"]
    assert "INCORRECT" in retry_prompt and "specific error" in retry_prompt and "7" in retry_prompt
    assert "6" not in retry_prompt.replace("62", "")    # expected value never leaked (answer was 7)
    assert vers[0]["failure_category"] in {"arithmetic", "formula"}


def test_max_three_retries_then_failed(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("7")])
    out = solver.solve(t, 1, NEVER, LEFT)
    assert out.status == "failed" and out.retries == 3 and len(calls) == 4
    assert len(s.attempts_for_task(t.task_id)) == 4
    assert learn_from_outcome(mem, out) is None and s.stats()["lessons"] == 0


def test_self_check_claims_do_not_make_it_verified(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("7", self_check="verified correct, confidence 1.0")],
                                     max_retries=0)
    assert solver.solve(t, 1, NEVER, LEFT).status == "failed"


def test_malformed_output_is_failure_then_recovers(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, ["I think maybe six??", "{{{ broken", model_json("6")])
    out = solver.solve(t, 1, NEVER, LEFT)
    assert out.status == "verified" and out.retries == 2
    assert s.attempts_for_task(t.task_id)[0]["parse_ok"] == 0


def test_timeout_attempt_recorded_then_retry(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("6")])
    solver.request_timeout = 0.2
    solver.client.transport = FakeOllama(lambda m, p: model_json("6"), delay=0.6)
    out = solver.solve(t, 1, NEVER, LEFT)
    assert out.status == "failed"
    assert all(r["error"] and "Timeout" in r["error"] for r in s.attempts_for_task(t.task_id))


def test_ollama_down_is_fatal(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("6")])
    solver.client.transport = FakeOllama(up=False)
    with pytest.raises(OllamaUnavailable):
        solver.solve(t, 1, NEVER, LEFT)


def test_model_missing_is_fatal(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("6")])
    solver.client.transport = FakeOllama(models=["other"])
    with pytest.raises(ModelNotAvailable):
        solver.solve(t, 1, NEVER, LEFT)


def test_abort_when_deadline_passed(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("6")])
    assert solver.solve(t, 1, NEVER, lambda: 0.0).status == "aborted" and not calls


def test_invalid_task_is_not_retried(tmp_path):
    s, solver, mem, _, calls = setup(tmp_path, [model_json("1")])
    bad = Task("quant_finance", 2, "bad", "u", "t", "finance_formula",
               {"formula": "net_margin_pct", "net_income": 1, "revenue": 0})
    s.insert_task(bad)
    out = solver.solve(bad, 1, NEVER, LEFT)
    assert out.status == "invalid" and len(calls) == 1


def test_coding_answer_taken_from_code_fence_when_json_missing(tmp_path):
    from tests.helpers import CODE
    s, solver, mem, _, _ = setup(tmp_path, ["```python\n" + CODE["fib"] + "```"])
    t = Task("coding", 2, "fib?", "u", "t", "code_tests", {"function_name": "fib", "tests": [[[10], 55]]})
    s.insert_task(t)
    assert solver.solve(t, 1, NEVER, LEFT).status == "verified"


# ------------------------------------------------------------------ verified memory
def test_lesson_only_from_verified(tmp_path):
    s, solver, mem, t, _ = setup(tmp_path, [model_json("7"), model_json("6", "use euclid")])
    out = solver.solve(t, 1, NEVER, LEFT)
    lid = learn_from_outcome(mem, out)
    row = s.list_lessons()[0]
    assert lid == row["lesson_id"] and row["domain"] == "numerical_reasoning"
    assert row["failure_pattern"].startswith(("arithmetic", "formula")) and "euclid" in row["correct_method"]
    assert row["source"] == "https://example.org/s" and row["created_at"]
    assert s.get_verification(row["verification_id"])["verified"] == 1


def test_add_lesson_refuses_unverified_verification(tmp_path):
    s, solver, mem, t, _ = setup(tmp_path, [model_json("7")], max_retries=0)
    out = solver.solve(t, 1, NEVER, LEFT)
    bad_vid = out.final.verification_id
    with pytest.raises(UnverifiedLessonError):
        mem.add_lesson(t, bad_vid, "7", "m", None)
    with pytest.raises(UnverifiedLessonError):
        mem.add_lesson(t, 99999, "7", "m", None)
    # a verified verification of ANOTHER task cannot be borrowed either
    t2 = task(20, 30)
    s.insert_task(t2)
    aid = s.insert_attempt(t2.task_id, 1, 0, "p", "r", None, {}, True, "10", None, 0.1)
    vid = s.insert_verification(aid, t2.task_id, verified("numeric_computation"))
    with pytest.raises(UnverifiedLessonError):
        mem.add_lesson(t, vid, "7", "m", None)
    assert s.stats()["lessons"] == 0


def test_uncertain_status_never_creates_lesson(tmp_path):
    s, solver, mem, t, _ = setup(tmp_path, [model_json("6 or 7")], max_retries=0)
    out = solver.solve(t, 1, NEVER, LEFT)
    assert out.status == "failed" and learn_from_outcome(mem, out) is None


def test_retrieval_is_small_relevant_subset(tmp_path):
    s = Storage(tmp_path / "m.db")
    mem = LessonMemory(s)
    for i in range(30):
        tk = task(10 + i, 20 + i)
        s.insert_task(tk)
        aid = s.insert_attempt(tk.task_id, 1, 0, "p", "r", None, {}, True, "1", None, 0.1)
        vid = s.insert_verification(aid, tk.task_id, verified("numeric_computation"))
        mem.add_lesson(tk, vid, "1", f"method {i}", "arithmetic: slip" if i % 2 else None)
    other = Task("logic", 1, "q", "u", "t", "logic_truth_table", {})
    s.insert_task(other)
    aid = s.insert_attempt(other.task_id, 1, 0, "p", "r", None, {}, True, "1", None, 0.1)
    mem.add_lesson(other, s.insert_verification(aid, other.task_id, verified("logic_truth_table")), "1", "logic method", None)
    got = mem.relevant(task(500, 600), k=3)
    assert len(got) == 3 and all(g["domain"] == "numerical_reasoning" for g in got)
    text = mem.format_for_prompt(got)
    assert text.count("\n- ") == 3 and len(text) < 1500


def test_lessons_appear_in_later_prompts(tmp_path):
    s, solver, mem, t, calls = setup(tmp_path, [model_json("6", "euclid method")])
    learn_from_outcome(mem, solver.solve(t, 1, NEVER, LEFT))
    t2 = task(30, 45)
    s.insert_task(t2)
    solver.solve(t2, 1, NEVER, LEFT)
    assert "Verified lessons" in user_text(calls[-1]) and "euclid method" in user_text(calls[-1])

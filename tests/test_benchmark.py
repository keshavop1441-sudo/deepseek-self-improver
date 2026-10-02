from app.benchmark.runner import BenchmarkAborted, BenchmarkRunner, score
from app.benchmark.tasks import BENCHMARK_TASKS, benchmark_hashes, benchmark_version, category_of, select_subset
from app.memory.lessons import LessonMemory
from app.ollama.client import OllamaClient
from app.solver.solver import Solver
from app.storage.db import Storage
from app.verifier import Verifier
from tests.helpers import FakeOllama, make_oracle, model_json
import pytest


def runner(tmp_path, responder):
    s = Storage(tmp_path / "b.db")
    client = OllamaClient("http://x", "deepseek-r1:1.5b", transport=FakeOllama(responder), request_timeout=5)
    mem = LessonMemory(s)
    return BenchmarkRunner(Solver(client, Verifier(), s, mem), mem, s, client.model), s


def test_benchmark_has_30_plus_tasks_all_categories_unique():
    assert len(BENCHMARK_TASKS) >= 30
    assert {category_of(t) for t in BENCHMARK_TASKS} == {"numerical", "mathematics", "statistics", "finance", "logic", "coding"}
    assert len({t.content_hash for t in BENCHMARK_TASKS}) == len(BENCHMARK_TASKS)
    assert all(t.source_url.startswith("benchmark://") for t in BENCHMARK_TASKS)


def test_perfect_scoring(tmp_path):
    r, s = runner(tmp_path, make_oracle(bench_always_correct=True))
    res = r.run(label="standalone")
    assert res.overall == 1.0 and all(v == 1.0 for v in res.scores.values()) and res.n_tasks == len(BENCHMARK_TASKS)
    assert set(res.scores) == {"numerical", "mathematics", "statistics", "finance", "logic", "coding"}


def test_zero_scoring_and_partial(tmp_path):
    r, s = runner(tmp_path, make_oracle(always_wrong=True))
    assert r.run().overall == 0.0
    res = [{"correct": True, "category": "logic"}, {"correct": False, "category": "logic"},
           {"correct": True, "category": "coding"}, {"correct": False, "category": "coding"}]
    o, sc = score(res)
    assert o == 0.5 and sc == {"logic": 0.5, "coding": 0.5}


def test_model_self_claims_do_not_score(tmp_path):
    r, s = runner(tmp_path, lambda m, p: model_json("totally wrong", self_check="correct with 100% confidence"))
    assert r.run().overall == 0.0


def test_isolation_from_training_tables(tmp_path):
    r, s = runner(tmp_path, make_oracle(bench_always_correct=True))
    r.run()
    st = s.stats()
    assert st["tasks"] == 0 and st["attempts"] == 0 and st["lessons"] == 0 and st["benchmarks"] == 1
    assert benchmark_hashes().isdisjoint({x["content_hash"] for x in s._all("SELECT content_hash FROM tasks")})


def test_benchmark_uses_deterministic_sampling(tmp_path):
    fake = FakeOllama(make_oracle(bench_always_correct=True))
    s = Storage(tmp_path / "b.db")
    client = OllamaClient("http://x", "deepseek-r1:1.5b", transport=fake, request_timeout=5)
    mem = LessonMemory(s)
    BenchmarkRunner(Solver(client, Verifier(), s, mem), mem, s, client.model).run(select_subset(3))
    assert all(c["options"]["temperature"] == 0.0 and c["options"]["seed"] == 1234 for c in fake.chat_calls)


def test_subset_is_stratified_and_deterministic():
    a, b = select_subset(12), select_subset(12)
    assert [t.task_id for t in a] == [t.task_id for t in b] and len(a) == 12
    assert len({category_of(t) for t in a}) == 6


def test_abort_discards_partial(tmp_path):
    r, s = runner(tmp_path, make_oracle(bench_always_correct=True))
    n = {"i": 0}

    def abort():
        n["i"] += 1
        return n["i"] > 3
    with pytest.raises(BenchmarkAborted):
        r.run(should_abort=abort)
    assert s.stats()["benchmarks"] == 0


def test_persisted_and_reuse_requires_same_lessons(tmp_path):
    r, s = runner(tmp_path, make_oracle(bench_always_correct=True))
    tasks = select_subset(6)
    first = r.run(tasks, "before", 1, allow_reuse=True)
    again = r.run(tasks, "before", 1, allow_reuse=True)
    assert again.reused_from == first.benchmark_id and again.overall == first.overall
    s._exec("INSERT INTO lessons(domain,failure_pattern,correct_method,example,source,created_at,task_id,verification_id,lesson_hash)"
            " VALUES ('d','p','m','e','s','now','t',1,'h')")
    third = r.run(tasks, "before", 1, allow_reuse=True)
    assert third.reused_from is None


def test_version_changes_with_content():
    assert benchmark_version(BENCHMARK_TASKS[:5]) != benchmark_version(BENCHMARK_TASKS)

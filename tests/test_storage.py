from app.storage.db import Storage
from app.tasks.model import Task, compute_content_hash
from app.verifier.base import verified


def mk(q="What is 2+2?", **kw):
    return Task("mathematics", 1, q, "https://example.org", "T", "numeric_computation", {"kind": "gcd", "a": 4, "b": 6}, **kw)


def test_all_tables_exist(tmp_path):
    s = Storage(tmp_path / "x.db")
    names = {r["name"] for r in s._all("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "tasks", "attempts", "verifications", "lessons", "benchmarks", "sources"} <= names


def test_duplicate_prevention_by_content_hash(tmp_path):
    s = Storage(tmp_path / "x.db")
    assert s.insert_task(mk())
    dup = mk(retrieved_at="2030-01-01T00:00:00+00:00")   # same content, different time
    assert dup.content_hash == mk().content_hash
    assert not s.insert_task(dup)
    assert s.stats()["tasks"] == 1


def test_hash_ignores_whitespace_only_changes():
    assert compute_content_hash("d", "a  b", {}) == compute_content_hash("d", "a b", {})


def test_persistence_across_reopen(tmp_path):
    p = tmp_path / "x.db"
    s = Storage(p)
    t = mk()
    s.insert_task(t)
    sid = s.create_session(20, "m")
    aid = s.insert_attempt(t.task_id, sid, 0, "p", "raw", None, {"answer": "2"}, True, "2", None, 1.0)
    vid = s.insert_verification(aid, t.task_id, verified("numeric_computation"))
    s.record_source("a", "https://example.org", "T", "ok", 200)
    s.close()
    s2 = Storage(p)
    assert s2.get_task_row(t.task_id)["question"] == t.question
    assert Task.from_row(s2.get_task_row(t.task_id)).content_hash == t.content_hash
    assert s2.attempts_for_task(t.task_id)[0]["answer"] == "2"
    assert s2.get_verification(vid)["verified"] == 1
    assert s2.get_session(sid)["status"] == "running"
    assert s2.list_sources()[0]["url"] == "https://example.org"
    assert s2.task_hash_exists(t.content_hash)


def test_requeue_unfinished(tmp_path):
    s = Storage(tmp_path / "x.db")
    t = mk()
    s.insert_task(t)
    s.set_task_status(t.task_id, "in_progress")
    assert s.requeue_unfinished_tasks() == 1
    assert s.get_task_row(t.task_id)["status"] == "pending"

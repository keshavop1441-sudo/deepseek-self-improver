"""SQLite persistence layer.

One `Storage` object wraps a single connection guarded by a lock so the GUI
thread and the worker thread can both use it safely.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    planned_minutes REAL NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,              -- running|completed|stopped|interrupted|failed
    stop_reason TEXT,
    benchmark_before_id INTEGER,
    benchmark_after_id INTEGER,
    errors_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    adapter TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    retrieved_at TEXT NOT NULL,
    status TEXT NOT NULL,              -- ok|error
    http_status INTEGER,
    error TEXT,
    content_hash TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    difficulty INTEGER NOT NULL,
    question TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_title TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    input_data TEXT NOT NULL,
    verification_method TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|in_progress|verified|failed|abandoned
    session_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    session_id INTEGER,
    attempt_no INTEGER NOT NULL,       -- 0 = original, 1..3 = retries
    prompt TEXT NOT NULL,
    raw_response TEXT,
    thinking TEXT,
    parsed_json TEXT,
    parse_ok INTEGER NOT NULL DEFAULT 0,
    answer TEXT,
    error TEXT,
    duration_s REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id),
    task_id TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,              -- verified|incorrect|uncertain
    verified INTEGER NOT NULL,
    failure_category TEXT,
    message TEXT,
    details_json TEXT,
    source_url TEXT,
    verified_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    failure_pattern TEXT NOT NULL,
    correct_method TEXT NOT NULL,
    example TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    task_id TEXT NOT NULL,
    verification_id INTEGER NOT NULL REFERENCES verifications(verification_id),
    lesson_hash TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    label TEXT NOT NULL,               -- before|after|standalone
    model TEXT NOT NULL,
    benchmark_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    lessons_fingerprint TEXT,
    n_tasks INTEGER NOT NULL,
    overall REAL NOT NULL,
    scores_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    reused_from INTEGER
);
CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts(task_id);
CREATE INDEX IF NOT EXISTS idx_verif_task ON verifications(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- generic helpers -------------------------------------------------
    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def _one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # -- sessions --------------------------------------------------------
    def create_session(self, planned_minutes: float, model: str) -> int:
        cur = self._exec(
            "INSERT INTO sessions(started_at, planned_minutes, model, status) VALUES (?,?,?,'running')",
            (now_iso(), planned_minutes, model),
        )
        return int(cur.lastrowid)

    def update_session(self, session_id: int, **fields: Any) -> None:
        if not fields:
            return
        if "errors" in fields:
            fields["errors_json"] = json.dumps(fields.pop("errors"))
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE sessions SET {cols} WHERE session_id=?", (*fields.values(), session_id))

    def get_session(self, session_id: int) -> Optional[dict]:
        return self._one("SELECT * FROM sessions WHERE session_id=?", (session_id,))

    def list_sessions(self, limit: int = 20) -> list[dict]:
        return self._all("SELECT * FROM sessions ORDER BY session_id DESC LIMIT ?", (limit,))

    def running_sessions(self) -> list[dict]:
        return self._all("SELECT * FROM sessions WHERE status='running'")

    # -- sources ---------------------------------------------------------
    def record_source(self, adapter: str, url: str, title: str | None, status: str,
                      http_status: int | None = None, error: str | None = None,
                      content_hash: str | None = None, retrieved_at: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO sources(adapter,url,title,retrieved_at,status,http_status,error,content_hash)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (adapter, url, title, retrieved_at or now_iso(), status, http_status, error, content_hash),
        )
        return int(cur.lastrowid)

    def list_sources(self, session_start: str | None = None) -> list[dict]:
        if session_start:
            return self._all("SELECT * FROM sources WHERE retrieved_at>=? ORDER BY source_id", (session_start,))
        return self._all("SELECT * FROM sources ORDER BY source_id")

    # -- tasks -----------------------------------------------------------
    def task_hash_exists(self, content_hash: str) -> bool:
        return self._one("SELECT 1 AS x FROM tasks WHERE content_hash=?", (content_hash,)) is not None

    def insert_task(self, task: "Any", session_id: int | None = None) -> bool:
        """Insert a task. Returns False (and stores nothing) if it is a duplicate."""
        try:
            self._exec(
                "INSERT INTO tasks(task_id,domain,difficulty,question,source_url,source_title,retrieved_at,"
                "input_data,verification_method,content_hash,status,session_id,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task.task_id, task.domain, task.difficulty, task.question, task.source_url,
                 task.source_title, task.retrieved_at, json.dumps(task.input_data, sort_keys=True),
                 task.verification_method, task.content_hash, "pending", session_id, now_iso()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def set_task_status(self, task_id: str, status: str, session_id: int | None = None) -> None:
        if session_id is not None:
            self._exec("UPDATE tasks SET status=?, session_id=? WHERE task_id=?", (status, session_id, task_id))
        else:
            self._exec("UPDATE tasks SET status=? WHERE task_id=?", (status, task_id))

    def get_task_row(self, task_id: str) -> Optional[dict]:
        return self._one("SELECT * FROM tasks WHERE task_id=?", (task_id,))

    def pending_task_rows(self, limit: int = 10) -> list[dict]:
        return self._all(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,))

    def requeue_unfinished_tasks(self) -> int:
        cur = self._exec("UPDATE tasks SET status='pending' WHERE status='in_progress'")
        return cur.rowcount

    # -- attempts / verifications ---------------------------------------
    def insert_attempt(self, task_id: str, session_id: int | None, attempt_no: int, prompt: str,
                       raw_response: str | None, thinking: str | None, parsed: dict | None,
                       parse_ok: bool, answer: str | None, error: str | None, duration_s: float) -> int:
        cur = self._exec(
            "INSERT INTO attempts(task_id,session_id,attempt_no,prompt,raw_response,thinking,parsed_json,"
            "parse_ok,answer,error,duration_s,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, session_id, attempt_no, prompt, raw_response, thinking,
             json.dumps(parsed) if parsed is not None else None, int(parse_ok), answer, error,
             duration_s, now_iso()),
        )
        return int(cur.lastrowid)

    def insert_verification(self, attempt_id: int, task_id: str, result: "Any") -> int:
        cur = self._exec(
            "INSERT INTO verifications(attempt_id,task_id,method,status,verified,failure_category,message,"
            "details_json,source_url,verified_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (attempt_id, task_id, result.method, result.status, int(result.verified),
             result.failure_category, result.message, json.dumps(result.details, default=str),
             result.source_url, result.verified_at),
        )
        return int(cur.lastrowid)

    def get_verification(self, verification_id: int) -> Optional[dict]:
        return self._one("SELECT * FROM verifications WHERE verification_id=?", (verification_id,))

    def attempts_for_task(self, task_id: str) -> list[dict]:
        return self._all("SELECT * FROM attempts WHERE task_id=? ORDER BY attempt_no, attempt_id", (task_id,))

    def verifications_for_task(self, task_id: str) -> list[dict]:
        return self._all("SELECT * FROM verifications WHERE task_id=? ORDER BY verification_id", (task_id,))

    # -- lessons ---------------------------------------------------------
    def insert_lesson(self, **f: Any) -> Optional[int]:
        try:
            cur = self._exec(
                "INSERT INTO lessons(domain,failure_pattern,correct_method,example,source,created_at,task_id,"
                "verification_id,lesson_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (f["domain"], f["failure_pattern"], f["correct_method"], f["example"], f["source"],
                 now_iso(), f["task_id"], f["verification_id"], f["lesson_hash"]),
            )
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            return None

    def list_lessons(self, domain: str | None = None, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM lessons"
        params: list[Any] = []
        if domain:
            sql += " WHERE domain=?"
            params.append(domain)
        sql += " ORDER BY lesson_id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self._all(sql, params)

    def lessons_fingerprint(self) -> str:
        row = self._one("SELECT COUNT(*) AS n, COALESCE(MAX(lesson_id),0) AS m FROM lessons") or {}
        return f"{row.get('n', 0)}:{row.get('m', 0)}"

    # -- benchmarks ------------------------------------------------------
    def insert_benchmark(self, session_id: int | None, label: str, model: str, version: str,
                         lessons_fingerprint: str, n_tasks: int, overall: float, scores: dict,
                         results: list, reused_from: int | None = None) -> int:
        cur = self._exec(
            "INSERT INTO benchmarks(session_id,label,model,benchmark_version,created_at,lessons_fingerprint,"
            "n_tasks,overall,scores_json,results_json,reused_from) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, label, model, version, now_iso(), lessons_fingerprint, n_tasks, overall,
             json.dumps(scores), json.dumps(results), reused_from),
        )
        return int(cur.lastrowid)

    def get_benchmark(self, benchmark_id: int) -> Optional[dict]:
        row = self._one("SELECT * FROM benchmarks WHERE benchmark_id=?", (benchmark_id,))
        if row:
            row["scores"] = json.loads(row["scores_json"])
            row["results"] = json.loads(row["results_json"])
        return row

    def latest_benchmark(self, model: str, version: str, n_tasks: int) -> Optional[dict]:
        row = self._one(
            "SELECT benchmark_id FROM benchmarks WHERE model=? AND benchmark_version=? AND n_tasks=?"
            " ORDER BY benchmark_id DESC LIMIT 1", (model, version, n_tasks))
        return self.get_benchmark(row["benchmark_id"]) if row else None

    def list_benchmarks(self, limit: int = 10) -> list[dict]:
        return self._all("SELECT benchmark_id,session_id,label,model,created_at,overall,n_tasks,scores_json"
                         " FROM benchmarks ORDER BY benchmark_id DESC LIMIT ?", (limit,))

    # -- stats -----------------------------------------------------------
    def stats(self) -> dict:
        def n(sql: str) -> int:
            return int((self._one(sql) or {"c": 0})["c"])
        return {
            "sessions": n("SELECT COUNT(*) AS c FROM sessions"),
            "tasks": n("SELECT COUNT(*) AS c FROM tasks"),
            "tasks_verified": n("SELECT COUNT(*) AS c FROM tasks WHERE status='verified'"),
            "tasks_failed": n("SELECT COUNT(*) AS c FROM tasks WHERE status='failed'"),
            "tasks_pending": n("SELECT COUNT(*) AS c FROM tasks WHERE status='pending'"),
            "attempts": n("SELECT COUNT(*) AS c FROM attempts"),
            "verifications": n("SELECT COUNT(*) AS c FROM verifications"),
            "lessons": n("SELECT COUNT(*) AS c FROM lessons"),
            "benchmarks": n("SELECT COUNT(*) AS c FROM benchmarks"),
            "sources": n("SELECT COUNT(*) AS c FROM sources"),
        }


def _verified_examples(self: "Storage") -> list[dict]:
    """Attempts whose verification is verified=1 (the only rows eligible for training)."""
    return self._all(
        "SELECT t.task_id, t.domain, t.question, t.content_hash, t.source_url, a.attempt_id, a.parsed_json, "
        "a.answer, v.verification_id, v.method FROM verifications v "
        "JOIN attempts a ON a.attempt_id = v.attempt_id JOIN tasks t ON t.task_id = v.task_id "
        "WHERE v.verified = 1 AND v.status = 'verified' AND a.parse_ok = 1 ORDER BY v.verification_id")


Storage.verified_examples = _verified_examples  # type: ignore[attr-defined]

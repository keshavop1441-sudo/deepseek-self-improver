"""Objective benchmark runner. The model never grades itself: the deterministic verifier does."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.benchmark.tasks import BENCHMARK_TASKS, benchmark_version, category_of
from app.memory.lessons import LessonMemory
from app.ollama.client import OllamaAborted, OllamaBadResponse, OllamaTimeout
from app.solver import prompts
from app.solver.solver import FATAL, Solver
from app.storage.db import Storage
from app.tasks.model import Task

log = logging.getLogger(__name__)
CATEGORIES = ("numerical", "mathematics", "statistics", "finance", "logic", "coding")
BENCH_SEED = 1234


class BenchmarkAborted(Exception):
    pass


@dataclass
class BenchmarkResult:
    overall: float
    scores: dict[str, float]
    results: list[dict[str, Any]]
    version: str
    lessons_fingerprint: str
    n_tasks: int
    benchmark_id: int | None = None
    reused_from: int | None = None
    complete: bool = True

    def summary(self) -> dict[str, Any]:
        return {"overall": self.overall, **{k: v for k, v in self.scores.items()}, "n_tasks": self.n_tasks,
                "version": self.version, "benchmark_id": self.benchmark_id, "reused_from": self.reused_from}


def score(results: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    n = len(results)
    overall = sum(1 for r in results if r["correct"]) / n if n else 0.0
    scores = {}
    for c in CATEGORIES:
        sub = [r for r in results if r["category"] == c]
        if sub:
            scores[c] = sum(1 for r in sub if r["correct"]) / len(sub)
    return overall, scores


class BenchmarkRunner:
    def __init__(self, solver: Solver, memory: LessonMemory, storage: Storage, model: str):
        self.solver, self.memory, self.storage, self.model = solver, memory, storage, model

    def run(self, tasks: list[Task] | None = None, label: str = "standalone", session_id: int | None = None,
            should_abort: Callable[[], bool] = lambda: False, progress: Callable[[int, int, dict], None] | None = None,
            use_lessons: bool = True, allow_reuse: bool = False, persist: bool = True) -> BenchmarkResult:
        tasks = list(BENCHMARK_TASKS if tasks is None else tasks)
        version = benchmark_version(BENCHMARK_TASKS)
        fp = self.storage.lessons_fingerprint() if use_lessons else "none"
        if allow_reuse and persist:
            prev = self.storage.latest_benchmark(self.model, version, len(tasks))
            if prev and prev["lessons_fingerprint"] == fp:
                res = BenchmarkResult(prev["overall"], prev["scores"], prev["results"], version, fp, len(tasks),
                                      reused_from=prev["benchmark_id"])
                res.benchmark_id = self.storage.insert_benchmark(session_id, label, self.model, version, fp, len(tasks),
                                                                 res.overall, res.scores, res.results,
                                                                 reused_from=prev["benchmark_id"])
                return res
        results: list[dict[str, Any]] = []
        for i, task in enumerate(tasks):
            if should_abort():
                raise BenchmarkAborted(f"benchmark aborted after {i}/{len(tasks)} tasks")
            results.append(self._run_one(task, use_lessons, should_abort))
            if progress:
                progress(i + 1, len(tasks), results[-1])
        overall, scores = score(results)
        res = BenchmarkResult(overall, scores, results, version, fp, len(tasks))
        if persist:
            res.benchmark_id = self.storage.insert_benchmark(session_id, label, self.model, version, fp, len(tasks),
                                                             overall, scores, results)
        return res

    def _run_one(self, task: Task, use_lessons: bool, should_abort: Callable[[], bool]) -> dict[str, Any]:
        text = self.memory.format_for_prompt(self.memory.relevant(task, self.solver.k)) if use_lessons else ""
        row: dict[str, Any] = {"task_id": task.task_id, "category": category_of(task), "domain": task.domain,
                               "correct": False, "status": "error", "failure_category": None, "answer": None,
                               "message": None}
        try:
            parsed, dur = self.solver.ask(task, prompts.build_messages(task, text), should_abort, None,
                                          temperature=0.0, seed=BENCH_SEED)
        except OllamaAborted as e:
            raise BenchmarkAborted(str(e)) from e
        except FATAL:
            raise
        except (OllamaTimeout, OllamaBadResponse) as e:
            row.update(status="error", message=f"{type(e).__name__}: {e}")
            return row
        ver = self.solver.verifier.verify(task, parsed)
        row.update(correct=ver.verified, status=ver.status, failure_category=ver.failure_category,
                   answer=(parsed.answer or "")[:200], message=ver.message, duration_s=round(dur, 1))
        return row

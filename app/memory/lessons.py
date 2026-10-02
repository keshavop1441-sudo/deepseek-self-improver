"""Verified memory: lessons are created ONLY from independently verified results."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.storage.db import Storage
from app.tasks.model import Task

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "to", "in", "is", "what", "for", "that", "with", "as", "at", "by", "on",
         "answer", "field", "only", "put", "number", "round", "decimal", "places"}


class UnverifiedLessonError(Exception):
    """Raised when someone tries to store a lesson without a verified verification row."""


def _tokens(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


class LessonMemory:
    def __init__(self, storage: Storage):
        self.storage = storage

    # -------------------------------------------------------------- write
    def add_lesson(self, task: Task, verification_id: int, answer: str, method: str,
                   failure_pattern: str | None) -> int | None:
        """Store a lesson. Refuses unless `verification_id` points at a verified=1 verification
        of this very task (defence in depth: the controller already checks, the DB is the authority)."""
        ver = self.storage.get_verification(verification_id)
        if not ver or not ver["verified"] or ver["status"] != "verified" or ver["task_id"] != task.task_id:
            raise UnverifiedLessonError("lessons may only be created from independently verified results")
        pattern = failure_pattern or "none (solved on first attempt)"
        correct = _clip(method, 500) or f"Final answer {_clip(answer, 80)} confirmed by {task.verification_method}"
        example = f"Q: {_clip(task.question, 400)} | Verified answer: {_clip(answer, 120)}"
        h = hashlib.sha256(f"{task.content_hash}|{correct}".encode()).hexdigest()
        return self.storage.insert_lesson(
            domain=task.domain, failure_pattern=_clip(pattern, 300), correct_method=correct, example=example,
            source=task.source_url, task_id=task.task_id, verification_id=verification_id, lesson_hash=h)

    # -------------------------------------------------------------- read
    def relevant(self, task: Task, k: int = 3) -> list[dict[str, Any]]:
        """Small, relevant subset (never the whole DB): same-domain lessons ranked by word overlap."""
        pool = self.storage.list_lessons(domain=task.domain, limit=200)
        q = _tokens(task.question)
        scored = []
        for les in pool:
            if les["task_id"] == task.task_id:
                continue
            lt = _tokens(les["example"] + " " + les["correct_method"])
            overlap = len(q & lt) / (len(q | lt) or 1)
            bonus = 0.15 if not les["failure_pattern"].startswith("none") else 0.0
            scored.append((overlap + bonus, les["lesson_id"], les))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [s[2] for s in scored[:k]]

    @staticmethod
    def format_for_prompt(lessons: list[dict[str, Any]]) -> str:
        if not lessons:
            return ""
        lines = ["Verified lessons from earlier problems (independently checked; use only if relevant):"]
        for les in lessons:
            pitfall = "" if les["failure_pattern"].startswith("none") else f" Pitfall: {_clip(les['failure_pattern'], 160)}."
            lines.append(f"- [{les['domain']}]{pitfall} Method: {_clip(les['correct_method'], 220)}")
        return "\n".join(lines)


def learn_from_outcome(memory: LessonMemory, outcome: Any) -> int | None:
    """Create a lesson from a SolveOutcome, only when it ended verified."""
    if outcome.status != "verified" or not outcome.final or not outcome.final.verification:
        return None
    final = outcome.final
    if not final.verification.verified or final.verification_id is None or final.answer is None:
        return None
    failed = [a for a in outcome.attempts if a.verification and not a.verification.verified]
    pattern = None
    if failed:
        v = failed[0].verification
        pattern = f"{v.failure_category}: {v.message}"
    return memory.add_lesson(outcome.task, final.verification_id, final.answer.answer, final.answer.method, pattern)

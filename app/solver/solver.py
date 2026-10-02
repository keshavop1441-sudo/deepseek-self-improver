"""Solve one task: ask -> parse -> independently verify -> retry (max 3) with verifier feedback."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.memory.lessons import LessonMemory
from app.ollama.client import (ModelNotAvailable, OllamaAborted, OllamaBadResponse, OllamaClient, OllamaTimeout,
                               OllamaUnavailable)
from app.ollama.parsing import ModelAnswer, parse_model_output
from app.solver import prompts
from app.storage.db import Storage
from app.tasks.model import Task
from app.verifier import VerificationResult, Verifier
from app.verifier.code_runner import extract_code

log = logging.getLogger(__name__)

FATAL = (OllamaUnavailable, ModelNotAvailable)


@dataclass
class AttemptRecord:
    attempt_no: int
    answer: ModelAnswer | None
    verification: VerificationResult | None
    attempt_id: int | None = None
    verification_id: int | None = None
    error: str | None = None


@dataclass
class SolveOutcome:
    task: Task
    status: str                       # verified | failed | invalid | aborted
    attempts: list[AttemptRecord] = field(default_factory=list)
    lesson_id: int | None = None
    started: int = 0                  # model calls handed out; > len(attempts) when a call was cut off by the deadline

    @property
    def retries(self) -> int:
        """Retries handed to the model (the first call is not a retry), including one aborted mid-call."""
        return max(0, self.started - 1)

    @property
    def final(self) -> AttemptRecord | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def failure_categories(self) -> list[str]:
        return [a.verification.failure_category for a in self.attempts
                if a.verification and not a.verification.verified and a.verification.failure_category]


class Solver:
    def __init__(self, client: OllamaClient, verifier: Verifier, storage: Storage, memory: LessonMemory,
                 max_retries: int = 3, lessons_per_prompt: int = 3, request_timeout: float = 300.0):
        self.client, self.verifier, self.storage, self.memory = client, verifier, storage, memory
        self.max_retries, self.k, self.request_timeout = max_retries, lessons_per_prompt, request_timeout

    # ----------------------------------------------------------------- one model call
    def ask(self, task: Task, messages: list[dict], should_abort: Callable[[], bool] | None = None,
            time_left: Callable[[], float] | None = None, temperature: float | None = None,
            seed: int | None = None) -> tuple[ModelAnswer, float]:
        """One chat call + parsing. Raises FATAL errors, OllamaAborted, OllamaTimeout, OllamaBadResponse."""
        budget = self.request_timeout if time_left is None else min(self.request_timeout, time_left())
        res = self.client.chat(messages, think=task.difficulty >= 3, timeout=budget,
                               should_abort=should_abort, temperature=temperature, seed=seed)
        parsed = parse_model_output(res.content, res.thinking)
        if task.domain == "coding" and not parsed.answer.strip():
            parsed.answer = extract_code(res.content)   # model ignored JSON but gave a code block
            if parsed.answer.strip():
                parsed.parse_note = (parsed.parse_note + "; code taken from fenced block").strip("; ")
        return parsed, res.duration_s

    # ----------------------------------------------------------------- full solve loop
    def solve(self, task: Task, session_id: int | None, should_abort: Callable[[], bool],
              time_left: Callable[[], float], emit: Callable[..., None] | None = None) -> SolveOutcome:
        emit = emit or (lambda *a, **k: None)
        lessons = self.memory.relevant(task, self.k)
        base = prompts.build_messages(task, self.memory.format_for_prompt(lessons))
        messages = base
        outcome = SolveOutcome(task=task, status="failed")
        for n in range(self.max_retries + 1):
            if should_abort() or time_left() <= 0:
                outcome.status = "aborted"
                return outcome
            emit("attempt_start", task=task, attempt_no=n)
            outcome.started += 1
            prompt_text = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
            try:
                parsed, dur = self.ask(task, messages, should_abort, time_left)
            except OllamaAborted:
                outcome.status = "aborted"
                return outcome
            except FATAL:
                raise
            except (OllamaTimeout, OllamaBadResponse) as e:
                if isinstance(e, OllamaTimeout) and time_left() <= 0:
                    outcome.status = "aborted"
                    return outcome
                aid = self.storage.insert_attempt(task.task_id, session_id, n, prompt_text, None, None, None,
                                                  False, None, f"{type(e).__name__}: {e}", 0.0)
                outcome.attempts.append(AttemptRecord(n, None, None, aid, None, error=str(e)))
                emit("attempt_error", task=task, attempt_no=n, error=str(e))
                if n < self.max_retries:
                    emit("retry", task=task, attempt_no=n + 1)
                continue
            aid = self.storage.insert_attempt(task.task_id, session_id, n, prompt_text, parsed.raw, parsed.thinking,
                                              parsed.to_dict(), parsed.parse_ok, parsed.answer, None, dur)
            ver = self.verifier.verify(task, parsed)
            vid = self.storage.insert_verification(aid, task.task_id, ver)
            rec = AttemptRecord(n, parsed, ver, aid, vid)
            outcome.attempts.append(rec)
            emit("verification", task=task, attempt_no=n, result=ver)
            if ver.verified:
                outcome.status = "verified"
                return outcome
            if ver.task_invalid:
                outcome.status = "invalid"
                return outcome
            if n < self.max_retries:
                emit("retry", task=task, attempt_no=n + 1)
                messages = prompts.build_retry_messages(base, parsed.raw or parsed.answer, parsed.answer,
                                                        ver.failure_category or "other", ver.message)
        outcome.status = "failed"
        return outcome

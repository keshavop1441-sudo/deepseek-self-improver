"""Solve one task: ask -> parse -> independently verify -> retry (max 3) with verifier feedback."""
from __future__ import annotations

import logging
import math
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


@dataclass(frozen=True)
class CallBudget:
    """Per-call budget for learning-loop model calls (Self Improve). Mirrors the Config fields of the same name."""
    call_share: float = 0.5
    min_call_seconds: float = 20.0
    min_tokens: int = 256
    assumed_tokens_per_second: float = 18.0


def measured_tokens_per_second(raw: Any) -> float | None:
    """Generation rate from Ollama's response metadata (eval_count tokens / eval_duration ns); None if unusable."""
    if not isinstance(raw, dict):
        return None
    count, dur_ns = raw.get("eval_count"), raw.get("eval_duration")
    for v in (count, dur_ns):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            return None
    rate = count / (dur_ns / 1e9)
    return rate if math.isfinite(rate) and 0.1 <= rate <= 100_000 else None


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
                 max_retries: int = 3, lessons_per_prompt: int = 3, request_timeout: float = 300.0,
                 budget: CallBudget | None = None):
        self.client, self.verifier, self.storage, self.memory = client, verifier, storage, memory
        self.max_retries, self.k, self.request_timeout = max_retries, lessons_per_prompt, request_timeout
        self.budget = budget or CallBudget()
        self.last_result: Any = None            # ChatResult of the latest call that returned (set by ask())
        self._rate: tuple[Any, float] | None = None   # (session_id, tokens/sec) from the latest returned call

    # ----------------------------------------------------------------- Self Improve per-call budget
    def call_wall_budget(self, remaining: float) -> float:
        """Seconds one model call may take: a share of the remaining session time, floored at min_call_seconds,
        but never above request_timeout or the time actually remaining."""
        remaining = max(0.0, remaining)
        b = self.budget
        return min(self.request_timeout, remaining, max(b.min_call_seconds, remaining * b.call_share))

    def call_token_cap(self, wall_budget: float, session_id: Any = None) -> int:
        """Generation-token cap = floor(wall_budget * tokens_per_second), clamped to [min_tokens, client.num_predict]
        (or just client.num_predict when that is itself below min_tokens). Always >= 1."""
        b, cap = self.budget, int(self.client.num_predict)
        rate = self._rate[1] if self._rate and self._rate[0] == session_id else None
        if rate is None:
            rate = b.assumed_tokens_per_second
        est = math.floor(max(0.0, wall_budget) * rate) if math.isfinite(rate) and rate > 0 else 0
        if cap < b.min_tokens:
            return max(1, cap)
        return max(1, b.min_tokens, min(cap, est))

    # ----------------------------------------------------------------- one model call
    def ask(self, task: Task, messages: list[dict], should_abort: Callable[[], bool] | None = None,
            time_left: Callable[[], float] | None = None, temperature: float | None = None,
            seed: int | None = None, num_predict: int | None = None) -> tuple[ModelAnswer, float]:
        """One chat call + parsing. Raises FATAL errors, OllamaAborted, OllamaTimeout, OllamaBadResponse."""
        budget = self.request_timeout if time_left is None else min(self.request_timeout, time_left())
        res = self.client.chat(messages, think=task.difficulty >= 3, timeout=budget,
                               should_abort=should_abort, temperature=temperature, seed=seed, num_predict=num_predict)
        self.last_result = res
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
            remaining = time_left()
            if should_abort() or remaining <= 0:
                outcome.status = "aborted"
                return outcome
            wall = self.call_wall_budget(remaining)      # recomputed for every attempt from the time left now
            n_predict = self.call_token_cap(wall, session_id)
            emit("attempt_start", task=task, attempt_no=n)
            outcome.started += 1
            prompt_text = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
            try:
                self.last_result = None
                parsed, dur = self.ask(task, messages, should_abort, lambda w=wall: w, num_predict=n_predict)
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
            rate = measured_tokens_per_second(getattr(self.last_result, "raw", None))
            if rate is not None:                         # only from a call that actually returned
                self._rate = (session_id, rate)
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

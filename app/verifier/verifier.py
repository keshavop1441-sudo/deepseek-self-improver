"""Independent verifier layer: verify(task, model_answer) -> VerificationResult.

Nothing from the model's self_check/method/confidence is consulted for the decision; only the
final `answer` is compared with a deterministic recomputation from the task's own data.
"""
from __future__ import annotations

import logging
import re
from fractions import Fraction
from typing import Any

from app.discovery.http import HttpClient
from app.tasks.model import Task
from app.verifier import compute, source_check, symbolic
from app.verifier.base import (VerificationResult, incorrect, invalid_task, uncertain, verified)
from app.verifier.code_runner import extract_code, run_code_tests
from app.verifier.numeric import (categorize_numeric, check_number, extract_numbers,
                                  extract_single_number)

log = logging.getLogger(__name__)

YES = {"yes", "valid", "true", "entails", "entailed", "tautology", "is a tautology"}
NO = {"no", "invalid", "false", "not valid", "not a tautology", "does not entail"}


class Verifier:
    def __init__(self, http: HttpClient | None = None, code_timeout: float = 10.0):
        self.http = http
        self.code_timeout = code_timeout
        self._recheck_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ public
    def verify(self, task: Task, model_answer: Any) -> VerificationResult:
        answer = getattr(model_answer, "answer", model_answer)
        answer = "" if answer is None else str(answer)
        method = task.verification_method
        try:
            res = self._dispatch(task, answer, method)
        except compute.InvalidTaskInput as e:
            return invalid_task(method, f"Task data cannot be verified: {e}", source_url=task.source_url)
        except Exception as e:  # noqa: BLE001 - a verifier crash must never become "verified"
            log.exception("verifier crashed")
            return uncertain(method, f"Verifier error: {type(e).__name__}: {e}",
                             details={"verifier_error": str(e)}, source_url=task.source_url)
        if res.source_url is None:
            res.source_url = task.source_url
        return res

    # --------------------------------------------------------------- internals
    def _dispatch(self, task: Task, answer: str, method: str) -> VerificationResult:
        d = task.input_data
        # 1. Independently re-validate recorded inputs against the public source (when possible).
        recheck = self._recheck(task)
        if recheck and recheck["state"] == "mismatch":
            return invalid_task(method, "Recorded task inputs no longer match the public source: "
                                + recheck["note"], details={"source_recheck": recheck})
        extra = {"source_recheck": recheck} if recheck else {}

        if method == "code_tests":
            return self._code(task, answer, extra)
        if method == "sympy_symbolic":
            return self._symbolic(task, answer, extra)
        if method not in compute.METHODS:
            return uncertain(method, f"No verifier registered for method '{method}'", details=extra)
        expected = compute.METHODS[method](d)
        if isinstance(expected, str):
            return self._compare_text(method, expected, answer, extra)
        return self._compare_number(method, expected, answer, d, extra)

    def _recheck(self, task: Task) -> dict | None:
        spec = task.input_data.get("recheck")
        if not spec:
            return None
        if task.task_id not in self._recheck_cache:
            self._recheck_cache[task.task_id] = source_check.recheck(self.http, spec.get("url") or task.source_url, spec)
        return self._recheck_cache[task.task_id]

    def _compare_number(self, method: str, expected: Fraction, answer: str, d: dict,
                        extra: dict) -> VerificationResult:
        got, note = extract_single_number(answer)
        details = {"expected": str(float(expected)), "expected_exact": str(expected),
                   "got": str(float(got)) if got is not None else None, **extra}
        if got is not None and check_number(expected, got, d):
            return verified(method, "answer matches independent recomputation", details=details)
        cat, hint = categorize_numeric(expected, got, answer, note)
        if got is None and answer.strip() and "several different" in note:
            # ambiguous formatting is "uncertain": we cannot tell which number is the final answer
            return uncertain(method, hint, category="interpretation", details=details)
        return incorrect(method, cat, hint, details=details)

    def _compare_text(self, method: str, expected: str, answer: str, extra: dict) -> VerificationResult:
        if expected in ("yes", "no"):
            a = re.sub(r"[^a-z ]", " ", answer.lower()).split()
            text = " ".join(a)
            says_yes = text in YES or (a[:1] == ["yes"]) or (a[-1:] == ["yes"])
            says_no = text in NO or (a[:1] == ["no"]) or (a[-1:] == ["no"])
            if says_yes == says_no:
                return uncertain(method, "Answer must clearly be 'yes' or 'no'.", category="interpretation",
                                 details=extra)
            got = "yes" if says_yes else "no"
        else:
            toks = re.findall(r"[0-9A-Za-z]+", answer)
            got = toks[-1].upper() if toks else ""
        details = {"expected": expected, "got": got, **extra}
        if got == expected:
            return verified(method, "answer matches independent recomputation", details=details)
        return incorrect(method, "formula" if expected in ("yes", "no") else "arithmetic",
                         "The final answer does not match the independent recomputation.", details=details)

    def _symbolic(self, task: Task, answer: str, extra: dict) -> VerificationResult:
        d, method = task.input_data, "sympy_symbolic"
        kind = d.get("kind")
        if kind in ("derivative", "antiderivative"):
            exp = symbolic.expected_expr(d)
            got = symbolic.parse_answer_expr(answer)
            if got is None:
                return incorrect(method, "interpretation", "Could not read the final answer as an expression in x.",
                                 details=extra)
            if kind == "derivative":
                ok = symbolic.equivalent(got, exp)
            else:  # antiderivative: constant of integration is free
                import sympy
                ok = symbolic.equivalent(sympy.diff(got, sympy.Symbol("x")), sympy.diff(exp, sympy.Symbol("x")))
            details = {"expected": str(exp), "got": str(got), **extra}
            if ok:
                return verified(method, "answer is symbolically equivalent", details=details)
            return incorrect(method, "formula", "The expression is not equivalent to the correct result.",
                             details=details)
        if kind == "roots":
            exp = symbolic.expected_roots(d)
            got = sorted(set(extract_numbers(answer)))
            details = {"expected": [str(r) for r in exp], "got": [str(g) for g in got], **extra}
            if not got:
                return incorrect(method, "interpretation", "No numeric roots found in the answer.", details=details)
            if got == exp:
                return verified(method, "all real roots match", details=details)
            return incorrect(method, "arithmetic", "The set of real roots is not correct.", details=details)
        exp = symbolic.expected_value(d)
        return self._compare_number(method, exp, answer, d, extra)

    def _code(self, task: Task, answer: str, extra: dict) -> VerificationResult:
        d, method = task.input_data, "code_tests"
        if "function_name" not in d or not d.get("tests"):
            raise compute.InvalidTaskInput("coding task needs function_name and tests")
        code = extract_code(answer)
        if not code.strip():
            return incorrect(method, "missing_data", "No code was provided.", details=extra)
        r = run_code_tests(code, d["function_name"], d["tests"], timeout=self.code_timeout)
        details = {"passed": r.passed, "total": r.total, "error": r.error, "timed_out": r.timed_out, **extra}
        if r.ok:
            return verified(method, f"all {r.total} hidden tests passed", details=details)
        msg = f"Code failed {r.total - r.passed}/{r.total} tests."
        if r.error:
            msg += f" First error: {r.error}"
        return incorrect(method, "coding_error", msg, details=details)

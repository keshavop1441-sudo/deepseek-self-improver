"""Test doubles: fake Ollama backend, fake internet, fake clock, oracle answers."""
from __future__ import annotations

import json
import threading
from fractions import Fraction
from typing import Any, Callable

from app.discovery.adapters import Adapter, AdapterError
from app.discovery.http import FetchError, Response
from app.tasks.model import Task
from app.verifier import compute, symbolic

CODE = {
    "primes_up_to": "def primes_up_to(n):\n    return [i for i in range(2, n + 1) if all(i % j for j in range(2, int(i ** 0.5) + 1))]\n",
    "to_roman": ("def to_roman(n):\n    out = ''\n    for v, s in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),"
                 "(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:\n        while n >= v:\n            out += s\n            n -= v\n    return out\n"),
    "binary_search": "def binary_search(xs, target):\n    return xs.index(target) if target in xs else -1\n",
    "fib": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "is_palindrome": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
    "sum_even": "def sum_even(xs):\n    return sum(x for x in xs if x % 2 == 0)\n",
    "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
    "fizzbuzz": ("def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        out.append('FizzBuzz' if i % 15 == 0 else "
                 "'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i))\n    return out\n"),
}


def correct_answer(task: Task) -> str:
    """Independent ground truth for tests (uses the same math as the verifier, never the model)."""
    d, m = task.input_data, task.verification_method
    if m == "code_tests":
        return CODE[d["function_name"]]
    if m == "sympy_symbolic":
        k = d["kind"]
        if k == "derivative":
            return str(symbolic.expected_expr(d))
        if k == "roots":
            return ", ".join(str(r) for r in symbolic.expected_roots(d))
        return f"{float(symbolic.expected_value(d)):.6f}"
    exp = compute.METHODS[m](d)
    if isinstance(exp, str):
        return exp
    return str(int(exp)) if Fraction(exp).denominator == 1 else f"{float(exp):.6f}"


def wrong_answer(task: Task) -> str:
    if task.verification_method == "code_tests":
        return "def " + task.input_data["function_name"] + "(*a):\n    return None\n"
    if task.verification_method == "sympy_symbolic" and task.input_data["kind"] == "derivative":
        return "x"
    if task.verification_method == "logic_truth_table" and task.input_data["kind"] in ("tautology", "entailment"):
        return "maybe"
    return "-999999.123"


def model_json(answer: str, method: str = "worked it out", self_check: str = "I am 100% sure") -> str:
    return json.dumps({"answer": answer, "method": method, "calculations": ["step"], "assumptions": [],
                       "self_check": self_check})


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeOllama:
    """Transport-compatible fake of an Ollama server."""

    def __init__(self, responder: Callable[[list[dict], dict], str] | None = None, models: list[str] | None = None,
                 up: bool = True, version: str = "0.9.0", on_chat: Callable[[int], None] | None = None,
                 delay: float = 0.0):
        self.responder = responder or (lambda messages, payload: model_json("0"))
        self.models = ["deepseek-r1:1.5b"] if models is None else models
        self.up, self.version, self.on_chat, self.delay = up, version, on_chat, delay
        self.chat_calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, method: str, url: str, payload: dict | None, timeout: float):
        from app.ollama.client import OllamaUnavailable
        if not self.up:
            raise OllamaUnavailable("connection refused (fake)")
        if url.endswith("/api/version"):
            return 200, {"version": self.version}
        if url.endswith("/api/tags"):
            return 200, {"models": [{"name": m, "model": m} for m in self.models]}
        if url.endswith("/api/chat"):
            with self.lock:
                self.chat_calls.append(payload)
                n = len(self.chat_calls)
            if self.delay:
                import time
                time.sleep(self.delay)
            if self.on_chat:
                self.on_chat(n)
            if payload["model"] not in self.models:
                return 404, {"error": "model not found"}
            content = self.responder(payload["messages"], payload)
            return 200, {"model": payload["model"], "message": {"role": "assistant", "content": content},
                         "done": True}
        return 404, "not found"


class FakeHttp:
    def __init__(self, routes: dict[str, Any] | None = None):
        self.routes = routes or {}
        self.calls: list[str] = []

    def get(self, url: str, timeout=None, headers=None) -> Response:
        self.calls.append(url)
        val = self.routes.get(url)
        if val is None:
            raise FetchError(f"no route for {url} (fake: offline)", 404)
        if isinstance(val, Exception):
            raise val
        return Response(url, 200, val)


class StaticAdapter(Adapter):
    """Yields a new simple gcd task each call (no internet involved)."""
    name = "static_test"
    domains = ("numerical_reasoning",)

    def __init__(self, fail: bool = False):
        super().__init__(FakeHttp())
        self.n = 0
        self.fail = fail

    def discover(self, rng):
        if self.fail:
            raise AdapterError("static_test: simulated source failure")
        self.n += 1
        a, b = 48 + self.n * 6, 36 + self.n * 4
        return [Task("numerical_reasoning", 1, f"What is the gcd of {a} and {b}?", "https://example.org/fake-source",
                     "Static test source", "numeric_computation", {"kind": "gcd", "a": a, "b": b})]


def user_text(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def make_oracle(bench_needs_lessons: bool = False, bench_always_correct: bool = False, wrong_first_gcd: bool = True,
                always_wrong: bool = False):
    """Fake model brain. Benchmark items are answered correctly depending on the flags, gcd training tasks are
    answered wrong once (to exercise retry) and then right."""
    import re
    from app.benchmark.tasks import BENCHMARK_TASKS
    by_q = {t.question: t for t in BENCHMARK_TASKS}
    seen: dict[str, int] = {}

    def responder(messages: list[dict], payload: dict) -> str:
        text = user_text(messages)
        bt = next((t for q, t in by_q.items() if q in text), None)
        if bt is not None:
            ok = bench_always_correct or (bench_needs_lessons and "Verified lessons" in text and bt.domain == "numerical_reasoning")
            return model_json(correct_answer(bt) if ok and not always_wrong else wrong_answer(bt))
        m = re.search(r"gcd of (\d+) and (\d+)", text)
        if m:
            from math import gcd
            key = m.group(0)
            seen[key] = seen.get(key, 0) + 1
            right = str(gcd(int(m.group(1)), int(m.group(2))))
            if always_wrong or (wrong_first_gcd and seen[key] == 1):
                return model_json("1000001")
            return model_json(right, method="Euclid's algorithm: repeatedly take remainders")
        return model_json("0")
    return responder

"""Run model-written Python against fixed tests in an isolated temp dir with a hard timeout.

Isolation: fresh temp directory as cwd, `python -I` (isolated mode), scrubbed environment,
wall-clock timeout, and (on POSIX) CPU/memory rlimits. This is NOT a security sandbox
against a determined attacker; it is meant to contain accidents from a small local model.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

HARNESS = r'''
import json, sys, traceback
sys.setrecursionlimit(5000)
spec = json.load(open("spec.json"))
ns = {"__name__": "solution"}
out = {"load_error": None, "results": []}
try:
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
    fn = ns[spec["function_name"]]
except BaseException as e:
    out["load_error"] = "%s: %s" % (type(e).__name__, e)
    fn = None
if fn is not None:
    for args, expected in spec["tests"]:
        try:
            got = fn(*args)
            out["results"].append({"ok": got == expected or (isinstance(got, tuple) and list(got) == expected), "error": None})
        except BaseException as e:
            out["results"].append({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
sys.stdout.write("\n@@RESULT@@" + json.dumps(out))
'''


@dataclass
class CodeRunResult:
    ok: bool
    passed: int
    total: int
    error: str | None
    timed_out: bool = False


def extract_code(answer: str) -> str:
    blocks = FENCE.findall(answer or "")
    return (blocks[-1] if blocks else (answer or "")).strip("\n")


def _limits() -> None:  # pragma: no cover - runs in child (POSIX only)
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


def run_code_tests(code: str, function_name: str, tests: list[Any], timeout: float = 10.0) -> CodeRunResult:
    if not code.strip():
        return CodeRunResult(False, 0, len(tests), "empty code")
    with tempfile.TemporaryDirectory(prefix="selfimp_code_") as tmp:
        with open(os.path.join(tmp, "solution.py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(os.path.join(tmp, "spec.json"), "w", encoding="utf-8") as fh:
            json.dump({"function_name": function_name, "tests": tests}, fh)
        with open(os.path.join(tmp, "harness.py"), "w", encoding="utf-8") as fh:
            fh.write(HARNESS)
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
               "PYTHONIOENCODING": "utf-8"}
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits
        try:
            proc = subprocess.run([sys.executable, "-I", "harness.py"], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired:
            return CodeRunResult(False, 0, len(tests), f"timed out after {timeout:.0f}s", timed_out=True)
        marker = proc.stdout.rfind("@@RESULT@@")
        if marker < 0:
            err = (proc.stderr or proc.stdout or "no output").strip().splitlines()[-1:] or ["no output"]
            return CodeRunResult(False, 0, len(tests), f"crashed: {err[0][:200]}")
        data = json.loads(proc.stdout[marker + 10:])
        if data["load_error"]:
            return CodeRunResult(False, 0, len(tests), data["load_error"][:200])
        res = data["results"]
        passed = sum(1 for r in res if r["ok"])
        first_err = next((r["error"] for r in res if r["error"]), None)
        return CodeRunResult(passed == len(tests), passed, len(tests), first_err)

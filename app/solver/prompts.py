"""Prompt construction. The model is asked for strict JSON; its self_check is never trusted."""
from __future__ import annotations

from app.tasks.model import Task

SYSTEM_PROMPT = """You are a careful quantitative problem solver. Solve the task and reply with ONE JSON object and nothing else:
{"answer": "<final answer only>", "method": "<short description of the method>", "calculations": ["<step>", ...], "assumptions": ["<assumption>", ...], "self_check": "<how you checked it>"}
Rules: the "answer" must contain only the final value (a number, yes/no, an expression in x, a comma-separated list of roots, a base-N string, or Python source code for coding tasks). Do not wrap the JSON in markdown."""

CODING_HINT = ("For coding tasks put the complete Python function source in \"answer\" (escape newlines as \\n). "
               "Use only the standard library.")


def build_messages(task: Task, lessons_text: str = "") -> list[dict]:
    parts = []
    if lessons_text:
        parts.append(lessons_text)
    parts.append(f"Domain: {task.domain}\nTask:\n{task.question}")
    if task.domain == "coding":
        parts.append(CODING_HINT)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def build_retry_messages(base: list[dict], previous_response: str, previous_answer: str, failure_category: str,
                         failure_message: str) -> list[dict]:
    """Give the verifier failure back to the model (without revealing the expected answer)."""
    prev = (previous_response or "")[-1500:]
    feedback = (
        f"Your previous answer was: {previous_answer[:300] or '(none)'}\n"
        f"An independent verifier checked it and found it INCORRECT. Failure type: {failure_category}. "
        f"Verifier feedback: {failure_message}\n"
        "Locate the specific error in your earlier work (name the step), then produce a corrected solution. "
        "Reply with the same single JSON object; put the specific error you found in \"self_check\"."
    )
    return base + [{"role": "assistant", "content": prev or previous_answer}, {"role": "user", "content": feedback}]

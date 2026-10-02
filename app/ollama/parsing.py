"""Robust parsing of (possibly malformed) model output into the answer schema."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass
class ModelAnswer:
    answer: str = ""
    method: str = ""
    calculations: list[Any] = field(default_factory=list)
    assumptions: list[Any] = field(default_factory=list)
    self_check: str = ""   # recorded for transparency only; NEVER used as verification
    parse_ok: bool = False
    raw: str = ""
    thinking: str = ""
    parse_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "method": self.method, "calculations": self.calculations,
                "assumptions": self.assumptions, "self_check": self.self_check}


def split_thinking(text: str) -> tuple[str, str]:
    """Return (content_without_think, thinking)."""
    thoughts = THINK_RE.findall(text or "")
    content = THINK_RE.sub("", text or "")
    # Unterminated <think> (e.g. truncated output): everything after it is thinking.
    if "<think>" in content.lower():
        idx = content.lower().index("<think>")
        thoughts.append(content[idx + 7:])
        content = content[:idx]
    return content.strip(), "\n".join(t.strip() for t in thoughts)


def _balanced_objects(text: str) -> list[str]:
    """Extract top-level {...} substrings, respecting strings."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
    return out


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _from_obj(obj: dict[str, Any]) -> ModelAnswer | None:
    lower = {str(k).lower(): v for k, v in obj.items()}
    if "answer" not in lower:
        return None
    ans = lower["answer"]
    if isinstance(ans, (dict, list)):
        ans = json.dumps(ans)
    return ModelAnswer(
        answer=str(ans).strip(), method=str(lower.get("method", "") or ""),
        calculations=_as_list(lower.get("calculations")), assumptions=_as_list(lower.get("assumptions")),
        self_check=str(lower.get("self_check", "") or ""), parse_ok=True,
    )


def parse_model_output(text: str, thinking: str = "") -> ModelAnswer:
    """Parse model output. Never raises. `parse_ok` False means the schema was not found."""
    raw = text or ""
    content, inline_think = split_thinking(raw)
    think = "\n".join(x for x in (thinking, inline_think) if x)
    candidates: list[str] = []
    for m in FENCE_RE.findall(content):
        candidates.extend(_balanced_objects(m) or [m.strip()])
    candidates.extend(_balanced_objects(content))
    # prefer the last object that has an "answer" key (models often draft then finalize)
    for cand in reversed(candidates):
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand), cand.replace("'", '"')):
            try:
                obj = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                parsed = _from_obj(obj)
                if parsed:
                    parsed.raw, parsed.thinking = raw, think
                    return parsed
    # Fallback: "answer: ..." line. Marked parse_ok=False (schema not followed) but answer extracted.
    m = re.search(r"(?im)^\W*(?:final\s+)?answer\W*[:=]\s*(.+)$", content)
    if m:
        return ModelAnswer(answer=m.group(1).strip().strip("*`\"' "), parse_ok=False, raw=raw,
                           thinking=think, parse_note="no valid JSON; extracted 'answer:' line")
    return ModelAnswer(answer="", parse_ok=False, raw=raw, thinking=think,
                       parse_note="no JSON object with an 'answer' field found")

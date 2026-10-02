"""Task object shared by discovery, solver, verifier and benchmark."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_content_hash(domain: str, question: str, input_data: dict[str, Any]) -> str:
    """Stable hash of what makes a task *the same task* (source/time excluded)."""
    blob = json.dumps({"d": domain, "q": " ".join(question.split()), "i": input_data},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Task:
    domain: str
    difficulty: int
    question: str
    source_url: str
    source_title: str
    verification_method: str
    input_data: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=utc_now)
    content_hash: str = ""
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.domain, self.question, self.input_data)
        if not self.task_id:
            self.task_id = "t_" + self.content_hash[:16]

    @classmethod
    def from_row(cls, row: dict) -> "Task":
        return cls(
            task_id=row["task_id"], domain=row["domain"], difficulty=row["difficulty"],
            question=row["question"], source_url=row["source_url"], source_title=row["source_title"],
            retrieved_at=row["retrieved_at"], input_data=json.loads(row["input_data"]),
            verification_method=row["verification_method"], content_hash=row["content_hash"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "domain": self.domain, "difficulty": self.difficulty,
            "question": self.question, "source_url": self.source_url,
            "source_title": self.source_title, "retrieved_at": self.retrieved_at,
            "input_data": self.input_data, "verification_method": self.verification_method,
            "content_hash": self.content_hash,
        }

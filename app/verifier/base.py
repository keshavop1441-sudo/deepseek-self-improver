"""Verification result type and failure categories."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

FAILURE_CATEGORIES = (
    "arithmetic", "formula", "interpretation", "unit_conversion", "sign_error",
    "unsupported_assumption", "missing_data", "hallucination", "date_mismatch",
    "source_mismatch", "coding_error", "other",
)

VERIFIED, INCORRECT, UNCERTAIN = "verified", "incorrect", "uncertain"


@dataclass
class VerificationResult:
    """Outcome of an independent, deterministic check.

    `message` is safe to show to the model (never contains the expected answer).
    `details` is private bookkeeping (expected values etc.) kept for reports.
    Only status == "verified" counts as verified; "uncertain" never does.
    """
    status: str
    method: str
    message: str = ""
    failure_category: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    verified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.status not in (VERIFIED, INCORRECT, UNCERTAIN):
            raise ValueError(f"bad status {self.status}")
        if self.status == VERIFIED:
            self.failure_category = None
        elif self.failure_category not in FAILURE_CATEGORIES:
            self.failure_category = "other"

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    @property
    def task_invalid(self) -> bool:
        return bool(self.details.get("task_invalid"))


def verified(method: str, message: str = "independent check matched", **kw: Any) -> VerificationResult:
    return VerificationResult(VERIFIED, method, message, None, kw.pop("details", {}), **kw)


def incorrect(method: str, category: str, message: str, **kw: Any) -> VerificationResult:
    return VerificationResult(INCORRECT, method, message, category, kw.pop("details", {}), **kw)


def uncertain(method: str, message: str, category: str = "other", **kw: Any) -> VerificationResult:
    return VerificationResult(UNCERTAIN, method, message, category, kw.pop("details", {}), **kw)


def invalid_task(method: str, message: str, **kw: Any) -> VerificationResult:
    d = kw.pop("details", {})
    d["task_invalid"] = True
    return VerificationResult(UNCERTAIN, method, message, "missing_data", d, **kw)

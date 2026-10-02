"""Numeric answer extraction and comparison (Fraction/Decimal based, never floats for decisions)."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

NUM_RE = re.compile(r"[-+]?(?:\d+/\d+(?![\d.])|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+)")


def _normalize(text: str) -> str:
    t = str(text).replace("−", "-").replace("–", "-").replace(" ", " ")
    t = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", t)   # thousands separators
    return t.replace("$", " ").replace("USD", " ")


def to_fraction(token: str) -> Fraction | None:
    try:
        if "/" in token:
            a, b = token.split("/")
            return Fraction(int(a), int(b)) if int(b) != 0 else None
        return Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def extract_numbers(text: str) -> list[Fraction]:
    out = []
    for tok in NUM_RE.findall(_normalize(text)):
        f = to_fraction(tok)
        if f is not None:
            out.append(f)
    return out


def extract_single_number(text: str) -> tuple[Fraction | None, str]:
    """Return (value, note). Ambiguous answers (several different numbers) give (None, reason)."""
    nums = extract_numbers(text or "")
    if not nums:
        return None, "no number found in answer"
    if len(set(nums)) > 1:
        return None, "answer contains several different numbers; give exactly one final value"
    return nums[0], ""


def check_number(expected: Fraction, got: Fraction, input_data: dict[str, Any]) -> bool:
    """Compare per the task's tolerance spec.

    decimals: |got-expected| <= 0.5*10^-decimals  (answer rounded to that precision)
    tolerance: {"abs": x} / {"rel": x}
    default: exact for integers, relative 1e-6 otherwise.
    """
    diff = abs(got - expected)
    tol = input_data.get("tolerance")
    if "decimals" in input_data:
        return diff <= Fraction(1, 2) * Fraction(1, 10 ** int(input_data["decimals"]))
    if isinstance(tol, dict):
        if "abs" in tol and diff <= Fraction(str(tol["abs"])):
            return True
        if "rel" in tol and diff <= abs(expected) * Fraction(str(tol["rel"])):
            return True
        return False
    if expected.denominator == 1:
        return got == expected
    return diff <= abs(expected) * Fraction(1, 10 ** 6)


def categorize_numeric(expected: Fraction, got: Fraction | None, answer_text: str, note: str = "") -> tuple[str, str]:
    """Heuristic failure label + a hint that does NOT reveal the expected value."""
    if not (answer_text or "").strip():
        return "missing_data", "No answer was given."
    if got is None:
        return "interpretation", note or "Could not read a single numeric final answer."
    if expected != 0 and got == -expected:
        return "sign_error", "The magnitude is right but the sign is wrong."
    if expected != 0 and got != 0:
        ratio = got / expected
        if ratio in (Fraction(100), Fraction(1, 100), Fraction(1000), Fraction(1, 1000)):
            return "unit_conversion", "The answer is off by a factor of a power of ten (percent/fraction or unit scaling)."
        rel = abs(got - expected) / abs(expected)
        if rel <= Fraction(1, 10):
            return "arithmetic", "The answer is close but numerically wrong; recheck the arithmetic."
    return "formula", "The answer does not match the independent recomputation; recheck the method/formula."

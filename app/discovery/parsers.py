"""Pure parsers for public source formats (shared by adapters and the verifier's re-check)."""
from __future__ import annotations

import html
import json
import re
from typing import Any


def html_to_text(fragment: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", fragment)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


_NUM = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def parse_nist_dataset(text: str) -> list[str]:
    """Extract the data column from a NIST StRD univariate .dat file.

    Data section starts after a 'Data:' line (optionally followed by a column name such as 'y').
    Values are returned as the original strings (to keep full precision).
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*data\s*:\s*\w{0,8}\s*$", ln, re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return []
    vals: list[str] = []
    for ln in lines[start:]:
        toks = ln.split()
        if not toks:
            continue
        if all(_NUM.match(t) for t in toks):
            vals.append(toks[-1])   # NIST rows may be "index value" or "value"; value is last
        else:
            if vals:
                break
    return vals


def parse_treasury_rates(text: str) -> list[dict[str, Any]]:
    """Parse Fiscal Data avg_interest_rates JSON into [{record_date, security_desc, rate_pct}]."""
    try:
        obj = json.loads(text)
        rows = obj["data"]
    except (ValueError, KeyError, TypeError):
        return []
    out = []
    for r in rows:
        try:
            out.append({"record_date": r["record_date"], "security_desc": r["security_desc"],
                        "rate_pct": str(float(r["avg_interest_rate_amt"]))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def sec_facts(text: str, taxonomy: str, tag: str, unit: str = "USD") -> list[dict[str, Any]]:
    """Return fact entries for a us-gaap tag from SEC companyfacts JSON."""
    try:
        obj = json.loads(text)
        return list(obj["facts"][taxonomy][tag]["units"][unit])
    except (ValueError, KeyError, TypeError):
        return []


def sec_entity_name(text: str) -> str:
    try:
        return str(json.loads(text).get("entityName", ""))
    except (ValueError, AttributeError):
        return ""

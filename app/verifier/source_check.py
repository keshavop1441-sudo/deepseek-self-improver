"""Re-retrieve a task's public source and compare the recorded inputs with it.

Result states: "match", "mismatch", "unavailable". `unavailable` never upgrades anything to
verified by itself; callers record it in details so reports show whether inputs were re-validated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.discovery import parsers
from app.discovery.http import FetchError, HttpClient


def recheck(http: HttpClient | None, url: str, spec: dict[str, Any] | None) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"source_url": url, "checked_at": stamp}
    if not spec or http is None:
        return {**base, "state": "unavailable", "note": "no re-check spec or no HTTP client"}
    try:
        resp = http.get(url)
    except FetchError as e:
        return {**base, "state": "unavailable", "note": str(e)}
    kind = spec.get("kind")
    if kind == "sec_facts":
        for want in spec["facts"]:
            entries = parsers.sec_facts(resp.text, want["taxonomy"], want["tag"], want.get("unit", "USD"))
            hit = [e for e in entries if e.get("end") == want["end"] and e.get("accn") == want["accn"]
                   and e.get("start") == want.get("start")]
            if not hit:
                return {**base, "state": "mismatch", "note": f"{want['tag']} {want['end']} not found in source"}
            if not any(str(e.get("val")) == str(want["val"]) for e in hit):
                return {**base, "state": "mismatch", "note": f"{want['tag']} value differs from source"}
        return {**base, "state": "match", "note": "all SEC facts re-confirmed"}
    if kind == "treasury_rate":
        rows = parsers.parse_treasury_rates(resp.text)
        for r in rows:
            if r["record_date"] == spec["record_date"] and r["security_desc"] == spec["security_desc"]:
                same = abs(float(r["rate_pct"]) - float(spec["rate_pct"])) < 1e-9
                return {**base, "state": "match" if same else "mismatch",
                        "note": "rate re-confirmed" if same else "rate differs from source"}
        return {**base, "state": "mismatch", "note": "record not present in source response"}
    if kind == "nist_dataset":
        vals = parsers.parse_nist_dataset(resp.text)
        off, want = int(spec.get("offset", 0)), list(spec["values"])
        same = vals[off:off + len(want)] == want
        return {**base, "state": "match" if same else "mismatch",
                "note": "dataset re-confirmed" if same else "dataset differs from source"}
    if kind == "text_contains":
        text = parsers.html_to_text(resp.text)
        ok = spec["needle"].lower() in text.lower()
        return {**base, "state": "match" if ok else "mismatch",
                "note": "statement re-confirmed" if ok else "statement text changed"}
    return {**base, "state": "unavailable", "note": f"unknown re-check kind {kind}"}

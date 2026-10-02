"""Session report generation (JSON + Markdown)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report(session: dict, state: dict, benchmark_before: dict | None, benchmark_after: dict | None,
                 lessons: list[dict], sources: list[dict], errors: list[str],
                 skipped_reason: str | None = None) -> dict[str, Any]:
    delta = None
    improved: bool | None = None
    note = skipped_reason
    if benchmark_before and benchmark_after:
        if benchmark_before.get("version") != benchmark_after.get("version"):
            note = "benchmark versions differ; no comparison made"
        else:
            delta = round(benchmark_after["overall"] - benchmark_before["overall"], 6)
            improved = delta > 0   # objective score only; the model never judges itself
            note = note or ("objective benchmark score increased" if improved else
                            "no objective improvement (score did not increase)")
    elif note is None:
        note = "benchmark before/after not both available; no improvement claimed"
    return {
        "session_id": session["session_id"], "model": session["model"], "status": session["status"],
        "stop_reason": session.get("stop_reason"), "started_at": session["started_at"], "ended_at": session.get("ended_at"),
        "planned_minutes": session["planned_minutes"], "duration_seconds": round(state.get("elapsed", 0.0), 1),
        "attempted": state["attempted"], "verified": state["verified"], "failed": state["failed"],
        "invalid_tasks": state.get("invalid", 0), "retries": state["retries"],
        "domains": state.get("domains", {}), "failure_categories": state.get("failure_categories", {}),
        "lessons_learned": state["lessons"], "lessons": [
            {"lesson_id": l["lesson_id"], "domain": l["domain"], "failure_pattern": l["failure_pattern"],
             "correct_method": l["correct_method"], "source": l["source"]} for l in lessons],
        "benchmark_before": benchmark_before, "benchmark_after": benchmark_after,
        "benchmark_delta": delta, "improved": improved, "improvement_note": note,
        "sources": [{"adapter": s["adapter"], "url": s["url"], "title": s["title"], "retrieved_at": s["retrieved_at"],
                     "status": s["status"], "error": s["error"]} for s in sources],
        "errors": errors,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _bench_table(b: dict | None) -> list[str]:
    if not b:
        return ["_not available_"]
    rows = ["| category | score |", "|---|---|", f"| **overall** | {_pct(b['overall'])} |"]
    for k, v in b.get("scores", {}).items():
        rows.append(f"| {k} | {_pct(v)} |")
    return rows


def to_markdown(r: dict[str, Any]) -> str:
    m = [f"# Self-improvement session {r['session_id']}", "",
         f"- Model: `{r['model']}`", f"- Status: **{r['status']}** ({r.get('stop_reason') or 'n/a'})",
         f"- Started: {r['started_at']}  |  Ended: {r.get('ended_at')}",
         f"- Planned: {r['planned_minutes']} min  |  Learning loop ran: {r['duration_seconds']} s", "",
         "## Results", "",
         f"- Tasks attempted: **{r['attempted']}**", f"- Independently verified correct: **{r['verified']}**",
         f"- Failed after retries: **{r['failed']}**", f"- Retries used: **{r['retries']}**",
         f"- Tasks with unusable data (not counted against the model): {r['invalid_tasks']}",
         f"- Lessons learned (verified only): **{r['lessons_learned']}**", "",
         "## Domains", ""]
    m += [f"- {d}: {c['verified']} verified / {c['failed']} failed" for d, c in sorted(r["domains"].items())] or ["_none_"]
    m += ["", "## Failure categories (from verifier, includes failed attempts that were later corrected)", ""]
    m += [f"- {k}: {v}" for k, v in sorted(r["failure_categories"].items())] or ["_none_"]
    m += ["", "## Benchmark (objective, fixed tasks, scored by deterministic verifier)", "", "### Before", ""]
    m += _bench_table(r["benchmark_before"]) + ["", "### After", ""] + _bench_table(r["benchmark_after"])
    delta = r["benchmark_delta"]
    m += ["", f"**Delta (overall): {'n/a' if delta is None else f'{delta * 100:+.1f} points'}**  ",
          f"**Improved: {r['improved']}** - {r['improvement_note']}", "", "## Lessons", ""]
    m += [f"- [{l['domain']}] {l['failure_pattern']} -> {l['correct_method'][:200]}" for l in r["lessons"]] or ["_none_"]
    m += ["", "## Sources", ""]
    m += [f"- {s['retrieved_at']} {s['status']} {s['adapter']}: {s['url']}" + (f" ({s['error']})" if s["error"] else "")
          for s in r["sources"]] or ["_none_"]
    m += ["", "## Errors", ""] + ([f"- {e}" for e in r["errors"]] or ["_none_"])
    return "\n".join(m) + "\n"


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = reports_dir / f"session_{report['session_id']:04d}_{stamp}"
    jp, mp = base.with_suffix(".json"), base.with_suffix(".md")
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    mp.write_text(to_markdown(report), encoding="utf-8")
    return jp, mp

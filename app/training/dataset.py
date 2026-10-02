"""Clean training dataset built ONLY from independently verified attempts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.benchmark.tasks import benchmark_hashes
from app.solver.prompts import SYSTEM_PROMPT
from app.storage.db import Storage

SELF_CHECK_NOTE = "Verified by an independent deterministic check."


def build_examples(storage: Storage) -> list[dict[str, Any]]:
    blocked = benchmark_hashes()
    seen: set[str] = set()
    out = []
    for row in storage.verified_examples():
        h = row["content_hash"]
        if h in blocked or h in seen or not row["parsed_json"]:
            continue
        try:
            parsed = json.loads(row["parsed_json"])
        except ValueError:
            continue
        if not str(parsed.get("answer", "")).strip():
            continue
        parsed["self_check"] = SELF_CHECK_NOTE   # never train on the model's own (unverified) self-assessment
        seen.add(h)
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Domain: {row['domain']}\nTask:\n{row['question']}"},
                {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
            ],
            "meta": {"task_id": row["task_id"], "domain": row["domain"], "verification_id": row["verification_id"],
                     "verification_method": row["method"], "source_url": row["source_url"]},
        })
    return out


def prepare_training(storage: Storage, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = build_examples(storage)
    path = out_dir / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    by_domain: dict[str, int] = {}
    for ex in examples:
        by_domain[ex["meta"]["domain"]] = by_domain.get(ex["meta"]["domain"], 0) + 1
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": str(path),
        "examples": len(examples), "by_domain": by_domain,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy": "only attempts with verified=1; benchmark tasks excluded; one example per task",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

"""Fallback chain over adapters with de-duplication and resume of pending tasks."""
from __future__ import annotations

import logging
import random
from typing import Callable, Iterable

from app.discovery.adapters import Adapter, AdapterError
from app.discovery.http import HttpClient, Response
from app.storage.db import Storage
from app.tasks.model import Task

log = logging.getLogger(__name__)


class NoTaskAvailable(Exception):
    """Every adapter failed or produced only duplicates."""


class CachingHttp:
    """Small in-memory cache so repeated discovery does not re-download big files (e.g. SEC JSON)."""

    def __init__(self, inner: HttpClient, max_entries: int = 8):
        self.inner, self.max = inner, max_entries
        self._cache: dict[str, Response] = {}

    def get(self, url: str, timeout: float | None = None, headers: dict | None = None) -> Response:
        if url in self._cache:
            return self._cache[url]
        resp = self.inner.get(url, timeout, headers)
        if len(self._cache) >= self.max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[url] = resp
        return resp


class DiscoveryChain:
    def __init__(self, adapters: list[Adapter], storage: Storage, rng: random.Random | None = None,
                 blocked_hashes: Iterable[str] = ()):
        self.adapters = adapters
        self.storage = storage
        self.rng = rng or random.Random()
        self.blocked = set(blocked_hashes)   # e.g. benchmark task hashes: never used for training
        self._idx = 0
        self._fails: dict[str, int] = {}
        self.errors: list[str] = []

    def next_task(self, session_id: int | None = None) -> Task:
        # 1. Resume tasks stored earlier but never finished.
        for row in self.storage.pending_task_rows(5):
            if row["content_hash"] not in self.blocked:
                return Task.from_row(row)
        # 2. Round-robin adapters (domain variety); temporarily skip repeatedly failing ones.
        n = len(self.adapters)
        order = [self.adapters[(self._idx + i) % n] for i in range(n)]
        self._idx = (self._idx + 1) % n
        order.sort(key=lambda a: self._fails.get(a.name, 0) >= 3)  # stable: failing adapters last
        summary: list[str] = []
        for ad in order:
            try:
                candidates = ad.discover(self.rng)
            except AdapterError as e:
                self._fails[ad.name] = self._fails.get(ad.name, 0) + 1
                msg = str(e)
                summary.append(msg)
                self.errors.append(msg)
                log.warning("adapter failed: %s", msg)
                continue
            except Exception as e:  # noqa: BLE001 - adapter bugs must not kill the session
                self._fails[ad.name] = self._fails.get(ad.name, 0) + 1
                msg = f"{ad.name}: unexpected {type(e).__name__}: {e}"
                summary.append(msg)
                self.errors.append(msg)
                continue
            self._fails[ad.name] = 0
            self.rng.shuffle(candidates)
            for task in candidates:
                if task.content_hash in self.blocked:
                    continue
                if self.storage.insert_task(task, session_id):
                    return task
            summary.append(f"{ad.name}: only duplicate tasks")
        raise NoTaskAvailable("No new task could be discovered. " + " | ".join(summary))

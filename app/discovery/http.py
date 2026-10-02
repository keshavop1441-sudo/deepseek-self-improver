"""Mockable HTTP layer for public-source retrieval (stdlib only)."""
from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class FetchError(Exception):
    """Network/HTTP failure while retrieving a public source."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Response:
    url: str
    status: int
    text: str


class HttpClient(Protocol):
    def get(self, url: str, timeout: float | None = None, headers: dict | None = None) -> Response: ...


class UrllibHttp:
    def __init__(self, user_agent: str, timeout: float = 20.0):
        self.user_agent, self.timeout = user_agent, timeout

    def get(self, url: str, timeout: float | None = None, headers: dict | None = None) -> Response:
        h = {"User-Agent": self.user_agent, "Accept": "*/*"}
        h.update(headers or {})
        req = urllib.request.Request(url, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return Response(url, r.status, r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            raise FetchError(f"HTTP {e.code} from {url}", e.code) from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise FetchError(f"Cannot retrieve {url}: {e}") from e

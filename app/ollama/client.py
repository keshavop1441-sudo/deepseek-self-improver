"""Minimal Ollama HTTP client (stdlib only) using /api/chat with stream=false."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.config import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


class OllamaError(Exception):
    """Base error."""


class OllamaUnavailable(OllamaError):
    """Server cannot be reached."""


class ModelNotAvailable(OllamaError):
    """Server is up but the requested model is not pulled."""


class OllamaTimeout(OllamaError):
    """The request exceeded its time budget."""


class OllamaAborted(OllamaError):
    """The caller asked to abort (STOP / deadline)."""


class OllamaBadResponse(OllamaError):
    """Server answered with something we cannot use."""


# transport(method, url, payload, timeout) -> (http_status, parsed_json_or_text)
Transport = Callable[[str, str, Optional[dict], float], tuple[int, Any]]


def urllib_transport(method: str, url: str, payload: Optional[dict], timeout: float) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        status = e.code
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        if "timed out" in str(e).lower():
            raise OllamaTimeout(f"Timed out talking to Ollama at {url}") from e
        raise OllamaUnavailable(f"Cannot reach Ollama at {url}: {e}") from e
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


@dataclass
class ChatResult:
    content: str
    thinking: str
    raw: dict
    duration_s: float


class OllamaClient:
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, model: str = DEFAULT_MODEL,
                 transport: Transport = urllib_transport, request_timeout: float = 300.0,
                 num_ctx: int = 4096, num_predict: int = 3072, temperature: float = 0.2):
        self.host = host.rstrip("/")
        self.model = model
        self.transport = transport
        self.request_timeout = request_timeout
        self.num_ctx, self.num_predict, self.temperature = num_ctx, num_predict, temperature

    # -- health ------------------------------------------------------------
    def check_server(self, timeout: float = 5.0) -> str:
        """Return the server version or raise OllamaUnavailable."""
        status, body = self.transport("GET", f"{self.host}/api/version", None, timeout)
        if status != 200 or not isinstance(body, dict):
            raise OllamaUnavailable(f"Unexpected reply from {self.host}/api/version (HTTP {status})")
        return str(body.get("version", "unknown"))

    def list_models(self, timeout: float = 10.0) -> list[str]:
        status, body = self.transport("GET", f"{self.host}/api/tags", None, timeout)
        if status != 200 or not isinstance(body, dict):
            raise OllamaBadResponse(f"/api/tags returned HTTP {status}")
        return [str(m.get("name") or m.get("model")) for m in body.get("models", [])]

    def has_model(self, model: str | None = None) -> bool:
        want = model or self.model
        names = self.list_models()
        return any(n == want or (":" not in want and n == f"{want}:latest") for n in names)

    def ensure_ready(self) -> str:
        """Raise a descriptive error unless server+model are usable. Returns server version."""
        version = self.check_server()
        if not self.has_model():
            raise ModelNotAvailable(
                f"Model '{self.model}' is not installed in Ollama. Run: ollama pull {self.model}")
        return version

    # -- chat --------------------------------------------------------------
    def chat(self, messages: list[dict], think: bool = False, timeout: float | None = None,
             should_abort: Callable[[], bool] | None = None, temperature: float | None = None,
             seed: int | None = None, num_predict: int | None = None) -> ChatResult:
        """Single non-streaming chat call. `num_predict=None` uses the client's configured cap.

        Runs the transport in a worker thread so STOP/deadline (`should_abort`) and the hard
        `timeout` are honoured even while Ollama is mid-generation.
        """
        timeout = self.request_timeout if timeout is None else timeout
        if timeout <= 0:
            raise OllamaTimeout("No time budget left for this request")
        if num_predict is not None and num_predict <= 0:
            raise ValueError("num_predict must be positive")
        options: dict[str, Any] = {"temperature": self.temperature if temperature is None else temperature,
                                   "num_ctx": self.num_ctx,
                                   "num_predict": self.num_predict if num_predict is None else num_predict}
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                   "options": options}
        if think:
            payload["think"] = True
        started = time.monotonic()
        status, body = self._call_with_budget(payload, timeout, should_abort)
        if status == 400 and think and "think" in json.dumps(body).lower():
            payload.pop("think")  # model/server without thinking support: retry once without it
            remaining = timeout - (time.monotonic() - started)
            status, body = self._call_with_budget(payload, remaining, should_abort)
        if status == 404:
            raise ModelNotAvailable(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        if status != 200 or not isinstance(body, dict):
            raise OllamaBadResponse(f"/api/chat returned HTTP {status}: {str(body)[:300]}")
        msg = body.get("message") or {}
        if not isinstance(msg, dict) or "content" not in msg:
            raise OllamaBadResponse(f"/api/chat reply has no message.content: {str(body)[:300]}")
        return ChatResult(content=str(msg.get("content") or ""), thinking=str(msg.get("thinking") or ""),
                          raw=body, duration_s=time.monotonic() - started)

    def _call_with_budget(self, payload: dict, timeout: float,
                          should_abort: Callable[[], bool] | None) -> tuple[int, Any]:
        if timeout <= 0:
            raise OllamaTimeout("Time budget exhausted")
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                box["result"] = self.transport("POST", f"{self.host}/api/chat", payload, timeout)
            except BaseException as e:  # noqa: BLE001 - re-raised in caller thread
                box["error"] = e

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        end = time.monotonic() + timeout
        while t.is_alive():
            if should_abort and should_abort():
                raise OllamaAborted("Aborted while waiting for Ollama")
            if time.monotonic() >= end:
                raise OllamaTimeout(f"Ollama did not answer within {timeout:.0f}s")
            t.join(0.05)
        if "error" in box:
            raise box["error"]
        return box["result"]

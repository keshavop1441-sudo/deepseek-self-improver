import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.config import Config  # noqa: E402
from app.controller import Controller  # noqa: E402
from app.ollama.client import OllamaClient  # noqa: E402
from tests.helpers import FakeClock, FakeHttp, FakeOllama, StaticAdapter  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return Config(home=tmp_path, request_timeout=5.0)


@pytest.fixture
def make_controller(cfg):
    def _make(ollama=None, adapters=None, http=None, clock=None, **kw):
        ollama = ollama or FakeOllama()
        clock = clock or FakeClock()
        client = OllamaClient(cfg.ollama_host, cfg.model, transport=ollama, request_timeout=5.0)
        c = Controller(cfg, client=client, http=http or FakeHttp(), clock=clock,
                       sleep=lambda s: clock.advance(s), adapters=adapters or [StaticAdapter()], **kw)
        return c, ollama, clock
    return _make

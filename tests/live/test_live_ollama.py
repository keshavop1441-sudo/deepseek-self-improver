"""LIVE tests against a real local Ollama + deepseek-r1:1.5b.

NOT run in the normal suite (and never run in the cloud build). On your PC:

    set RUN_LIVE_OLLAMA=1            (Windows cmd)   |   export RUN_LIVE_OLLAMA=1
    python -m pytest tests/live -s
"""
import os

import pytest

from app.config import Config
from app.ollama.client import OllamaClient
from app.ollama.parsing import parse_model_output
from app.solver.prompts import SYSTEM_PROMPT

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_OLLAMA") != "1",
                                reason="live Ollama test: set RUN_LIVE_OLLAMA=1 on the machine running Ollama")


def test_real_deepseek_round_trip():
    cfg = Config()
    client = OllamaClient(cfg.ollama_host, cfg.model, request_timeout=600)
    client.ensure_ready()
    res = client.chat([{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": "What is the greatest common divisor of 12 and 18? "
                                                   "Put only the number in the `answer` field."}], think=True)
    parsed = parse_model_output(res.content, res.thinking)
    print("RAW:", res.content[:500])
    assert parsed.answer, "model returned no parsable answer"

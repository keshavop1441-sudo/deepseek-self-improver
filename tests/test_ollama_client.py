import pytest

from app.ollama.client import (ModelNotAvailable, OllamaAborted, OllamaBadResponse, OllamaClient, OllamaTimeout,
                               OllamaUnavailable)
from tests.helpers import FakeOllama, model_json


def client(fake, **kw):
    return OllamaClient("http://x", "deepseek-r1:1.5b", transport=fake, **kw)


def test_server_unavailable():
    with pytest.raises(OllamaUnavailable):
        client(FakeOllama(up=False)).check_server()


def test_model_missing_gives_pull_hint():
    c = client(FakeOllama(models=["llama3:8b"]))
    assert c.check_server() == "0.9.0"
    assert not c.has_model()
    with pytest.raises(ModelNotAvailable, match="ollama pull deepseek-r1:1.5b"):
        c.ensure_ready()


def test_chat_uses_api_chat_stream_false_and_think():
    fake = FakeOllama(lambda m, p: model_json("42"))
    res = client(fake).chat([{"role": "user", "content": "hi"}], think=True)
    p = fake.chat_calls[0]
    assert p["stream"] is False and p["think"] is True and p["model"] == "deepseek-r1:1.5b"
    assert "42" in res.content


def test_think_omitted_when_not_requested():
    fake = FakeOllama()
    client(fake).chat([{"role": "user", "content": "hi"}])
    assert "think" not in fake.chat_calls[0]


def test_retries_without_think_on_400():
    calls = []

    def transport(method, url, payload, timeout):
        if url.endswith("/api/chat"):
            calls.append(dict(payload))
            if payload.get("think"):
                return 400, {"error": "\"deepseek\" does not support thinking"}
            return 200, {"message": {"content": "ok"}}
        return 200, {}
    res = client(transport).chat([{"role": "user", "content": "x"}], think=True)
    assert res.content == "ok" and len(calls) == 2 and "think" not in calls[1]


def test_unknown_model_404():
    fake = FakeOllama(models=["other"])
    with pytest.raises(ModelNotAvailable):
        client(fake).chat([{"role": "user", "content": "x"}])


def test_bad_response_shape():
    def transport(m, u, p, t):
        return 200, {"unexpected": True}
    with pytest.raises(OllamaBadResponse):
        client(transport).chat([{"role": "user", "content": "x"}])


def test_http_500_is_bad_response():
    with pytest.raises(OllamaBadResponse):
        client(lambda m, u, p, t: (500, {"error": "boom"})).chat([{"role": "user", "content": "x"}])


def test_timeout_enforced():
    fake = FakeOllama(delay=1.0)
    with pytest.raises(OllamaTimeout):
        client(fake).chat([{"role": "user", "content": "x"}], timeout=0.2)


def test_abort_while_waiting():
    fake = FakeOllama(delay=1.0)
    with pytest.raises(OllamaAborted):
        client(fake).chat([{"role": "user", "content": "x"}], should_abort=lambda: True)


def test_zero_budget_is_timeout():
    with pytest.raises(OllamaTimeout):
        client(FakeOllama()).chat([{"role": "user", "content": "x"}], timeout=0)

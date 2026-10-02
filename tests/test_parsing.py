from app.ollama.parsing import parse_model_output
from tests.helpers import model_json


def test_clean_json():
    p = parse_model_output(model_json("5"))
    assert p.parse_ok and p.answer == "5" and p.self_check


def test_think_block_and_fence():
    raw = "<think>hmm {\"answer\": \"wrong\"}</think>\n```json\n" + model_json("7") + "\n```"
    p = parse_model_output(raw)
    assert p.parse_ok and p.answer == "7" and "hmm" in p.thinking


def test_trailing_comma_and_single_quotes():
    assert parse_model_output('{"answer": "3", "method": "m",}').answer == "3"
    assert parse_model_output("{'answer': '4'}").answer == "4"


def test_last_object_wins():
    raw = 'draft {"answer": "1"} final {"answer": "2", "method": "x"}'
    assert parse_model_output(raw).answer == "2"


def test_non_json_falls_back_but_flags():
    p = parse_model_output("Reasoning...\nFinal answer: 12.5")
    assert not p.parse_ok and p.answer == "12.5"


def test_garbage_never_raises():
    for raw in ["", "{{{", "no json at all", "<think>unterminated", None]:
        p = parse_model_output(raw)
        assert not p.parse_ok and p.answer == ""


def test_unterminated_think():
    p = parse_model_output("<think>still thinking about {\"answer\": 9}")
    assert p.answer == ""


def test_non_string_fields():
    p = parse_model_output('{"answer": 42, "calculations": "one", "assumptions": null}')
    assert p.answer == "42" and p.calculations == ["one"] and p.assumptions == []

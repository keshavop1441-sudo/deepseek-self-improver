import json

import pytest

from app.main import main
from app.memory.lessons import learn_from_outcome
from app.training import dataset
from app.training.hardware import detect_hardware
from app.training.lora import TrainingUnavailable, check_ready
from tests.helpers import FakeClock, FakeOllama, make_oracle, model_json


def test_prepare_training_only_verified_and_no_benchmark(make_controller, cfg):
    clock = FakeClock()
    fake = FakeOllama(make_oracle(wrong_first_gcd=True), on_chat=lambda n: clock.advance(10))
    c, _, _ = make_controller(fake, clock=clock)
    rep = c.run_session(1, run_benchmarks=False)
    # add an unverified (failed) attempt that must never be exported
    from app.tasks.model import Task
    bad = Task("numerical_reasoning", 1, "failed task", "u", "t", "numeric_computation", {"kind": "gcd", "a": 5, "b": 10})
    c.storage.insert_task(bad)
    aid = c.storage.insert_attempt(bad.task_id, 1, 0, "p", "r", None, {"answer": "99"}, True, "99", None, 1)
    from app.verifier.base import incorrect
    c.storage.insert_verification(aid, bad.task_id, incorrect("numeric_computation", "arithmetic", "no"))
    m = c.prepare_training()
    rows = [json.loads(l) for l in open(m["path"])]
    assert m["examples"] == len(rows) == rep["verified"]
    ids = {r["meta"]["task_id"] for r in rows}
    assert bad.task_id not in ids
    for r in rows:
        assert r["messages"][-1]["role"] == "assistant"
        a = json.loads(r["messages"][-1]["content"])
        assert a["self_check"] == dataset.SELF_CHECK_NOTE      # model's own self-assessment never trained on
        v = c.storage._one("SELECT verified FROM verifications WHERE verification_id=?", (r["meta"]["verification_id"],))
        assert v["verified"] == 1
    assert m["auto_finetune"] is False and "CPU-only" in m["note"]
    assert (c.config.training_dir / "manifest.json").exists()


def test_dataset_excludes_wrong_first_attempts(make_controller):
    clock = FakeClock()
    c, _, _ = make_controller(FakeOllama(make_oracle(), on_chat=lambda n: clock.advance(10)), clock=clock)
    c.run_session(1, run_benchmarks=False)
    answers = [json.loads(e["messages"][-1]["content"])["answer"] for e in dataset.build_examples(c.storage)]
    assert answers and "1000001" not in answers          # the wrong first attempts were not exported


def test_cpu_only_never_trains(tmp_path):
    assert detect_hardware().kind == "cpu"
    with pytest.raises(TrainingUnavailable, match="CPU-only"):
        check_ready(tmp_path / "train.jsonl", "deepseek-r1:1.5b", force_cpu=False)


def test_training_requires_optional_deps_and_data(tmp_path):
    with pytest.raises(TrainingUnavailable, match="PyTorch is not installed"):
        check_ready(tmp_path / "train.jsonl", "deepseek-r1:1.5b", force_cpu=True)


def test_normal_operation_does_not_import_torch():
    import subprocess, sys
    code = ("import sys; import app.main, app.controller, app.gui.presenter; "
            "bad=[m for m in ('torch','transformers','peft') if m in sys.modules]; sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=".").returncode == 0


# ---------------------------------------------------------------------- CLI
def test_cli_commands_with_shared_controller(make_controller, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_IMPROVER_HOME", str(tmp_path))
    clock = FakeClock()
    c, _, _ = make_controller(FakeOllama(make_oracle(), on_chat=lambda n: clock.advance(10)), clock=clock)
    assert main(["doctor"], c) in (0, 1)
    assert main(["run", "--minutes", "1", "--skip-benchmark"], c) == 0
    out = capsys.readouterr().out
    assert "verified=" in out and "Reports:" in out
    assert main(["stats"], c) == 0 and '"tasks_verified"' in capsys.readouterr().out
    assert main(["lessons"], c) == 0 and "pattern:" in capsys.readouterr().out
    assert main(["prepare-training"], c) == 0 and '"examples"' in capsys.readouterr().out
    assert main(["train-adapter"], c) == 3 and "Training not started" in capsys.readouterr().out
    assert main(["ping"], c) == 0


def test_cli_run_reports_preflight_error(make_controller, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_IMPROVER_HOME", str(tmp_path))
    c, _, _ = make_controller(FakeOllama(up=False))
    assert main(["run", "--minutes", "1"], c) == 2
    assert "not reachable" in capsys.readouterr().err
    assert main(["ping"], c) == 1

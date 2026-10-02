"""Optional LoRA/PEFT adapter training. Never required for normal operation.

* Torch/Transformers/PEFT are imported lazily, only here.
* Refuses on CPU-only hardware unless force_cpu=True (CPU fine-tuning is impractical and
  is never started automatically).
* Writes ONLY a new adapter directory under data/adapters/. The base model is loaded read-only
  and is never saved or overwritten; Ollama's model store is never touched.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.training.hardware import detect_hardware

# Ollama tag -> Hugging Face base (weights for training; Ollama's GGUF cannot be fine-tuned directly)
HF_BASES = {"deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
MIN_EXAMPLES = 50


class TrainingUnavailable(Exception):
    """Training cannot/should not run; the message says why and what to do."""


def check_ready(dataset: Path, model_tag: str, force_cpu: bool, min_examples: int = MIN_EXAMPLES) -> dict[str, Any]:
    hw = detect_hardware()
    if not hw.accelerated and not force_cpu:
        raise TrainingUnavailable(
            "CPU-only machine detected: adapter training is not started automatically. Keep collecting verified "
            "lessons (`prepare-training` keeps the dataset ready) and run `train-adapter` on a machine with a "
            "CUDA/XPU GPU, or pass --force-cpu if you really want a very slow CPU run.")
    if not hw.torch_available:
        raise TrainingUnavailable("PyTorch is not installed. Install the optional extras: "
                                  "pip install -r requirements-training.txt")
    if not dataset.exists():
        raise TrainingUnavailable(f"Dataset {dataset} not found. Run `python -m app.main prepare-training` first.")
    n = sum(1 for _ in dataset.open(encoding="utf-8"))
    if n < min_examples:
        raise TrainingUnavailable(f"Only {n} verified examples; at least {min_examples} are required.")
    if model_tag not in HF_BASES:
        raise TrainingUnavailable(f"No Hugging Face base mapping for '{model_tag}'. Pass --hf-base explicitly.")
    return {"hardware": hw.kind, "examples": n}


def train_adapter(dataset: Path, adapters_dir: Path, model_tag: str, hf_base: str | None = None,
                  epochs: int = 1, force_cpu: bool = False, lr: float = 2e-4, max_len: int = 1024) -> Path:  # pragma: no cover
    info = check_ready(dataset, model_tag, force_cpu)
    import torch  # noqa: WPS433
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = hf_base or HF_BASES[model_tag]
    out = adapters_dir / f"adapter_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if out.exists():
        raise TrainingUnavailable(f"{out} already exists; refusing to overwrite")
    device = {"cuda": "cuda", "xpu": "xpu", "mps": "mps"}.get(info["hardware"], "cpu")
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "v_proj"]))
    model.to(device)
    rows = [json.loads(l) for l in dataset.open(encoding="utf-8")]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for _ in range(epochs):
        for r in rows:
            text = tok.apply_chat_template(r["messages"], tokenize=False)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            loss = model(**enc, labels=enc["input_ids"]).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    out.mkdir(parents=True)
    model.save_pretrained(out)           # PEFT writes the adapter weights only, never the base model
    (out / "adapter_manifest.json").write_text(json.dumps({
        "base_model": base_id, "ollama_tag": model_tag, "examples": info["examples"], "epochs": epochs,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "note": "Adapter only. Base model unchanged. To use with Ollama, convert the adapter to GGUF and "
                "reference it with ADAPTER in a Modelfile (see README)."}, indent=2), encoding="utf-8")
    return out

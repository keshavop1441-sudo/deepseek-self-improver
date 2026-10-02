"""Detect whether an accelerator is available. Importing this never requires PyTorch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareInfo:
    kind: str            # cpu | cuda | xpu | mps
    torch_available: bool
    detail: str

    @property
    def accelerated(self) -> bool:
        return self.kind != "cpu"


def detect_hardware() -> HardwareInfo:
    try:
        import torch  # type: ignore  # noqa: WPS433
    except ImportError:
        return HardwareInfo("cpu", False, "PyTorch not installed (not needed for normal operation)")
    try:
        if torch.cuda.is_available():
            return HardwareInfo("cuda", True, torch.cuda.get_device_name(0))
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return HardwareInfo("xpu", True, "Intel XPU")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return HardwareInfo("mps", True, "Apple MPS")
    except Exception as e:  # noqa: BLE001
        return HardwareInfo("cpu", True, f"accelerator probe failed: {e}")
    return HardwareInfo("cpu", True, "PyTorch installed but no CUDA/XPU/MPS accelerator found")

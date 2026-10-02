"""Central configuration. All paths are relative to the project home."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "deepseek-r1:1.5b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
ALLOWED_DURATIONS = (2, 20, 30, 45)
MAX_RETRIES = 3

DOMAINS = (
    "mathematics",
    "probability",
    "statistics",
    "logic",
    "quant_finance",
    "financial_calculation",
    "numerical_reasoning",
    "coding",
)

# Benchmark categories (spec): numerical, mathematics, statistics, finance, logic, coding
DOMAIN_TO_CATEGORY = {
    "mathematics": "mathematics",
    "probability": "statistics",
    "statistics": "statistics",
    "logic": "logic",
    "quant_finance": "finance",
    "financial_calculation": "finance",
    "numerical_reasoning": "numerical",
    "coding": "coding",
}


def project_home() -> Path:
    env = os.environ.get("SELF_IMPROVER_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    home: Path = field(default_factory=project_home)
    model: str = field(default_factory=lambda: os.environ.get("SELF_IMPROVER_MODEL", DEFAULT_MODEL))
    ollama_host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST_URL", DEFAULT_OLLAMA_HOST)
    )
    # CPU inference of a 1.5B reasoning model is slow: be generous per request.
    request_timeout: float = 300.0
    num_ctx: int = 4096
    num_predict: int = 3072
    temperature: float = 0.2
    max_retries: int = MAX_RETRIES
    lessons_per_prompt: int = 3
    http_timeout: float = 20.0
    user_agent: str = "deepseek-self-improver/1.0 (local research tool; contact: set SELF_IMPROVER_CONTACT)"
    code_timeout: float = 10.0
    max_consecutive_discovery_failures: int = 8
    # Benchmark settings
    benchmark_quick_size: int = 12

    def __post_init__(self) -> None:
        self.home = Path(self.home)
        contact = os.environ.get("SELF_IMPROVER_CONTACT")
        if contact:
            self.user_agent = f"deepseek-self-improver/1.0 (local research tool; contact: {contact})"

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "self_improver.db"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training"

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.reports_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

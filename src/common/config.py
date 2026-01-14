from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunConfig:
    input_csv: Path
    out_dir: Path
    model: str = "gpt-4o-mini"
    genai_enabled: bool = True
    guardrails: bool = True
    seed: int = 42
    n_genai_reliability_runs: int = 5

"""Select fresh inference and a local evidence directory for live gates."""
from __future__ import annotations

from pathlib import Path

from simulation.source import SimulationConfigOverrides, copy_simulation_config


def live_model_config(source: Path, evidence: Path, prompt_folder: str = "prompts") -> Path:
    """Copy a model setup into this gate's private evidence directory."""
    path = evidence / "model-config.yaml"
    copy_simulation_config(source, path, SimulationConfigOverrides(prompt_directory=evidence / prompt_folder))
    return path

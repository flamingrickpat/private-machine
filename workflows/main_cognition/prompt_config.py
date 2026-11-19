from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field
from pm.context_render.story import ShardLabels
from pm.runs.role_io import load_yaml

class PhasePrompts(BaseModel):
    """The prompt texts of one phase, read from its YAML file.

    Every phase has ``system_prompt`` and ``user_template``. The other fields
    exist only in the files of the phases that use them.
    """

    system_prompt: str = ""
    user_template: str = ""
    repair_template: str = ""
    job_template: str = ""
    reality_labels: dict[str, str] = Field(default_factory=dict)
    reality_rules: dict[str, str] = Field(default_factory=dict)
    # The shared story prefix (story.yaml) and the per-call parts of the codelets phase.
    story_template: str = ""
    story_ack: str = ""
    shard_template: str = ""
    shard_labels: ShardLabels = Field(default_factory=ShardLabels)
    # The attention phase: the mental aperture texts of the old code.
    aperture_narrow: str = ""
    aperture_broad: str = ""
    aperture_default: str = ""
    view_template: str = ""
    conversion_template: str = ""
    selector_template: str = ""
    echo_template: str = ""
    # The appraise phase: the expectation check. The route phase: the anticipation call.
    check_template: str = ""
    anticipation_template: str = ""
    # The goals and decide phases: the regulation mode in story words, by mode.
    regulation_words: dict[str, str] = Field(default_factory=dict)
    # The decide and route phases: a stimulus group in story words, by group kind.
    part_entity: str = ""
    part_workflow: str = ""
    part_self: str = ""
    # The route phase: the routed action in words, by action type.
    action_words: dict[str, str] = Field(default_factory=dict)


def load_prompts(path: str | Path) -> PhasePrompts:
    """Read and validate one prompt file. A missing or invalid file raises."""
    return load_yaml(path, PhasePrompts)

from __future__ import annotations

import json
from pm.context_render.budget import TokenEstimator, evidence_excerpt
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, JsonValue
from pm.context_render.prompt import RenderSettings, RenderedPrompt, require_rendered_prompt
from pm.context_render.turns import Turn, Role


class MinecraftActionTool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    description: str | None = None
    parameters: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class MinecraftActionRenderInput:
    goal: str
    observation: str
    tools: tuple[MinecraftActionTool, ...]


def render_minecraft_action(input: MinecraftActionRenderInput, settings: RenderSettings) -> RenderedPrompt:
    tool_lines: list[str] = []
    for tool in input.tools:
        tool_lines.append(f"{tool.name}: {tool.description or ''}\nArguments: " + json.dumps(tool.parameters, ensure_ascii=False))
    system = ("Choose exactly one Minecraft action from the allowed tool list. "
              "Reply with only the tool name and a JSON argument object on one line: <tool> <json>.")
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    retained = max(len(input.goal), len(input.observation))
    while True:
        goal = evidence_excerpt(input.goal, retained)
        observation = evidence_excerpt(input.observation, retained)
        user = f"Goal: {goal}\nAllowed tools:\n" + "\n\n".join(tool_lines) + f"\nObservation:\n{observation}\nChoose one action."
        turns = (Turn(Role.System, system), Turn(Role.User, user))
        if estimator.estimate(turns) <= settings.input_ceiling or retained == 0:
            return require_rendered_prompt("Minecraft action choice", turns, settings)
        retained //= 2

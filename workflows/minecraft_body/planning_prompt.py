from __future__ import annotations

import json
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
    user = f"Goal: {input.goal}\nAllowed tools:\n" + "\n\n".join(tool_lines) + f"\nObservation:\n{input.observation}\nChoose one action."
    return require_rendered_prompt("Minecraft action choice", (Turn(Role.System, system), Turn(Role.User, user)), settings)

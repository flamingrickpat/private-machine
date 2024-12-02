import json

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


class DecideModelMode(BaseModel):
    """Decide what model mode to use next. Assistant or story mode."""
    reason: str = Field(description="Few words why you chose your option.")
    story_mode: bool = Field(description="True for story mode, false for assistant mode")

    def execute(self, s):
        s["story_mode"] = self.story_mode


def assistant_story_check(current_chat: str) -> bool:
    schema = json.dumps(DecideModelMode.model_json_schema())

    # add system prompt
    messages = [(
        "system",
        "Based on the last few messages of this chat between the user and the AI, "
        "decide if the AI should switch to assistant or story mode for the NEXT message!"
        "Assistant mode is for API calls and story mode for natural conversation like between two people."
        f"Decide by creating a valid JSON string, and ONLY a valid JSON string. This is your schema:\n"
        f"{schema}"
    ), (
        "user",
        f"This is the current chat:"
        f"\n### BEGIN CHAT\n"
        f"{current_chat}"
        f"\n### END CHAT\n"
        f"Decide by creating a valid JSON string, and ONLY a valid JSON string!"
    )]

    state = {}
    _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=128, temperature=0.1), tools=[DecideModelMode])
    for call in calls:
        call.execute(state)

    return state["story_mode"]
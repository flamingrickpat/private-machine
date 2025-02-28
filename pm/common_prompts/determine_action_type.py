from typing import List

from pydantic import BaseModel, Field

from pm.character import companion_name, char_card_3rd_person_emotional
from pm.controller import controller
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.system_classes import ActionType
from pm.tools.common import tool_docs
from pm.utils.enum_utils import make_enum_subset


class ActionSelector(BaseModel):
    thoughts: str = Field(description="pros and cons for different selections")
    reason: str = Field(description="why you chose this action type?")
    action_type: ActionType = Field(description="the action type")

def determine_action_type(block, allowed_actions: List[ActionType]) -> ActionSelector:
    enum_subset = make_enum_subset(ActionType, allowed_actions)

    class WorkingActionSelector(BaseModel):
        thoughts: str = Field(description="pros and cons for different selections")
        action_type: enum_subset = Field(description="the action type")
        reason: str = Field(description="why you chose this action type?")

    msgs = [("system", f"You are a helpful assistant that determines what action the AI companion {companion_name} should do in this endearing story."
                       f"### BEGIN CHARACTER DESCRIPTION"
                       f"\n{char_card_3rd_person_emotional}\n"
                       f"### END CHARACTER DESCRIPTION"
                       f"\nThe following tools are available to {companion_name}:\n"
                       f"### BEGIN TOOLS"
                       f"\n{tool_docs}\n"
                       f"### END TOOLS"),
            ("user", f"Hi, I'm trying to write a story about a user and his AI companion. I want the AI to be really smart and self-aware and use tools and API calls wisely. This is the latest "
                     f"dialogue I have: \n"
                     f"### BEGIN MESSAGES"
                     f"\n{block}\n"
                     f"### END MESSAGES\n"
                     f"What action should {companion_name} choose? Make sure to include her internal thoughts in your decision! And don't forget about tools (API calls) she could use! But remember,"
                     f"this function might be called recursively. So if a tool result comes in, it might be good that she reports to the user or continues her usual stuff (if she came up with the "
                     f"API call herself)")
            ]

    _, calls = controller.completion_tool(LlmPreset.Default, msgs, comp_settings=CommonCompSettings(temperature=0, repeat_penalty=1, max_tokens=1024),
                                          tools=[WorkingActionSelector])
    return calls[0]
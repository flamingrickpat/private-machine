from typing import Optional, Any, List, Tuple

from pydantic import BaseModel, Field

from pm.controller import controller, log_conversation
from pm.character import sysprompt, database_uri, cluster_split, companion_name, timestamp_format, sysprompt_addendum, char_card_3rd_person_neutral, user_name, char_card_3rd_person_emotional
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.system_classes import ActionType
from pm.utils.enum_utils import make_enum_subset
from pm.tools.common import tool_docs

class ActionSelector(BaseModel):
    thoughts: str = Field(description="pros and cons for different selections")
    action_type: ActionType = Field(description="the action type")
    reason: str = Field(description="why you chose this action type?")

def determine_action_type(block, allowed_actions: List[ActionType]) -> ActionSelector:
    enum_subset = make_enum_subset(ActionType, allowed_actions)

    class WorkingActionSelector(BaseModel):
        thoughts: str = Field(description="pros and cons for different selections")
        action_type: enum_subset = Field(description="the action type")
        reason: str = Field(description="why you chose this action type?")

    msgs = [("system", f"You are a helpful assistant that determines what action the AI companion {companion_name} should do."
                       f"### BEGIN CHARACTER DESCRIPTION"
                       f"\n{char_card_3rd_person_emotional}\n"
                       f"### END CHARACTER DESCRIPTION"
                       f"\nThe following tools are available to {companion_name}:\n"
                       f"### BEGIN TOOLS"
                       f"\n{tool_docs}\n"
                       f"### END TOOLS"),
            ("user", f"### BEGIN MESSAGES"
                     f"{block}"
                     f"### END MESSAGES"
                     f"What action should {companion_name} choose? Make sure to include her internal thoughts in your decision! And don't forget about tools (API calls) she could use!")
            ]

    _, calls = controller.completion_tool(LlmPreset.Default, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024, presence_penalty=1, frequency_penalty=1),
                                          tools=[WorkingActionSelector])
    return calls[0]
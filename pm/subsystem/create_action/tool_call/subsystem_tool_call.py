from datetime import datetime
from typing import List

from pm.character import companion_name
from pm.common_prompts.determine_tool import determine_tools
from pm.common_prompts.get_tool_call import get_tool_call
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse, ActionType
from pm.system_utils import get_recent_messages_block
from pm.tools.common import tools_list_str


class SubsystemToolCall(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Tool Calling"

    def get_subsystem_description(self) -> str:
        return f"The 'Tool Calling' subsystem handles calls to APIs such as for smart homes and other IoT-devices the user has connected. The following tools are available: {tools_list_str}"

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.CreateAction]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.UserInput, ImpulseType.ToolResult, ImpulseType.Thought]

    def get_action_types(self):
        return [ActionType.ToolCall]

    def proces_state(self, state: GhostState):
        ctx = get_recent_messages_block(6)
        tool_type = determine_tools(ctx)

        res = {}
        tool = get_tool_call(ctx, tool_type)
        tool.execute(res)

        content = f"{companion_name} uses her AI powers to call the tool. The tool response: {res['output']}"

        action_event = Event(
            source="system",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ConsciousSubsystem.value,
            turn_story="user",
            turn_assistant="assistant")
        update_database_item(action_event)

        state.output = Impulse(is_input=False, impulse_type=ImpulseType.ToolResult, endpoint="self", payload=content)
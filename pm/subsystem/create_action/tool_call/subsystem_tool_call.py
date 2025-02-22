from datetime import datetime
from typing import List

from pm.character import companion_name
from pm.common_prompts.determine_tool import determine_tools
from pm.common_prompts.get_tool_call import get_tool_call
from pm.controller import controller
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse
from pm.system_utils import get_recent_messages_block


class SubsystemToolCall(SubsystemBase):
    def get_subsystem_name(self) -> str:
        pass

    def get_subsystem_description(self) -> str:
        pass

    def get_pipeline_state(self) -> List[PipelineStage]:
        pass

    def get_impulse_type(self) -> List[ImpulseType]:
        pass

    def get_action_types(self):
        pass

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
            interlocus=InterlocusType.SystemMessage.value
        )
        


        state.output = Impulse(is_input=False, impulse_type=ImpulseType.ToolResult, endpoint="self", payload=content)
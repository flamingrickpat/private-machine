from datetime import datetime
from typing import List

from pm.common_prompts.determine_action_type import determine_action_type
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage, GhostAction
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, ActionType
from pm.system_utils import get_recent_messages_block


def get_allowed_actions_for_impulse(st: ImpulseType):
    if st == ImpulseType.UserInput:
        return [ActionType.Reply, ActionType.ToolCall]
    elif st == ImpulseType.ToolResult:
        return [ActionType.Reply, ActionType.ToolCall]
    elif st == ImpulseType.Thought:
        return [ActionType.InitiateUserConversation, ActionType.InitiateInternalContemplation, ActionType.ToolCall]
    else:
        raise Exception("not allowed as input")


class SubsystemActionSelection(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Action Selection"

    def get_subsystem_description(self) -> str:
        return "The action selection subsystem is the executive function of the Ghost."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.ChooseAction]

    def get_action_types(self) -> List[ActionType]:
        return [ActionType.All]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.All]

    def proces_state(self, state: GhostState):
        allowed_actions = get_allowed_actions_for_impulse(state.sensation.impulse_type)

        block = get_recent_messages_block(16, True)
        action_selection = determine_action_type(block, allowed_actions)
        content = None
        action_type = ActionType[str(action_selection.action_type.value)]
        if action_type == ActionType.InitiateUserConversation:
            content = f"I will message my user because of the following reasons: {action_selection.reason}"
        elif action_type == ActionType.InitiateIdleMode:
            content = f"I will stay idle until the next heartbeat because of the following reasons: {action_selection.reason}"
        elif action_type == ActionType.InitiateInternalContemplation:
            content = f"I will contemplate for the time being because of the following reasons: {action_selection.reason}"
        elif action_type == ActionType.Reply:
            content = f"I will now reply to the user."
        elif action_type == ActionType.Ignore:
            content = f"I will ignore this input because: {action_selection.reason}"
        elif action_type == ActionType.ToolCall:
            content = f"I will use my AI powers to make an API call: {action_selection.reason}"

        action_event = Event(
            source=self.get_subsystem_name(),
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ActionDecision.value
        )

        update_database_item(action_event)

        state.action = GhostAction(
            action_type=action_type,
            event=action_event
        )


from datetime import datetime
from typing import List

from pm.character import companion_name
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.create_action.user_reply.user_reply_prompting import completion_story_mode
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse, ActionType


class SubsystemUserReply(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Verbal Communication"

    def get_subsystem_description(self) -> str:
        return "The verbal communication subsystem is responsible for converting the states of the AI companion and its subsystems into coherent dialog to communicate with the world."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.CreateAction]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.UserInput]

    def get_action_types(self):
        return [ActionType.Reply]

    def proces_state(self, state: GhostState):
        content = completion_story_mode()

        action_event = Event(
            source=companion_name,
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Public.value
        )
        update_database_item(action_event)

        state.output = Impulse(is_input=False, impulse_type=ImpulseType.UserOutput, endpoint="user", payload=content)
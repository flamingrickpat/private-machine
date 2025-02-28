from datetime import datetime
from typing import List

from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import Event, InterlocusType
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType


class SubsystemImpulseToEvent(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Sensation Handler"

    def get_subsystem_description(self) -> str:
        return "The Sensation Handler Subsystem saves raw sensations to the memory."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.AddImpulse]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.All]

    def get_action_types(self):
        pass

    def proces_state(self, state: GhostState):
        if state.sensation.impulse_type == ImpulseType.UserInput:
            content = state.sensation.payload
            event = Event(
                source=f"{state.sensation.endpoint}",
                content=content,
                embedding=controller.get_embedding(content),
                token=get_token(content),
                timestamp=datetime.now(),
                interlocus=InterlocusType.Public.value,
                turn_story="assistant",
                turn_assistant="user"
            )
            update_database_item(event)
            state.buffer_add_impulse.append(event)
        elif state.sensation.impulse_type == ImpulseType.SystemMessage:
            content = state.sensation.payload
            event = Event(
                source=f"{state.sensation.endpoint}",
                content=content,
                embedding=controller.get_embedding(content),
                token=get_token(content),
                timestamp=datetime.now(),
                interlocus=InterlocusType.Public.value,
                turn_story="system",
                turn_assistant="user"
            )
            update_database_item(event)
            state.buffer_add_impulse.append(event)


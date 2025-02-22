from datetime import datetime
from typing import List

from pm.character import companion_name
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.sensation_evaluation.emotion.emotion_schema import execute_agent_group_emotion
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse


class SubsystemEmotion(SubsystemBase):
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
        content = execute_agent_group_emotion()
        event = Event(
            source=f"{companion_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Public.value
        )
        update_database_item(event)

        state.output = Impulse(is_input=False, impulse_type=ImpulseType.UserOutput, endpoint="user", payload=content)
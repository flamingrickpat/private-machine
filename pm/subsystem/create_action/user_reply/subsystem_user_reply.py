from datetime import datetime
from typing import List

from pm.actions import generate_action_selection
from pm.character import companion_name
from pm.common_prompts.determine_action_type import determine_action_type
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage, GhostAction
from pm.subsystem.subsystem_base import SubsystemBase
from pm.subsystem.user_reply.user_reply_prompting import completion_story_mode
from pm.system_classes import ImpulseType, ActionType, Impulse
from pm.system_utils import get_recent_messages_block


class SubSystemUserReply(SubsystemBase):
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
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
from pm.system_utils import get_recent_messages_block


class SubsystemEmotion(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Emotion"

    def get_subsystem_description(self) -> str:
        return "The emotion subsystem handles low-level emotional states such as joy, happiness and try to keep the AI companion in a good mood."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.EvalImpulse]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.Thought, ImpulseType.UserInput, ImpulseType.ToolResult]

    def get_action_types(self):
        pass

    def proces_state(self, state: GhostState):
        ctx = get_recent_messages_block(16, internal=True)
        lm = get_recent_messages_block(1)

        content = execute_agent_group_emotion(ctx, lm)
        event = Event(
            source=f"{self.get_subsystem_name()}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ConsciousSubsystem.value,
            turn_story="user",
            turn_assistant="assistant")
        update_database_item(event)
        state.buffer_sensation_evaluation.append(event)
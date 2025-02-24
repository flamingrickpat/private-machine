from datetime import datetime
from typing import List

from pm.character import companion_name
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.common_prompts.rate_emotional_state import rate_emotional_state
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.ghost.mental_state import base_model_to_db
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
        ctx = get_recent_messages_block(16, internal=True, max_tick=state.prev_tick_id if state.prev_tick_id is not None else -1)
        lm = get_recent_messages_block(1, max_tick=state.prev_tick_id if state.prev_tick_id is not None else -1)
        old_emotional_state = rate_emotional_state(ctx, lm)

        ctx = get_recent_messages_block(16, internal=True)
        lm = get_recent_messages_block(1)
        new_emotional_state = rate_emotional_state(ctx, lm)

        emotional_impact = rate_emotional_impact(ctx, lm)
        emotional_delta = old_emotional_state.get_delta(new_emotional_state, emotional_impact)

        emotional_state = state.emotional_state + emotional_delta

        base_model_to_db(state.tick_id, emotional_state)
        state.emotional_state = emotional_state

        content = execute_agent_group_emotion(ctx, lm, emotional_state=emotional_state)

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
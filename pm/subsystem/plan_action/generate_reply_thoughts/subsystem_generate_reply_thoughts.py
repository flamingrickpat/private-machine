import random
from typing import List

from pm.character import companion_name
from pm.common_prompts.rewrite_cot_as_story_instruction import rewrite_cot_as_story_instruction
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import Event, InterlocusType
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.plan_action.generate_reply_thoughts.generate_reply_thoughts_tree import generate_tree_of_thoughts_str
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, ActionType
from pm.system_utils import get_recent_messages_block, get_facts_block, get_recent_user_messages_block, get_public_event_count


class SubsystemGenerateReplyThoughts(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Enhanced Reply and Self-Reflection Subsystem"

    def get_subsystem_description(self) -> str:
        return "This subsystem handles higher-order concepts in conversation and ensures the companion chooses interesting, diverse and reflective responses."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.PlanAction]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.All]

    def get_action_types(self):
        return [ActionType.Reply, ActionType.InitiateUserConversation]

    def proces_state(self, state: GhostState):
        if get_public_event_count() < 32:
            return

        # 15% chance to execute tot
        chance = 0.15
        if random.uniform(0, 1) > chance:
            return

        ctx = get_recent_messages_block(32, internal=True)
        lum = get_recent_user_messages_block(1)
        k = get_facts_block(64, get_recent_messages_block(4))

        max_depth = 3 # thought, critic, thought
        thought = generate_tree_of_thoughts_str(ctx, lum,k, max_depth)
        thought = f"I have some ideas how the dialogue can unfold.\n### BEGIN IDEAS\n'{thought}'\n### END IDEAS\nPlease integrate those ideas, reflections and example dialog in your story!"
        #thought = rewrite_cot_as_story_instruction(thought)
        controller.cache_tot = thought

        action_event = Event(
            source=f"{self.get_subsystem_name()}",
            content=thought,
            embedding=controller.get_embedding(thought),
            token=get_token(thought),
            timestamp=controller.get_timestamp(),
            interlocus=InterlocusType.ConsciousSubsystem,
            turn_story="user",
            turn_assistant="assistant")

        update_database_item(action_event)
        state.buffer_plan_action.append(action_event)

        return action_event



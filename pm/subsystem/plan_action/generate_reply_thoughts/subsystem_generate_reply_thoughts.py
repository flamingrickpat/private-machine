from typing import List

from pm.character import companion_name
from pm.controller import controller
from pm.database.tables import Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.plan_action.generate_reply_thoughts.generate_reply_thoughts_tree import generate_tree_of_thoughts_str
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType
from pm.system_utils import get_recent_messages_block, get_facts_block


class SubsystemGenerateReplyThoughts(SubsystemBase):
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
        max_depth = 4
        ctx = get_recent_messages_block(24, internal=True)
        k = get_facts_block(64, get_recent_messages_block(4))
        thought = generate_tree_of_thoughts_str(ctx, k, max_depth)
        # thought = rewrite_as_thought(thought)
        controller.cache_tot = thought

        event = Event(
            source=f"{companion_name}",
            content=thought,
            embedding=controller.get_embedding(thought),
            token=get_token(thought),
            timestamp=controller.get_timestamp(),
            interlocus=-1
        )

        return event



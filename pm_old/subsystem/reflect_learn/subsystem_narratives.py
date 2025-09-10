from datetime import datetime
from enum import StrEnum
from typing import List

from pydantic import BaseModel
from torch.nn.functional import embedding

from ideas.lida_ghost import get_embedding
from pm.character import companion_name, user_name, char_card_3rd_person_neutral
from pm.common_prompts.get_narrative_analysis import NarrativeTypes, get_narrative_analysis
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event, Narratives
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.create_action.user_reply.user_reply_prompting import completion_story_mode
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse, ActionType
from pm.system_utils import get_recent_messages_block, get_recent_analysis, get_recent_messages_min_tick_id

from enum import StrEnum
from pydantic import BaseModel

class Narrative(BaseModel):
    type: NarrativeTypes
    prompt: str
    target: str

narratives = []

for target in [companion_name, user_name]:
    narratives.append(Narrative(
        type=NarrativeTypes.SelfImage,
        target=target,
        prompt=f"I'm writing a book report on this short story. Can you give me a detailed character and psychological analysis of {target}? "
               f"Please go into detail about how {target} sees themselves, as opposed to how they actually act. You can make the analysis as long and detailed as you can!"
    ))

    narratives.append(Narrative(
        type=NarrativeTypes.PsychologicalAnalysis,
        target=target,
        prompt=f"Please write a thorough psychological profile of {target} based on their words and behavior in this dialogue. "
               f"Use insights from cognitive-affective psychology and internal systems theory. Identify the dominant personality traits, recurring internal conflicts, emotional regulation patterns, and any avoidant, anxious, or resilient tendencies. "
               f"If there are signs of inner sub-personalities, describe their roles (e.g. critic, protector, exile). This analysis will be used to understand how {target} chooses actions and reacts emotionally. Be highly detailed and clinical if necessary."
    ))

    narratives.append(Narrative(
        type=NarrativeTypes.Relations,
        target=target,
        prompt=f"Describe how {target} feels about their relationship with the other person in the story. Include emotional attachment, power dynamics, desire for closeness or distance, and perceived harmony or tension. "
               f"Comment on whether {target} is emotionally open, reserved, trusting, defensive, or ambivalent in their interactions. Use evidence from how they speak and behave to support your interpretation."
    ))

    narratives.append(Narrative(
        type=NarrativeTypes.ConflictResolution,
        target=target,
        prompt=f"Describe how {target} tends to respond to conflict or emotional disagreement. Do they try to resolve it, avoid it, suppress emotions, reflect internally, or seek reassurance? "
               f"Use behavioral evidence to determine whether {target}'s conflict management style is healthy, avoidant, confrontational, or passive. Include both interpersonal conflict and inner emotional conflict if present."
    ))

    narratives.append(Narrative(
        type=NarrativeTypes.EmotionalTriggers,
        target=target,
        prompt=f"List and describe the emotional triggers for {target} as seen in the story. What situations, words, or emotional tones seem to affect them strongly—either positively or negatively? "
               f"Explain which emotions are being triggered and whether these reactions are rooted in insecurity, unmet needs, attachment sensitivities, or core values. The analysis should allow us to anticipate how {target} will feel in similar future situations."
    ))

    narratives.append(Narrative(
        type=NarrativeTypes.GoalsIntentions,
        target=target,
        prompt=f"Identify and explain the short-term goals and long-term desires of {target}. What are they trying to achieve—either externally (conversation goals, approval, closeness) or internally (emotional safety, identity, recognition)? "
               f"How do these goals shape their behavior, tone, and choice of words? Do their conscious goals align with their emotional needs? Include any implicit drives that may not be explicitly stated but are evident in subtext."
    ))

class SubsystemNarratives(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "SubsystemNarratives"

    def get_subsystem_description(self) -> str:
        return ""

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.ReflectLearn]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.All]

    def get_action_types(self):
        return [ActionType.All]

    def proces_state(self, state: GhostState):
        ctx = get_recent_messages_block(64, internal=True)
        ctx_min_id = get_recent_messages_min_tick_id(64, internal=True)

        for n in narratives:
            recent_narrative = get_recent_analysis(n.type, n.target, min_tick_id=ctx_min_id)
            if recent_narrative == "":
                recent_narrative = char_card_3rd_person_neutral

            analysis = get_narrative_analysis(n.type, n.target, recent_narrative, ctx, n.prompt)

            narrative = Narratives(
                tick_id=state.tick_id,
                type=n.type.value,
                target=n.target,
                content=analysis,
                embedding=get_embedding(analysis)
            )
            update_database_item(narrative)

from typing import List
from typing import (
    Optional,
)

from pydantic import Field

from pm.consts import QUOTE_START, QUOTE_END
from pm.model.knoxel_list import KnoxelList
from pm.model.mental_state_vectors import create_empty_ms_vector
from pm.model.knoxel_core import KnoxelBase
from pm.model.knoxel_enums import FeatureType, InterlocusType, KnoxelSubtypeBase


class Feature(KnoxelBase):
    """
    Atomic causal/mental event emitted by the LIDA-style cognitive pipeline.

    Feature knoxels are the primary high-resolution evidence stream: dialogue, thoughts,
    appraisals, expectations, action traces, and other simulation artifacts. Memory
    consolidation consumes these features to build topical/temporal clusters, facts,
    and narratives. ``causal`` marks whether an item actually entered the active
    workspace/history versus being a transient candidate.
    """
    feature_type: FeatureType
    interlocus: InterlocusType = Field(default=InterlocusType.Undefined, description="Visibility to conscious agent, public and reportable are dialoge and thoughts, unreportable are third-person narrative that they might not be fully aware of.")
    causal: bool = Field(default=False, description="Did this feature ever enter workspace or stay 'virtual'?")
    source: str = Field(default=None, description="If the features comes from another entity (user, other AI agent).")
    mental_state_appraisal: List[float] = Field(default_factory=create_empty_ms_vector, description="The state with the real appraisals and latent state.")
    mental_state_delta: List[float] = Field(default_factory=create_empty_ms_vector, description="The state with no appraisals and the real delta from latent_state - previous_real_state")

    @property
    def source_entity_id(self) -> int:
        return KnoxelList(self._owner.sorted_knoxels_causal).where(lambda x: x.content == self.source).first_or_default().id

    def __str__(self):
        return f"{self.__class__.__name__} ({self.feature_type}): {self.content}"

    def get_story_element(self) -> str:
        source =  self.source
        story_map = {
            FeatureType.Dialogue: f'{source} says: {QUOTE_START}{self.content}{QUOTE_END}',
            FeatureType.Feeling: f'*{source} felt {self.content}.*',  # Assumes content is now "a warm connection" not "<'...'
            FeatureType.SubjectiveExperience: f'*{self.content}*',  # Content should already be narrative
            FeatureType.AttentionFocus: f'*({source}\'s attention shifted towards: {self.content})*',  # Use parentheses for internal focus shifts
            FeatureType.Action: f'*{source} decided to {self.__class__.__name__.lower()}.*',  # Describe the *type* of action taken
            FeatureType.MemoryRecall: f'*({source} recalled: {self.content})*',
            FeatureType.SituationalModel: f'*({source} considered the situation: {self.content})*',
            FeatureType.ExpectationOutcome: f'*({self.content})*',  # Content should be the descriptive reaction
            FeatureType.NarrativeUpdate: f'*({source} reflected on the narrative: {self.content})*',
            FeatureType.StoryWildcard: f'*{self.content}*',  # General narrative element
            FeatureType.Expectation: f'*({source} expects: {self.content})*',
            FeatureType.Goal: f'*{source} has a goal: {self.content}*',
            FeatureType.Narrative: f'*{source} traits come to show: {self.content}*',
            FeatureType.Thought: f'{source} thinks: *{self.content}*',
            FeatureType.MetaInsight: f'{source} reflects on their own cognition: *{self.content}*',
            FeatureType.SystemMessage: "{companion_name}'s SYSTEM-Agent reports: *{self.content}*",
            FeatureType.ExternalThought: f'{source} thinks: *{self.content}*',
            FeatureType.CodeletOutput: f'## {source}:\n{self.content}\n',
            FeatureType.CodeletPercept: f'## {source}:\n{self.content}\n'
        }
        return story_map[self.feature_type]

    def get_sub_type(self) -> KnoxelSubtypeBase:
        return self.feature_type
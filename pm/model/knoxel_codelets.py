from typing import List

from pydantic import BaseModel, Field

from pm.model.knoxel_core import KnoxelBase


class AffectMeta(BaseModel):
    """
    Lightweight affect annotation used when scoring knoxel relevance in a procedure.

    This is not a full persistent mental-state object; it is a local weighting tag for
    attention/selection steps where valence, salience, arousal, and dominance influence
    prioritization.
    """
    valence: float = Field(default=0, le=1, ge=-1, description="Signed value: negative = aversive, positive = appetitive")
    salience: float = Field(default=0, le=1, ge=0, description="Attention capture strength")
    arousal: float = Field(default=0, le=1, ge=-1, description="Activation/urgency proxy")
    dominance: float = Field(default=0, le=1, ge=-1, description="Control/agency proxy")


class PerceptCoalition(BaseModel):
    """
    Procedure-local perception bundle over existing knoxels for the current situation.

    Percepts are intentionally cheap and ephemeral: multiple subsystems can create
    different Percept views over the same underlying knoxel IDs, each with its own
    affect weighting for context selection or competition.
    """
    knoxel_ids: List[int] = Field(default_factory=list)
    affect: AffectMeta = Field(default_factory=AffectMeta)


class CodeletPercept(BaseModel):
    """
    Minimal header that every percept carries for selection and blending.
    Values are strings because you enforce schema via llama.cpp; keep parsing downstream.
    """
    salience: float = Field(ge=0, le=1, description="How salient is this Percept? [0..1]. 0 = irrelevant, 1 = decisive for {companion_name}.")
    valence: float = Field(ge=-1, le=1, description="Affect sign for {companion_name}. [-1..1]. -1 = aversive, 0 = neutral, 1 = positive.")
from __future__ import annotations

from pm.records.enums import IntentionType

"""
Additional knoxel types and narrative feature subtype seeds for the companion architecture.

Design goals
------------
This module is intentionally pragmatic rather than elegant.

It extends the existing knoxel ecosystem with a few new runtime-persistent object types that
fit the current project style:
- broad BaseModel objects with string-heavy fields
- linked IDs instead of deep nesting
- partially overlapping conceptual roles
- enough scalar fields for heuristics and LLM-guided ranking
- rich docstrings so downstream tooling can gather schema intent dynamically

Important integration note
--------------------------
These classes inherit from ``KnoxelBase``. In the current project, ``KnoxelBase.type`` resolves
the class name through the central ``KnoxelType`` enum in ``knoxel_core.py``. If you want the
``type`` property to work for these classes, add matching enum entries there.

Suggested additions in KnoxelType:
- BeliefKnoxel
- BeliefShiftKnoxel
- AttentionKnoxel
- AttentionFrameKnoxel
- OntologyTermKnoxel
- OntologyBindingKnoxel
- PromptFeatureKnoxel

This file does not monkey-patch the central enum on purpose.
"""

from enum import StrEnum, auto
from typing import Any, Dict, List, Optional

from pydantic import Field

from pm.records.base import RecordBase


class Intention(RecordBase):
    """
    Active goal or expectation tracked across future ticks.
    and lifecycle markers (status/timeout). Often created by actions and later evaluated
    against observed outcomes.
    Can be hierarchical with big intentiosn and sub intentions.
    """
    entity_id: int = Field(description="Usually the ID of the AI companion, might be used later to track intentions of other entites.")
    active: bool = Field(default=True, description="Intentions are disable when fulfilled, discarded or replaced/refactored into new intention.")
    urgency: float = Field(default=0.5, ge=0.0, le=1.0, description="Urgency factor, dynamically changes on situation.")
    affective_valence: float = Field(..., ge=-1.0, le=1.0, description="Expected valence if fulfilled (for internal goals) or desired valence if met (for external expectations).")
    incentive_salience: float = Field(default=0.5, ge=0.0, le=1.0, description="How much it 'pulls' attention/action.")
    fulfilment: float = Field(default=0.0, ge=0.0, le=1.0, description="Current fulfillment status (0=not met, 1=fully met).")
    grandness: float = Field(default=0.0, ge=0.0, le=1.0, description="Size of goal, with 0.0 being getting a glass of water and 1.0 being solving world hunger. Dynamically adjusted if subgoal.")
    control: float = Field(default=0.5, ge=0.0, le=1.0, description="How much entity can affect fulfillment.")
    intention_type: IntentionType = Field(description="Internal goal, drive, attractor, external expectation, state")
    precursor_ids: List[int] = Field(default_factory=list, description="IDs of coalesced past intentions or other forms of precursor due to consolidation.")
    owner_ids: List[int] = Field(default_factory=list, description="IDs of 'owning' intentions in case this is intention is a subset of others due to consolidation.")
    status: str = Field(default="active", description="active, completed, failed, pending")
    timeout: int = Field(default=100, description="Ticks until this intention expires")

class BeliefRecord(RecordBase):
    """
    Persistent-but-revisable epistemic stance.

    A BeliefRecord represents what the companion currently takes to be the case, suspects,
    lacks, or theorizes. It is intentionally broader and messier than a crisp logical proposition.
    The goal is not formal proof but architecture-facing traceability.

    Typical uses
    ------------
    - track known, guessed, missing, or theorized information
    - attach confidence, importance, and valence for heuristics and prompt shaping
    - anchor belief updates from appraisal, simulation, structure building, or critique modules
    - support contradiction tracking without requiring a full theorem prover

    Recommended belief_kind values
    ------------------------------
    - "known"
    - "guess"
    - "missing"
    - "theory"
    - "counterfactual"
    - "self_model"
    - "user_model"
    - "world_model"

    content
        Human-readable belief statement. Keep it short enough that it still makes sense when
        directly shown in traces, debug logs, or prompt assembly.

    belief_key
        Cheap dedupe / lookup key. Not guaranteed unique globally, but should usually be stable
        across nearby reformulations of the same belief.

    belief_kind
        Seed label describing what kind of epistemic stance this is.

    confidence
        How strongly the system currently endorses the belief.

    certainty
        Similar to confidence but focused on felt settledness rather than endorsement strength.
        Keeping both is mildly redundant, which is fine here because different heuristics may
        want one or the other.

    importance
        How consequential this belief is for action selection, memory prioritization, or prompting.

    valence
        Affective sign attached to the belief's implications, not whether it is true.

    about_entity_ids
        Entities this belief is about, if any.

    supporting_knoxel_ids / contradicting_knoxel_ids
        Cheap traceability hooks for evidence and conflict analysis.

    source
        Where this belief came from. Example values: "user_message", "memory_recall",
        "world_simulation", "meta_reflection", "manual_seed"

    active
        Whether this belief is still live in current use. Inactive beliefs may still be kept for
        history, comparison, or later revival.

    superseded_by_id
        Optional link to a newer belief that replaced this one.
    """

    belief_key: str = Field(default="", description="Cheap lookup key for grouping or deduping similar beliefs.")
    belief_kind: str = Field(default="guess", description="Seed label for the epistemic stance, e.g. known, guess, missing, theory.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Current endorsement strength of this belief.")
    certainty: float = Field(default=0.5, ge=0.0, le=1.0, description="How settled or resolved this belief feels to the system.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="How much this belief matters for current cognition or action.")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Affective sign attached to the implications of the belief.")
    about_entity_ids: List[int] = Field(default_factory=list, description="IDs of entities this belief concerns.")
    supporting_knoxel_ids: List[int] = Field(default_factory=list, description="IDs that support or motivate this belief.")
    contradicting_knoxel_ids: List[int] = Field(default_factory=list, description="IDs that currently conflict with this belief.")
    source: str = Field(default="unknown", description="Origin hint for debugging, critique, and trust heuristics.")
    active: bool = Field(default=True, description="Whether the belief is currently live in reasoning and prompt building.")
    superseded_by_id: Optional[int] = Field(default=None, description="If replaced, points at the newer belief knoxel.")


class BeliefShiftRecord(RecordBase):
    """
    Proposed or recorded change to belief state.

    Use this when a module does not want to directly overwrite persistent epistemic state but
    instead wants to contribute a traceable update proposal. This is especially useful for:
    - appraisal outputs
    - simulation summaries
    - meta-level critique
    - contradiction handling
    - commit-phase architecture designs

    Recommended shift_type values
    -----------------------------
    - "add"
    - "increase_confidence"
    - "decrease_confidence"
    - "mark_missing"
    - "supersede"
    - "split"
    - "merge"
    - "deactivate"

    target_belief_id
        Existing belief this shift refers to, if any.

    proposed_belief_key
        Key for a new or not-yet-created belief when no target exists yet.

    delta_confidence / delta_importance
        Suggested scalar changes. These are not required to be mathematically elegant. They are
        practical knobs for heuristics and later commit logic.

    reason
        Human-readable why-string. Keep it concise but informative because it may later be shown
        in debugging views or introspective summaries.

    causal_knoxel_ids
        Knoxels that drove the proposed shift.

    accepted
        Whether the shift has been committed to persistent state. Useful if you separate propose
        from commit but still want history retained.
    """

    target_belief_id: Optional[int] = Field(default=None, description="Existing belief affected by this shift, if present.")
    proposed_belief_key: str = Field(default="", description="Lookup key for a belief that may be created or updated.")
    shift_type: str = Field(default="add", description="Seed label for the kind of change being proposed.")
    delta_confidence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Suggested confidence change.")
    delta_importance: float = Field(default=0.0, ge=-1.0, le=1.0, description="Suggested importance change.")
    reason: str = Field(default="", description="Compact explanation of why the belief shift is being proposed.")
    causal_knoxel_ids: List[int] = Field(default_factory=list, description="Knoxels that causally motivated the shift.")
    accepted: bool = Field(default=False, description="Whether the shift has already been committed.")


class AttentionRecord(RecordBase):
    """
    Candidate attention nomination for competition and ranking.

    This is the cheap bridge between raw percept/codelet outputs and later selection heuristics.
    Multiple modules can nominate the same targets for different reasons without mutating the
    targets themselves.

    Recommended attention_kind values
    ---------------------------------
    - "threat"
    - "novelty"
    - "goal_relevance"
    - "relationship"
    - "uncertainty"
    - "self_relevance"
    - "meta_alert"

    target_knoxel_ids
        The items this attention nomination points at.

    score
        Overall precomputed priority score. Often redundant with the other scalar fields, but it is
        practical to keep because some heuristics or DSPy modules may want to emit a ready-made
        ranking signal.

    novelty / urgency / goal_relevance / reportability
        Separate dimensions that can later be combined differently by object-level and meta-level
        selectors.

    competition_group
        Lets you divide the space into separate competitions if needed. For example, you may want
        social-threat items to compete mostly with other social-threat items before final arbitration.

    selected
        Whether this nomination ended up winning or contributing to the winning frame.

    from_codelet
        Light provenance field naming the nominating process.
    """

    target_knoxel_ids: List[int] = Field(default_factory=list, description="Knoxels nominated for attention by this object.")
    attention_kind: str = Field(default="goal_relevance", description="Seed label describing why the targets deserve attention.")
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall priority estimate for competition heuristics.")
    novelty: float = Field(default=0.0, ge=0.0, le=1.0, description="How unexpected or fresh the nominated content is.")
    urgency: float = Field(default=0.0, ge=0.0, le=1.0, description="How time-sensitive the nominated content is.")
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="How strongly the nominated content matters to active goals.")
    reportability: float = Field(default=0.0, ge=0.0, le=1.0, description="How likely the nominated content should become prompt-visible.")
    competition_group: str = Field(default="default", description="Optional grouping hint for staged or local competition.")
    selected: bool = Field(default=False, description="Whether this nomination won or helped define the active frame.")
    from_codelet: str = Field(default="", description="Name of the codelet or module that produced this nomination.")


class AttentionFrameRecord(RecordBase):
    """
    Snapshot of what currently matters.

    This is the cheap, temporally bounded object that says: these are the attention winners,
    these are the active goals around them, and this is the current control stance. It can serve
    as the bridge between CSM competition and workspace/prompt materialization.

    focus_knoxel_ids
        Everything currently in the narrow frame of concern.

    winning_attention_ids
        The attention nominations that defined the frame.

    active_goal_ids
        Goals or intentions that are currently steering the frame.

    control_mode
        Broad control regime such as "reactive", "reflective", or "blended".

    frame_reason
        Human-readable summary of why this frame exists.

    expires_after_ticks
        Lightweight lifetime hint. Useful if you want sticky attention without making it permanent.
    """

    focus_knoxel_ids: List[int] = Field(default_factory=list, description="Knoxels currently inside the active attention frame.")
    winning_attention_ids: List[int] = Field(default_factory=list, description="Attention nominations that shaped this frame.")
    active_goal_ids: List[int] = Field(default_factory=list, description="Goals or intentions currently steering the frame.")
    control_mode: str = Field(default="reactive", description="Broad control regime for the current frame.")
    frame_reason: str = Field(default="", description="Compact explanation of why this frame is active.")
    expires_after_ticks: int = Field(default=1, ge=0, description="Approximate lifetime hint before the frame should be reconsidered.")


class OntologyTermRecord(RecordBase):
    """
    Evolving private-ontology vocabulary item.

    This is not a formal OWL class and should not pretend to be one. It is a pragmatic record of
    a term the agent uses or may use to refer to recurring internal patterns, social situations,
    self-descriptions, or reportable qualitative states.

    Recommended term_kind values
    ----------------------------
    - "quale"
    - "self_descriptor"
    - "relationship_label"
    - "mental_move"
    - "mood_word"
    - "goal_word"

    term
        Preferred surface form of the term. Often duplicates or specializes ``content``.

    definition
        Local meaning as the system currently understands it.

    usage_rule
        Optional guidance for when the term should or should not be used in prompt-visible speech.

    example_knoxel_ids
        Historical examples or anchor cases supporting this term's meaning.

    related_term_ids
        Nearby or competing ontology terms.

    embedding_query_hint
        Extra retrieval phrasing that helps vector search find relevant uses.

    stability
        How entrenched the term currently is. Low stability means the term is still experimental.

    preferred
        Whether this is currently the favored label among related alternatives.
    """

    term: str = Field(default="", description="Preferred surface form of the ontology term.")
    term_kind: str = Field(default="quale", description="Seed label for the rough role of the term.")
    definition: str = Field(default="", description="Current local definition of the term.")
    usage_rule: str = Field(default="", description="Guidance for when the term is appropriate to use.")
    example_knoxel_ids: List[int] = Field(default_factory=list, description="Example usages or anchor episodes for the term.")
    related_term_ids: List[int] = Field(default_factory=list, description="IDs of nearby or competing ontology terms.")
    embedding_query_hint: str = Field(default="", description="Helpful retrieval hint for vector search or semantic lookup.")
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="How stable or entrenched this term currently is.")
    preferred: bool = Field(default=False, description="Whether this is the currently preferred label among alternatives.")


class OntologyBindingRecord(RecordBase):
    """
    Temporary or current fit between a private-ontology term and live state.

    This object says that a specific term currently fits a bundle of active state items for a
    specific reason and with a specific strength.

    Recommended binding_kind values
    -------------------------------
    - "quale_match"
    - "self_model_match"
    - "narrative_match"
    - "belief_pattern_match"
    - "relationship_pattern_match"

    term_id
        The ontology term being bound.

    bound_knoxel_ids
        Current items that jointly justify the binding.

    fit_score
        How well the current state matches this term.

    explanation
        Compact explanation of the match.

    active
        Whether the binding is live right now.
    """

    term_id: int = Field(description="Ontology term currently being applied.")
    bound_knoxel_ids: List[int] = Field(default_factory=list, description="Current-state knoxels that justify the binding.")
    binding_kind: str = Field(default="quale_match", description="Seed label for the kind of fit represented here.")
    fit_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How well the term fits the current state.")
    explanation: str = Field(default="", description="Compact explanation of why the term fits.")
    active: bool = Field(default=True, description="Whether the binding is live in the current cycle.")

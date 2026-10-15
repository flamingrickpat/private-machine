"""A18 self-inspection: source grants, capability overlay, episodes, and beliefs.

The package gathers evidence about the companion's own implementation. It
distinguishes design, wiring, and observed execution, and it keeps every claim
beside its evidence revision. It owns no model call and no state outside the
accepted memory store.
"""
from pm.introspection.capabilities import (
    CapabilityContext,
    CapabilityDeclaration,
    CapabilityEvidence,
    CapabilityFinding,
    CapabilityOverlay,
    CapabilityState,
    build_overlay,
    default_capability_declarations,
)
from pm.introspection.episodes import Episode, EpisodeIndex, answer_questions
from pm.introspection.grants import (
    SourceDenied,
    SourceGrant,
    SourceInspector,
    SourceUnavailable,
    build_default_grant,
    git_revision,
)
from pm.introspection.self_model import (
    capability_belief_key,
    mark_stale_beliefs,
    narrative_belief_key,
    record_capability_beliefs,
    record_narrative_belief,
)

__all__ = [
    "CapabilityContext",
    "CapabilityDeclaration",
    "CapabilityEvidence",
    "CapabilityFinding",
    "CapabilityOverlay",
    "CapabilityState",
    "Episode",
    "EpisodeIndex",
    "SourceDenied",
    "SourceGrant",
    "SourceInspector",
    "SourceUnavailable",
    "answer_questions",
    "build_default_grant",
    "build_overlay",
    "capability_belief_key",
    "default_capability_declarations",
    "git_revision",
    "mark_stale_beliefs",
    "narrative_belief_key",
    "record_capability_beliefs",
    "record_narrative_belief",
]

"""The theory-of-mind specialization of a shared private inquiry.

See :mod:`workflows.inner_thought.host` for the canonical overview of the
private loop and the inquiry graph. This module gives one deliberate
perspective inquiry (the ``theory_of_mind`` mental action) what it may read
about its subject: the companion's accepted entity model of that subject and
the original messages behind it. It renders that material for the thought
prompt and states the epistemic rules that keep inference, report, attributed
knowledge, and unknown separate.

The subject is the entity that the action's partner named. The material is
captured without a model call from the accepted entity-context family of the
same observer, reality, and frame, so a perspective step never reads another
character's model or another conversation. Thinking proposes entries; only
main cognition writes them (decision D2 in the B06 plan).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from pm.constants import WORKSPACE_TAIL_TOKENS
from pm.memory.snapshot import Snapshot
from pm.records.memory_records import Entity
from workflows.entity_context.owner import active_claims
from workflows.entity_context.projection import project_entity_context, render_entity_context
from workflows.entity_context.schemas import EntityContextKey, EntityContextView, EntityProjectionQuery
from workflows.inner_thought.schemas import InquiryContract, InquiryGrant


class PerspectiveContext(InquiryContract):
    """What one perspective-inquiry step reads about its subject.

    ``evidence_source_ids`` are the original messages behind the accepted
    entries, filtered to the current view, the accepted commits, and this
    inquiry's reality and frame. They become citable candidates and extend
    the step grant; the subject and observer identity anchors do not.
    """

    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity this inquiry models.")
    subject_name: str = Field(min_length=1, description="Canonical name of the subject entity.")
    subject_class: str = Field(min_length=1, description="EntityClass value, so a non-human subject is visible.")
    entity_view: EntityContextView = Field(description="Accepted view of the subject, captured for this step.")
    evidence_source_ids: tuple[int, ...] = Field(
        description="Original sources behind that view: current, accepted, same scope, not identity anchors.")


def capture_perspective_context(view: Snapshot, *, subject_entity_id: int, grant: InquiryGrant,
                                accepted_ids: frozenset[int], as_of_utc: datetime) -> PerspectiveContext:
    """Freeze the accepted entity view of one subject for one perspective step.

    The claim keys come from the active family, so the whole accepted model
    of the subject is readable, and the projection budget still bounds it.
    Only the observer's own reality and frame is projected: the grant already
    carries that scope, and a subject outside it cannot appear.
    """
    if as_of_utc.tzinfo is not None:
        raise ValueError("Perspective context requires naive UTC world time.")
    subject = view.get(subject_entity_id)
    if not isinstance(subject, Entity):
        raise ValueError("A perspective inquiry requires an entity subject.")
    key = EntityContextKey(subject_entity_id=subject_entity_id,
                           observer_entity_id=grant.observer_entity_id, scope=grant.scope)
    claim_keys = list(dict.fromkeys(claim.claim_key for claim in active_claims(view, key=key)))
    entity_view = project_entity_context(view,
        query=EntityProjectionQuery(key=key, claim_keys=claim_keys, token_budget=WORKSPACE_TAIL_TOKENS),
        as_of_utc=as_of_utc)
    evidence_source_ids: list[int] = []
    for dependency in entity_view.source_dependencies:
        record_id = dependency.record_id
        if record_id not in view.all_knoxels or record_id not in accepted_ids:
            continue
        record = view.get(record_id)
        if record.scope != grant.scope or isinstance(record, Entity):
            # The identity anchors are header material, not evidence. A source
            # in another reality or frame stays out of this inquiry's grant.
            continue
        evidence_source_ids.append(record_id)
    return PerspectiveContext(subject_entity_id=subject.id, subject_name=subject.content,
        subject_class=subject.entity_class.value, entity_view=entity_view,
        evidence_source_ids=tuple(evidence_source_ids))


def render_perspective_context(context: PerspectiveContext) -> str:
    """Write the subject line and the accepted entity view for one perspective step."""
    lines = [f"Subject {context.subject_name} (entity {context.subject_entity_id}, "
             f"class {context.subject_class})."]
    if not context.entity_view.claims:
        lines.append("No accepted entry exists for this subject.")
    else:
        lines.append(render_entity_context(context.entity_view))
    return "\n".join(lines)


def perspective_rule(evidence: str, *, companion_name: str, subject_name: str) -> str:
    """State how a perspective step reads and uses its subject material.

    The evidence is the rendered accepted view of the subject. The rules keep
    the four epistemic roles apart, forbid contact, and stop repeated thinking
    from masquerading as support. Main cognition decides anything that asks
    the subject directly.
    """
    return (
        f"This inquiry models {subject_name} as {companion_name} understands them.\n"
        "The lines below are accepted entries. A report is what was said. An inference is an estimate. "
        "Entries marked contested, expired, or due for review are uncertain.\n"
        "Keep four things apart: what the subject said or did (cite the record), what the companion infers, "
        "what the subject may know, believe, or want (attributed and fallible), and what is unknown.\n"
        "The companion's knowledge is not the subject's knowledge.\n"
        "Consider one alternative explanation and one exception when the evidence allows it.\n"
        "Do not give a non-human entity human sleep, work, motives, or feelings without evidence.\n"
        "A routine is an estimate. It is never an observation of today.\n"
        "Repeating an earlier thought adds no support.\n"
        "This inquiry cannot contact the subject. If asking is the best way to learn something, state it as "
        "an information gap. Main cognition decides whether to ask.\n"
        "Perspective material:\n" + evidence)


__all__ = [
    "PerspectiveContext", "capture_perspective_context", "perspective_rule", "render_perspective_context",
]

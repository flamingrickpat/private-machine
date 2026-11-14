"""The metacognition specialization of a shared private inquiry.

See :mod:`workflows.inner_thought.host` for the canonical overview of the
private loop and the inquiry graph. This module gives a noticing step
(``rumination_notice`` or ``attention_drift_notice``) the recorded facts about
the companion's own recent thinking: which sources several steps returned to,
which of those steps accepted no progress, and the window they came from. The
facts are built without a model call from the recent-thought views.

The facts are deliberately bare. They carry no cause and no judgment: why the
thinking returns to a source is the step's own reading and can be wrong. A
noticing is delivered to main cognition as an intermediate result while the
episode continues, and the ``repeated_concern`` trigger stops re-firing for a
source the noticing already covered (the "notice once" rule).
"""
from __future__ import annotations

from collections.abc import Sequence

from workflows.inner_thought.request_views import (
    RecentThoughtView, RepeatedSourceActivity, ThoughtActivityView,
)

#: Operations whose step reads the recorded activity of the companion's own
#: recent thinking. They are the noticing bundle of the delivered catalogue.
METACOGNITION_OPERATIONS: frozenset[str] = frozenset({"rumination_notice", "attention_drift_notice"})


def _stalled(step: RecentThoughtView) -> bool:
    """Only an inquiry step records a progress judgment; none is a stall."""
    return step.inquiry is not None and step.inquiry.progress_kind == "none"


def _new_evidence(step: RecentThoughtView) -> bool:
    """An inquiry step that accepted newly inspected original evidence."""
    return step.inquiry is not None and step.inquiry.progress_kind == "new_evidence"


def thought_activity(recent: Sequence[RecentThoughtView]) -> ThoughtActivityView:
    """Count recorded facts about recent private thinking; no judgment and no cause.

    ``recent`` is newest first, and every returned step list keeps that order.
    A linear thought (``loop == \"inner\"``) is a step but never stalled: it
    records no progress judgment. ``repeated`` lists each source that at
    least two steps cite, with the episodes of the citing steps.
    """
    window_step_ids = tuple(step.knoxel_id for step in recent)
    stalled_ids = [step.knoxel_id for step in recent if _stalled(step)]
    new_evidence_ids = [step.knoxel_id for step in recent if _new_evidence(step)]
    cited_by: dict[int, list[int]] = {}
    episode_of: dict[int, str] = {}
    for step in recent:
        for source_id in step.source_ids:
            cited_by.setdefault(source_id, []).append(step.knoxel_id)
            if step.inquiry is not None:
                episode_of[step.knoxel_id] = step.inquiry.episode_id
    repeated: list[RepeatedSourceActivity] = []
    for source_id in sorted(cited_by):
        step_ids = cited_by[source_id]
        if len(step_ids) < 2:
            continue
        step_id_set = set(step_ids)
        stalled_step_ids = tuple(step.knoxel_id for step in recent
                                 if step.knoxel_id in step_id_set and _stalled(step))
        episode_ids = tuple(dict.fromkeys(episode_of[step_id] for step_id in step_ids
                                          if step_id in episode_of))
        repeated.append(RepeatedSourceActivity(source_id=source_id, step_ids=tuple(step_ids),
            stalled_step_ids=stalled_step_ids, episode_ids=episode_ids))
    return ThoughtActivityView(window_step_ids=window_step_ids, repeated=tuple(repeated),
        stalled_step_count=len(stalled_ids), new_evidence_step_count=len(new_evidence_ids))


def render_thought_activity(activity: ThoughtActivityView) -> str:
    """Render the recorded activity facts without interpreting their cause."""
    lines = [f"Window: your {len(activity.window_step_ids)} most recent private steps "
             f"{list(activity.window_step_ids)}.",
             f"{activity.stalled_step_count} of them accepted no new evidence or comparison; "
             f"{activity.new_evidence_step_count} accepted newly inspected evidence."]
    if not activity.repeated:
        lines.append("No source returns in more than one of these steps.")
    for item in activity.repeated:
        lines.append(f"Source {item.source_id} returns in {len(item.step_ids)} steps "
                     f"{list(item.step_ids)}; {len(item.stalled_step_ids)} of those accepted no progress.")
    return "\n".join(lines)


def activity_rule(evidence: str) -> str:
    """State what the recorded activity is and is not, for the step's instruction."""
    return (
        "These lines are recorded facts about your own recent private thinking. They are not evidence "
        "about anyone else.\n"
        "They do not show why you return to this. Any reason you give is your own reading, and it can be "
        "wrong.\n"
        "Returning to a thought adds no support for it.\n"
        "You can pause, name the evidence that is missing, or continue with one specific different question.\n"
        "Nothing requires you to feel calmer, and nothing requires you to stop.\n"
        "Recorded activity:\n" + evidence)


__all__ = [
    "METACOGNITION_OPERATIONS", "activity_rule", "render_thought_activity", "thought_activity",
]

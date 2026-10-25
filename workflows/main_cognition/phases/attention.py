"""Attention phase: rate the structures for each shard and update the shards.

Port of ``_simulate_attention_on_coalitions`` from
``private-machine-head/pm/ghosts/ghost_lida.py``. The phase updates the
self shard and the shard of each current stimulus group. Steps:

1. Deliberate focus. Fresh focus proposals are accepted or rejected
   (``resolve_focus``). Accepted advice becomes CSM endorsements with the
   endorser ``deliberate_focus``. Each endorsement adds at most
   ``FOCUS_ACTIVATION_WEIGHT``. A reduced record loses at most that much.
2. Rating (model). One story-framed call for each shard rates every
   structure from 0 to 1 for that shard. One call per shard keeps each
   prompt about one kind of awareness. The calls read the character card
   (in the shared prefix), the AttentionFocus narrative, the state of mind
   in words, and the mental aperture instruction of the old code. The calls
   run in parallel.
3. Auxiliary rating (Python). The old rule-based rating of the records of a
   structure. Normalized and weighted with ``COALITION_AUX_RATING_FACTOR``,
   it only breaks ties. Then only the best structures of each shard win:
   one at a narrow mental aperture, as in the old code, two at a medium
   aperture, and three at a broad one (:func:`winner_count`).
4. Selection (Python). ``workspace_selection.select_shards`` changes each
   shard in place: reserved places, then the other content in score order
   within the budget and the caps. Incumbents get ``WORKSPACE_HOLD_BONUS``.
   A record gets ``WORKSPACE_HOME_BONUS`` in its home shard.
5. Effects. A codelet whose percept entered a shard gets a higher
   base-level activation (LIDA attentional learning). The commit makes each
   entered percept causal, so its delta enters the EMA mental state of the
   next cycle.

Exhausted repairs of a rating call raise ``ModelFormatError``, and the host
retries the whole cycle.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, create_model, model_validator

from pm.constants import (
    ATTENTION_APERTURE_THRESHOLD,
    ATTENTION_LINE_CHARS,
    ATTENTION_WINNERS_BROAD,
    ATTENTION_WINNERS_DEFAULT,
    ATTENTION_WINNERS_NARROW,
    COALITION_AUX_RATING_FACTOR,
    COALITION_RECENCY_DECAY,
    FOCUS_ACTIVATION_WEIGHT,
    THREAT_APPRAISAL_THRESHOLD,
    WORKSPACE_HOME_BONUS,
)
from pm.context_render.story import shard_focus
from pm.context_render.turns import format_content
from pm.llm.concurrency import ordered_parallel_map
from pm.records.situational_model import SituationalItem
from pm.records.workspace import SELF_SHARD, ShardKey, WorkspaceShard

from workflows.cognitive_control.attention import resolve_focus
from workflows.cognitive_control.schemas import FocusState
from workflows.inference_settings import ATTENTION_WORKERS
from workflows.main_cognition.codelet_registry import reinforce
from workflows.main_cognition.csm_view import (
    Embedder, match_context, record_tokens, story_line, workspace_requirements,
)
from workflows.main_cognition.cycle_work import CycleWork, provisional_tick
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import (
    AppraisalView, AttentionResult, RatedStructure, ShardRatings, Stimulus, StructureDraft, StructureRatings,
)
from workflows.main_cognition.story_prefix import StoryPrefix, shard_label, shown_shards
from workflows.main_cognition.workspace_selection import Candidate, SelectionInput, select_shards, selector_matches

#: The endorser name of deliberate focus in the CSM.
DELIBERATE_FOCUS = "deliberate_focus"


class FocusEvidence(BaseModel):
    """The cycle context that the focus decision reads beside each proposal."""

    appraisal: AppraisalView = Field(default_factory=AppraisalView)
    codelet_notes: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    stimuli: list[Stimulus] = Field(default_factory=list)


def focus_evidence(work: CycleWork) -> FocusEvidence:
    """Return the evidence of the focus decision: the appraisal, the codelet texts, the goals, the stimuli."""
    notes = [text for output in work.codelets.outputs for variant in output.variants
             for text in [variant.recommendation, *(percept.text for percept in variant.percepts)]]
    return FocusEvidence(appraisal=work.appraisal_view(), codelet_notes=notes, goals=work.request.active_goals,
                         stimuli=work.request.stimuli)


def apply_focus(work: CycleWork, focus: FocusState | None, embedder: Embedder, *,
                danger: bool = False) -> dict[int, float]:
    """Turn accepted focus advice into bounded CSM endorsements. Return the focus bonus by record id.

    A promoted record gets ``add_or_boost`` with the endorser
    ``deliberate_focus`` and a boost of ``min(strength,
    FOCUS_ACTIVATION_WEIGHT)``. A record that the CSM does not hold enters
    it, when the cycle has its view. A reduced record loses the same amount
    of activation, down to 0. In danger, reduction does nothing: the
    companion cannot look away from a threat (as the old A15 focus bias).
    Prune protection and reserved places still apply, so focus cannot
    remove a required item. Rejected or missing advice changes nothing.
    """
    if focus is None or not focus.accept:
        return {}
    amount = min(max(0.0, focus.strength), FOCUS_ACTIVATION_WEIGHT)
    tick = provisional_tick(work)
    csm = work.state.csm
    bonus: dict[int, float] = {}
    for record_id in focus.promote_ids:
        if record_id not in work.records:
            continue
        item = csm.add_or_boost(SituationalItem(record_id=record_id, entered_tick=tick, last_boost_tick=tick,
                                                activation=amount), DELIBERATE_FOCUS, embedder.record, boost=amount)
        bonus[item.record_id] = amount
    for record_id in [] if danger else focus.reduce_ids:
        if record_id in csm.items:
            csm.items[record_id].activation = max(0.0, csm.items[record_id].activation - amount)
            bonus[record_id] = -amount
    return bonus


def attention_narrative(work: CycleWork) -> str:
    """Return the newest AttentionFocus narrative of the self, or a placeholder."""
    self_id = work.request.self_entity_id
    narratives = [view for view in work.records.values()
                  if view.record_class == "Narrative" and view.category == "AttentionFocus"
                  and self_id in view.entity_ids]
    if not narratives:
        return "(no narrative yet)"
    return max(narratives, key=lambda view: (view.timestamp is not None, view.timestamp or 0, view.record_id)).content


def aperture_text(work: CycleWork, prompts: PhasePrompts) -> str:
    """Return the old mental aperture instruction for the current state."""
    aperture = work.state.mental_state.state_cognition.mental_aperture
    if aperture < -ATTENTION_APERTURE_THRESHOLD:
        return prompts.aperture_narrow
    if aperture > ATTENTION_APERTURE_THRESHOLD:
        return prompts.aperture_broad
    return prompts.aperture_default


def winner_count(work: CycleWork) -> int:
    """Return how many structures can win one shard, by the mental aperture.

    The old code put only the one winning coalition into the workspace. A
    narrow aperture keeps that rule. A medium or broad aperture lets two or
    three structures win, because a shard holds several kinds of content.
    """
    aperture = work.state.mental_state.state_cognition.mental_aperture
    if aperture < -ATTENTION_APERTURE_THRESHOLD:
        return ATTENTION_WINNERS_NARROW
    if aperture > ATTENTION_APERTURE_THRESHOLD:
        return ATTENTION_WINNERS_BROAD
    return ATTENTION_WINNERS_DEFAULT


def aux_rating(work: CycleWork, structure: StructureDraft) -> float:
    """Return the old auxiliary rating of a structure: the sum over its records.

    Port of the rule part of ``_simulate_attention_on_coalitions``. For each
    record, ``base * recency + public * (1 + ego_strength)``:

    - ``recency`` is ``COALITION_RECENCY_DECAY ** ticks`` since the record
      entered the CSM, at least 0.1.
    - ``base`` is 0.3 for a goal, the larger of the absolute valence and 0.1
      for a feeling, an experience, or an expectation outcome, 0.15 for a
      narrative update, 0.1 for a narrative or a memory, and the salience for
      a codelet percept. The old code had percepts as features without a
      base score. Here their salience counts, so a strong impulse can break
      a tie.
    - ``public`` is 1 for a dialogue turn, as the old interlocus value.
    """
    tick = provisional_tick(work)
    ego = 1.0 + work.state.mental_state.state_cognition.ego_strength
    total = 0.0
    for record_id in structure.record_ids:
        view = work.records[record_id]
        item = work.state.csm.items[record_id] if record_id in work.state.csm.items else None
        recency = max(0.1, COALITION_RECENCY_DECAY ** (tick - item.entered_tick if item is not None else 0))
        base = 0.0
        if view.record_class == "Intention":
            base = 0.3
        elif view.record_class == "CodeletPercept":
            base = (item.salience or 0.0) if item is not None else 0.0
        elif view.feature_type in {"Feeling", "SubjectiveExperience", "ExpectationOutcome"}:
            base = max(abs(item.valence or 0.0) if item is not None else 0.3, 0.1)
        elif view.feature_type == "NarrativeUpdate":
            base = 0.15
        elif view.record_class in {"Narrative", "MemoryClusterRecord", "DeclarativeFactRecord"}:
            base = 0.1
        public = 1.0 if view.feature_type == "Dialogue" else 0.0
        total += base * recency + public * ego
    return total


class ShardJob(BaseModel):
    """One rating call: the shard, its labels, and its instruction."""

    key: ShardKey = Field(default_factory=ShardKey)
    labels: list[str] = Field(default_factory=list)
    instruction: str = ""


def render_structures(work: CycleWork, structures: list[StructureDraft]) -> str:
    """Render the structures with labels ``S1``, ``S2``, ... for a rating call.

    Each record is one story line without its id, cut to
    ``ATTENTION_LINE_CHARS`` characters, so one prompt can show every
    structure.
    """
    blocks = []
    for index, structure in enumerate(structures, start=1):
        lines = [story_line(work.records[record_id], with_id=False)[:ATTENTION_LINE_CHARS]
                 for record_id in structure.record_ids]
        blocks.append("\n".join([f"### S{index}: {structure.theme}", f"Why: {structure.reason}",
                                 *(f"- {line}" for line in lines)]))
    return "\n\n".join(blocks)


def shard_job(work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts, shard: WorkspaceShard,
              structures: list[StructureDraft]) -> ShardJob:
    """Return the rating call of one shard."""
    name = shard_label(shard.key, work.request)
    instruction = format_content(prefix.fill(prompts.user_template),
        attention_narrative=attention_narrative(work),
        mental_state=work.affect.felt or work.state_summary or "(no feeling recorded)",
        shard_focus=shard_focus(shard.key, name, prefix.companion, prompts.shard_labels),
        structures=render_structures(work, structures),
        aperture=prefix.fill(aperture_text(work, prompts)),
        winners=str(winner_count(work)),
        count=str(len(structures)),
    ).text
    return ShardJob(key=shard.key, labels=[f"S{index}" for index in range(1, len(structures) + 1)],
                    instruction=instruction)


def rate(context, work: CycleWork, prefix: StoryPrefix, job: ShardJob) -> StructureRatings:
    """Run one rating call. Runs in a worker thread and does not change ``work``.

    A missing, repeated, or unknown label is invalid output and gets a
    repair turn.
    """
    wanted = job.labels

    def check(answer):
        given = [item.structure for item in answer.ratings]
        if sorted(given) != sorted(wanted):
            raise ValueError(f"Rate each label exactly once: {', '.join(wanted)}.")
        return answer

    schema = create_model("StructureRatingsChecked", __base__=StructureRatings,
                          __validators__={"check": model_validator(mode="after")(check)})
    return complete(context, work, prefix.ask(job.instruction), schema)


def scored(work: CycleWork, job: ShardJob, answer: StructureRatings,
           structures: list[StructureDraft]) -> ShardRatings:
    """Combine the model ratings and the auxiliary ratings of one shard.

    ``score = rating + aux / max_aux * COALITION_AUX_RATING_FACTOR``. When
    every auxiliary rating is 0, the score is the rating.
    """
    by_label = {item.structure: item for item in answer.ratings}
    aux = [aux_rating(work, structure) for structure in structures]
    top = max(aux, default=0.0)
    ratings = []
    for label, structure, value in zip(job.labels, structures, aux, strict=True):
        rating = by_label[label]
        bonus = value / top * COALITION_AUX_RATING_FACTOR if top > 0.0 else 0.0
        ratings.append(RatedStructure(structure_id=structure.record_id, label=label, rating=rating.rating,
                                      aux=round(value, 4), score=round(rating.rating + bonus, 4),
                                      reasoning=rating.reasoning.strip()))
    # Only the best structures of the shard win. The others stay below
    # awareness in this cycle, even with a high rating.
    for rated in sorted(ratings, key=lambda item: (-item.score, item.label))[:winner_count(work)]:
        rated.won = True
    return ShardRatings(key=job.key, ratings=ratings)


def record_home(work: CycleWork, record_id: int) -> ShardKey | None:
    """Return the shard that a record belongs to, or None when it belongs to no particular shard.

    - A line that the companion said belongs to the shard of its listener.
      A line without a known listener belongs to the user.
    - A record about another entity belongs to that entity's shard. An
      entity of a current group comes first.
    - A record about the companion belongs to the self shard.
    - A codelet percept belongs to the shard that its variant named.
    - Any other record, for example an episode without an entity, belongs
      to no particular shard.
    """
    view = work.records[record_id]
    request = work.request
    self_id = request.self_entity_id
    if view.feature_type == "Dialogue" and self_id is not None and self_id in view.entity_ids:
        partner = view.addressee_entity_id or request.user_entity_id
        return ShardKey(kind="entity", ref=str(partner)) if partner else SELF_SHARD
    others = [entity_id for entity_id in view.entity_ids if entity_id != self_id]
    if others:
        present = [int(group.key.ref) for group in work.state.stimulus_groups if group.key.kind == "entity"]
        return ShardKey(kind="entity", ref=str(next((item for item in others if item in present), others[0])))
    if self_id is not None and self_id in view.entity_ids:
        return SELF_SHARD
    csm = work.state.csm.items
    if view.record_class == "CodeletPercept" and record_id in csm and csm[record_id].shard_hint is not None:
        return csm[record_id].shard_hint
    return None


def admits(work: CycleWork, key: ShardKey, record_id: int, updated: set[ShardKey]) -> bool:
    """Return True when the record may be an item of the shard ``key``.

    A record with a home shard (:func:`record_home`) may only enter that
    shard. When its home shard is not updated in this cycle, it may rest in
    the self shard. A record without a home shard may enter any shard.
    This rule keeps the content of one entity out of the shard of another
    entity. The decision and the reply for a group read only the self shard
    and the group's shard, so a misplaced record would leak into them.
    """
    home = record_home(work, record_id)
    if home is None or home == key:
        return True
    return key == SELF_SHARD and home not in updated


def home_shard(work: CycleWork, record_id: int, structure: StructureDraft) -> ShardKey:
    """Return the shard where a record belongs when the ratings tie.

    This is :func:`record_home`, or the target shard of the structure for a
    record without a home shard.
    """
    home = record_home(work, record_id)
    return home if home is not None else structure.target_shard


def newcomers(work: CycleWork, ratings: list[ShardRatings], bonus: dict[int, float]) -> list[Candidate]:
    """Return one candidate for each record of each winning structure and shard that admits it.

    The priority is the structure score, plus ``WORKSPACE_HOME_BONUS`` in the
    home shard of the record (:func:`home_shard`). A shard admits a record
    by :func:`admits`. An entity or workflow shard admits a record without a
    home shard only when the structure targets that shard. The self shard
    admits it from any structure.
    """
    structures = {draft.record_id: draft for draft in work.structures.structures}
    updated = {shard.key for shard in ratings}
    result = []
    for shard in ratings:
        for rated in shard.ratings:
            if not rated.won:
                continue
            structure = structures[rated.structure_id]
            reason = f"{structure.theme} (rated {rated.rating:.2f}): {rated.reasoning}"
            for record_id in structure.record_ids:
                if not admits(work, shard.key, record_id, updated):
                    continue
                if (record_home(work, record_id) is None and shard.key != SELF_SHARD
                        and structure.target_shard != shard.key):
                    continue
                home = WORKSPACE_HOME_BONUS if home_shard(work, record_id, structure) == shard.key else 0.0
                result.append(Candidate(
                    record_id=record_id, key=shard.key, priority=rated.score + home,
                    focus_bonus=bonus[record_id] if record_id in bonus else 0.0,
                    structure_id=structure.record_id, reason=reason))
    return result


def select(work: CycleWork, ratings: list[ShardRatings], bonus: dict[int, float]):
    """Update the shown shards in place and return their changes."""
    shards = shown_shards(work)
    updated = {shard.key for shard in shards}
    blocked = {item.knoxel_id for item in work.request.stimuli} | set(work.state.workspace.tail_ids())
    blocked |= {record_id for shard in work.state.workspace.all_shards() if shard.key not in updated
                for record_id in shard.item_ids()}
    offered = newcomers(work, ratings, bonus)
    csm = work.state.csm
    ids = set(csm.items) | {item.record_id for item in offered} | {
        record_id for shard in shards for record_id in shard.item_ids()}
    context = match_context(work)
    requirements = workspace_requirements()
    selection = SelectionInput(
        tick=provisional_tick(work), shards=shards,
        requirement_sets={kind: requirements.shards[kind] for kind in requirements.shards},
        newcomers=offered, activation={record_id: item.activation for record_id, item in csm.items.items()},
        tokens={record_id: record_tokens(work.records, record_id) for record_id in ids},
        blocked=sorted(blocked), focus_ids=sorted(record_id for record_id, value in bonus.items() if value > 0.0))
    changes = select_shards(selection, lambda requirement, record_id, key: selector_matches(
        requirement.select, work.records[record_id], key, context),
        lambda key, record_id: admits(work, key, record_id, updated))
    for change in changes:
        change.tail_ids = list(work.state.workspace.shard(change.key).tail_ids)
    return changes


def reinforce_entries(work: CycleWork, result: AttentionResult) -> list[str]:
    """Raise the base-level activation of each codelet whose percept entered a shard. Return the names."""
    names = list(dict.fromkeys(work.records[item.record_id].source for change in result.changes
                               for item in change.entered
                               if work.records[item.record_id].record_class == "CodeletPercept"))
    for name in names:
        reinforce(work.state.codelets.runtime(name))
    return names


def run(context, work: CycleWork) -> CycleWork:
    prompts = phase_prompts("attention")
    # The prefix reads the workspace and the feeling only. The focus check
    # changes neither, so the rating calls get the same prefix bytes.
    prefix = StoryPrefix(work)
    focus = resolve_focus(context, work.request.active_focus, work.request.focus_proposals, focus_evidence(work),
                          prefix.turns)
    danger = work.appraisal.general().bodily_threat >= THREAT_APPRAISAL_THRESHOLD
    bonus = apply_focus(work, focus, Embedder(context, work.records), danger=danger)
    structures = work.structures.structures
    jobs = [shard_job(work, prefix, prompts, shard, structures) for shard in shown_shards(work)] if structures else []
    answers = ordered_parallel_map(lambda job: rate(context, work, prefix, job), jobs,
                                   workers=ATTENTION_WORKERS, cancel=context.runtime.stop_event)
    ratings = [scored(work, job, answer, structures) for job, answer in zip(jobs, answers, strict=True)]
    result = AttentionResult(ratings=ratings, focus=focus, calls=len(jobs),
                             focus_ids=sorted(record_id for record_id, value in bonus.items() if value > 0.0))
    result.changes = select(work, ratings, bonus)
    work.attention = result
    reinforced = reinforce_entries(work, result)
    entered = sum(len(change.entered) for change in result.changes)
    left = sum(len(change.left) for change in result.changes)
    items = sum(len(work.state.workspace.shard(change.key).items) for change in result.changes)
    top = max((rated.score for shard in ratings for rated in shard.ratings), default=0.0)
    work.receipt("attention", f"rated {len(structures)} structures for {len(jobs)} shards (best {top:.2f}); "
                              f"{entered} items entered, {left} left, {items} present; "
                              f"reinforced {', '.join(reinforced) or 'no codelet'}")
    return work


__all__ = [
    "DELIBERATE_FOCUS",
    "FocusEvidence",
    "admits",
    "record_home",
    "ShardJob",
    "apply_focus",
    "attention_narrative",
    "aperture_text",
    "aux_rating",
    "winner_count",
    "focus_evidence",
    "home_shard",
    "newcomers",
    "rate",
    "reinforce_entries",
    "render_structures",
    "run",
    "scored",
    "select",
    "shard_job",
]

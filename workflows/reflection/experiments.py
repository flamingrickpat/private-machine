"""Commit experiment history beside the accepted change it describes.

The ledger uses existing Feature records and the MemoryStore checkpoint.
Every lifecycle step checks its source revisions. No helper commits or sends
an event. The private host owns cancellation, persistence, and publication.
"""

from pm import vclock

from datetime import timedelta

from pm.memory.snapshot import GhostConflict, Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.base import RecordBase, Provenance
from pm.records.memory_records import Feature, Narrative, Stimulus
from pm.introspection.episodes import EpisodeIndex
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType, SourceKind
from workflows.cognitive_control.attention import focus_versions, stage_focus
from workflows.cognitive_control.schemas import FocusState
from workflows.reflection.schemas import (EffectiveInputs, ExperimentPlan, ReflectionDecision,
    ExperimentMetadata, ExperimentEvidence, ExperimentEvent, ExperimentState, ExperimentResult,
    AttentionMetadata, ObservationGrant, PreviousIntervention, RecordArchitecture, ReflectionContext, ReflectionEpisode, ReflectionSource)

SCHEMA = "pm.reflection-experiment.v1"
OPEN_STATES = {"proposed", "applied", "awaiting_evidence", "evaluated"}


def architecture(record: RecordBase) -> RecordArchitecture:
    """Project known architecture fields at the generic metadata boundary."""
    return RecordArchitecture.model_validate(record.metadata)


def experiment_metadata(view: Snapshot, record: Feature) -> ExperimentMetadata:
    """Read a ledger and resolve the proposal identity of a trusted legacy chain."""
    data = ExperimentMetadata.model_validate(record.metadata)
    if data.experiment_id is not None:
        return data
    predecessor: RecordBase = record
    while predecessor.supersedes_id is not None:
        predecessor = view.get(predecessor.supersedes_id)
    data.experiment_id = predecessor.id
    return data


def experiments(view: Snapshot) -> list[Feature]:
    """Return current experiment records in creation order."""
    selected: list[Feature] = []
    for record in view.all_knoxels.values():
        if isinstance(record, Feature) and architecture(record).schema_version == SCHEMA:
            selected.append(record)
    return sorted(selected, key=lambda record: record.id)


def context(view: Snapshot) -> ReflectionContext:
    """Build frozen typed evidence. Keep full ledgers separate from rendering."""
    records = view.all_knoxels
    ledger = [(record, experiment_metadata(view, record)) for record in experiments(view)]
    episodes = [ReflectionEpisode.model_validate(episode.to_dict()) for episode in EpisodeIndex(view).recent(limit=6)]
    opened = [(record, data) for record, data in ledger if data.state in OPEN_STATES][:8]
    terminal = [(record, data) for record, data in ledger if data.state not in OPEN_STATES][-4:]
    selected = [*opened, *terminal]
    source_ids: set[int] = set()
    for episode in episodes:
        source_ids.update(episode.inputs)
        source_ids.update(episode.outputs)
        for links in episode.links.values():
            source_ids.update(links)
    for record, data in selected:
        source_ids.add(record.id)
        source_ids.update(data.source_revisions)
        source_ids.update(data.observation_ids)
        if data.effective_record_id is not None:
            source_ids.add(data.effective_record_id)
        if data.plan.intervention.expected_record_id is not None:
            source_ids.add(data.plan.intervention.expected_record_id)
    focus = focus_versions(view)
    if focus:
        source_ids.add(max(focus))
    recent: list[RecordBase] = []
    observations: list[RecordBase] = []
    for record in records.values():
        details = architecture(record)
        dialogue = isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue
        if details.operation == "action_selection" or details.origin == "appraisal" or dialogue or isinstance(record, BeliefRecord) and record.active:
            recent.append(record)
        if details.reflection_versions or dialogue and (record.provenance is None or record.provenance.source_kind != SourceKind.Self):
            observations.append(record)
    source_ids.update(record.id for record in sorted(recent, key=lambda record: record.id)[-24:])
    turns = sorted(view.gwt_turns.values(), key=lambda turn: turn.tick_id)
    for turn in [turn for turn in turns if turn.slot_id == "main"][-8:]:
        source_ids.update(key for key in [turn.decision_id, turn.experience_id, *turn.stimulus_ids] if key is not None and key in records)
    newest = max((record.id for record in observations), default=0)
    due = any(data.state in {"proposed", "applied"} or data.state in {"awaiting_evidence", "evaluated"}
              and newest > data.reviewed_source_id for _, data in selected)
    rows = [source_evidence(view, view.get(key)) for key in sorted(source_ids)
            if key in view.revisions]
    entries: list[ExperimentEvidence] = []
    for record, data in selected:
        assert data.experiment_id is not None
        entry = ExperimentEvidence(record_id=record.id, experiment_id=data.experiment_id, revision=view.revisions[record.id],
            stale_source_ids=[key for key, revision in data.source_revisions.items() if view.revisions.get(key) != revision or key in data.head_source_ids and key not in view.all_knoxels],
            observation_grant=observation_grant(data.experiment_id, data, rows), metadata=data)
        entries.append(entry)
    return ReflectionContext(experiments=entries, omitted_experiments=len(ledger) - len(selected),
        attention_head_id=max(focus) if focus else None, review_due=due, newest_observation_id=newest,
        episodes=episodes, episode_limits=["The frozen record grant excludes managed-run receipts."], records=rows)


def effective_inputs(view: Snapshot, focus: FocusState | None) -> EffectiveInputs:
    """Return exact experiment versions and original sources used by a main cycle."""
    from workflows.main_cognition.schemas import ReflectionGuidance
    versions: dict[str, int] = {}
    sources: set[int] = set()
    guidance: list[ReflectionGuidance] = []
    control_record_id = focus.control_record_id if focus is not None else None
    for record in experiments(view):
        data = experiment_metadata(view, record)
        assert data.experiment_id is not None
        effective_id = data.reversal_record_id or data.effective_record_id
        if effective_id is None:
            continue
        target = view.get(effective_id)
        owner = data.plan.intervention.owner
        if owner == "attention" and control_record_id != effective_id:
            continue
        if owner == "belief" and (not isinstance(target, BeliefRecord) or not target.active):
            continue
        versions[str(data.experiment_id)] = effective_id
        sources.add(effective_id)
        if isinstance(target, BeliefRecord):
            sources.update(target.supporting_knoxel_ids)
            sources.update(target.contradicting_knoxel_ids)
        if owner == "attention":
            assert focus is not None
            instruction = focus.instruction
        else:
            instruction = target.content
        guidance.append(ReflectionGuidance(experiment_id=data.experiment_id, owner=owner, effective_record_id=effective_id,
            scope=data.plan.scope, instruction=instruction, status=data.state))
    return EffectiveInputs(versions=versions, source_ids=sorted(sources), guidance=guidance)


def source_evidence(view: Snapshot, record: RecordBase) -> ReflectionSource:
    """Project original content and named architecture fields without dumping metadata."""
    from pm.memory.references import referenced_ids
    from pm.memory.retrieval import evidence_role
    source = record.source if isinstance(record, (Feature, Narrative, Stimulus)) else ""
    evidence = ReflectionSource(knoxel_id=record.id, revision=view.revisions[record.id], tick_id=record.tick_id,
        text=record.content, kind=type(record).__name__, metadata=architecture(record),
        reference_ids=sorted(referenced_ids(record)), evidence_role=evidence_role(view, record.as_payload(include_embedding=False)),
        source=source, provenance=record.provenance)
    if isinstance(record, BeliefRecord):
        evidence.belief_kind = record.belief_kind
        evidence.active = record.active
        evidence.supporting_ids = list(record.supporting_knoxel_ids)
        evidence.opposing_ids = list(record.contradicting_knoxel_ids)
    return evidence


def observation_grant(experiment_id: int, data: ExperimentMetadata, rows: list[ReflectionSource]) -> ObservationGrant:
    """Grant only later observations with explicit consumption links."""
    if data.effective_record_id is None:
        return ObservationGrant()
    consumers = {row.knoxel_id for row in rows if row.tick_id > data.applied_tick
                 and row.metadata.reflection_versions.get(experiment_id) == data.effective_record_id}
    original = set(data.source_revisions)
    eligible = {row.knoxel_id for row in rows if row.tick_id > data.applied_tick and row.knoxel_id not in original
                and (row.knoxel_id in consumers or row.evidence_role in {"user_report", "runtime_event", "world_observation"}
                     and consumers.intersection(row.reference_ids))}
    return ObservationGrant(consumer_ids=sorted(consumers), observation_ids=sorted(eligible))


def _plan_ids(plan: ExperimentPlan) -> set[int]:
    """Collect record references without confusing an episode ID with a record ID."""
    ids: set[int] = set()
    for hypothesis in plan.hypotheses:
        ids.update(hypothesis.supporting_ids)
        ids.update(hypothesis.opposing_ids)
    change = plan.intervention
    if change.expected_record_id is not None:
        ids.add(change.expected_record_id)
    if change.focus:
        ids.update(change.focus.promote_ids)
        ids.update(change.focus.reduce_ids)
    return ids


def _episode_sources(plan: ExperimentPlan, payload: ReflectionContext) -> set[int]:
    """Resolve a supplied A18 episode to its original inputs, outputs, and links."""
    episodes = [item for item in payload.episodes if item.episode_id == plan.triggering_episode_id]
    if not episodes:
        raise ValueError("The triggering episode must name an episode_id from the supplied episodes, not a tick or source ID.")
    episode = next((item for item in episodes if item.kind != "gwt_turn"), episodes[0])
    return set(episode.inputs) | set(episode.outputs) | {key for values in episode.links.values() for key in values}


def validate_decision(decision: ReflectionDecision, payload: ReflectionContext) -> None:
    """Reject invented evidence or a lifecycle step outside the supplied grant."""
    known = {item.knoxel_id for item in payload.records}
    if decision.action == "none":
        return
    if not decision.reason.strip():
        raise ValueError("A reflection step needs a reason.")
    if decision.action == "propose":
        if decision.plan is None or decision.experiment_id is not None:
            raise ValueError("A new experiment needs a plan and no existing experiment ID.")
        missing = (_plan_ids(decision.plan) | _episode_sources(decision.plan, payload)) - known
        if missing:
            raise ValueError(f"An experiment cites sources outside its frozen grant: {sorted(missing)}.")
        hypotheses = decision.plan.hypotheses
        if not hypotheses[0].supporting_ids or len({item.explanation.casefold() for item in hypotheses}) < 2:
            raise ValueError("An experiment needs sourced support and a distinct alternative explanation.")
    else:
        if decision.plan is not None and decision.action != "revise":
            raise ValueError("An existing prediction cannot change during a lifecycle step.")
        experiment = next((item for item in payload.experiments
                           if item.experiment_id == decision.experiment_id), None)
        if experiment is None:
            raise ValueError("The experiment is outside the frozen grant.")
        if set(decision.observation_ids) - known:
            raise ValueError("An evaluation cites evidence outside its frozen grant.")
        allowed_states = {"apply": {"proposed"}, "await": {"applied", "awaiting_evidence"},
            "evaluate": {"applied", "awaiting_evidence", "evaluated"}, "retain": {"evaluated"},
            "reverse": {"applied", "awaiting_evidence", "evaluated", "retained"},
            "revise": {"evaluated", "reversed", "interrupted"}, "interrupt": OPEN_STATES}
        _require(experiment.metadata, allowed_states[decision.action])
        if decision.action == "evaluate":
            grant = observation_grant(experiment.experiment_id, experiment.metadata, payload.records)
            _validate_outcome(decision, grant)
        if decision.action == "revise":
            validate_decision(ReflectionDecision(action="propose", plan=decision.plan, reason=decision.reason), payload)


def _event(data: ExperimentMetadata, batch: WriteBatch, action: ExperimentState, reason: str, *,
           effective_record_id: int | None = None, reversal_record_id: int | None = None,
           replacement_experiment_id: int | None = None, result: ExperimentResult = "unresolved",
           observation_ids: list[int] | None = None) -> None:
    """Append one typed transition. The owner commits it with the intervention."""
    data.history.append(ExperimentEvent(action=action, reason=reason, tick_id=batch.tick_id,
        at=vclock.utcnow().isoformat(), effective_record_id=effective_record_id,
        reversal_record_id=reversal_record_id, replacement_experiment_id=replacement_experiment_id,
        result=result, observation_ids=[] if observation_ids is None else observation_ids))


def stage_decision(batch: WriteBatch, view: Snapshot, decision: ReflectionDecision, payload: ReflectionContext, *,
                   companion_name: str, maximum_focus_bias: float = 0.35,
                   revised_from_experiment_id: int | None = None) -> Feature | None:
    """Stage one experiment transition and any owned change in the same batch.

    ValueError rejects an invalid transition. GhostConflict rejects stale state.
    A cancelled caller can commit neither the intervention nor its receipt.
    Proposed predictions survive cancellation of a later application attempt.
    """
    validate_decision(decision, payload)
    if decision.action == "none":
        return None
    supplied = {item.knoxel_id: item.revision for item in payload.records}
    if decision.action == "propose":
        plan = decision.plan
        assert plan is not None
        ids = _plan_ids(plan) | _episode_sources(plan, payload)
        for key in ids:
            batch.expect_revision(key, supplied[key])
        previous = _capture_previous(view, plan)
        assert plan is not None
        data = ExperimentMetadata(state="proposed", plan=plan, previous=previous,
            revised_from_experiment_id=revised_from_experiment_id,
            source_revisions={key: supplied[key] for key in sorted(ids)},
            head_source_ids=sorted(key for key in ids if key in view.all_knoxels))
        _event(data, batch, "proposed", decision.reason)
        data.executed_at = vclock.utcnow().isoformat()
        proposal = batch.create(Feature(content=plan.finding, feature_type=FeatureType.StateUpdate, causal=True,
            override_source=companion_name, provenance=Provenance(source_kind=SourceKind.Self, source_ref=companion_name),
            metadata=data.model_dump(mode="json", by_alias=True)))
        data.experiment_id = proposal.id
        proposal.metadata = data.model_dump(mode="json", by_alias=True)
        return proposal
    assert decision.experiment_id is not None
    entry = next(item for item in payload.experiments if item.experiment_id == decision.experiment_id)
    record = view.get(entry.record_id)
    assert isinstance(record, Feature)
    batch.expect_head(record.id, supplied[record.id])
    data = experiment_metadata(view, record)
    assert data.experiment_id is not None
    data.reviewed_source_id = payload.newest_observation_id
    action = decision.action
    plan = data.plan
    if action == "apply":
        _require(data, {"proposed"})
        for key, revision in data.source_revisions.items():
            batch.expect_revision(int(key), revision)
        for key in data.head_source_ids:
            batch.expect_head(key, data.source_revisions[key])
        effective = _apply(batch, view, data.experiment_id, plan, data, companion_name, maximum_focus_bias)
        data.state = "applied"
        data.effective_record_id = effective.id
        data.applied_tick = batch.tick_id
        _event(data, batch, "applied", decision.reason, effective_record_id=effective.id)
    elif action == "await":
        _require(data, {"applied", "awaiting_evidence"})
        data.state = "awaiting_evidence"
        _event(data, batch, "awaiting_evidence", decision.reason)
    elif action == "evaluate":
        _require(data, {"applied", "awaiting_evidence", "evaluated"})
        consumer_ids = _validate_observations(view, data.experiment_id, data, decision)
        for key in consumer_ids:
            batch.expect_revision(key, view.revisions[key])
        for key in decision.observation_ids:
            batch.expect_revision(key, supplied[key])
        data.observation_ids = sorted(set(data.observation_ids) | set(decision.observation_ids))
        data.state = "evaluated"
        data.result = decision.result
        _event(data, batch, "evaluated", decision.reason, result=decision.result,
               observation_ids=list(decision.observation_ids))
    elif action == "retain":
        _require(data, {"evaluated"})
        supported = {key for event in data.history if event.action == "evaluated"
                     and event.result == "supported" for key in event.observation_ids}
        if data.result != "supported" or len({view.get(key).tick_id for key in supported}) < 2:
            raise ValueError("Strategy retention needs support from at least two later observations.")
        data.state = "retained"
        _event(data, batch, "retained", decision.reason)
    elif action == "reverse":
        _require(data, {"applied", "awaiting_evidence", "evaluated", "retained"})
        restored = _reverse(batch, view, data.experiment_id, data, plan, companion_name)
        data.state = "reversed"
        data.reversal_record_id = restored.id
        _event(data, batch, "reversed", decision.reason, reversal_record_id=restored.id)
    elif action == "revise":
        _require(data, {"evaluated", "reversed", "interrupted"})
        assert decision.plan is not None
        if record.id in _plan_ids(decision.plan):
            raise ValueError("A revised hypothesis must cite original observations. Its failed parent is linked automatically.")
        replacement = stage_decision(batch, view, ReflectionDecision(action="propose", plan=decision.plan,
            reason=decision.reason), payload, companion_name=companion_name, maximum_focus_bias=maximum_focus_bias,
            revised_from_experiment_id=data.experiment_id)
        assert replacement is not None
        data.state = "revised"
        data.replacement_experiment_id = replacement.id
        _event(data, batch, "revised", decision.reason, replacement_experiment_id=replacement.id)
    elif action == "interrupt":
        _require(data, OPEN_STATES)
        data.state = "interrupted"
        _event(data, batch, "interrupted", decision.reason,
               effective_record_id=data.effective_record_id)
    successor = record.copy_for_edit()
    successor.metadata = data.model_dump(mode="json", by_alias=True)
    return batch.replace(successor)


def _require(data: ExperimentMetadata, states: set[str]) -> None:
    if data.state not in states:
        raise ValueError(f"Experiment state {data.state} does not permit this transition.")


def _capture_previous(view: Snapshot, plan: ExperimentPlan) -> PreviousIntervention:
    """Capture reversal data or reject a changed owner and an impermissible target."""
    change = plan.intervention
    if change.owner == "attention":
        versions = focus_versions(view)
        head = max(versions) if versions else None
        if change.expected_record_id != head:
            raise GhostConflict("Attention advice changed before the experiment proposal.")
        previous = AttentionMetadata.model_validate(view.get(head).metadata) if head is not None else AttentionMetadata()
        return PreviousIntervention(record_id=head, focus=previous.focus, expires_at=previous.expires_at)
    assert change.expected_record_id is not None
    record = view.get(change.expected_record_id)
    from workflows.main_cognition.cycle_commits import subjective_belief
    if not subjective_belief(record):
        raise ValueError("The belief owner permits only active subjective interpretations.")
    assert isinstance(record, BeliefRecord)
    return PreviousIntervention(record_id=record.id, text=record.content,
        supporting_ids=list(record.supporting_knoxel_ids), opposing_ids=list(record.contradicting_knoxel_ids))


def _apply(batch: WriteBatch, view: Snapshot, experiment_id: int, plan: ExperimentPlan,
           data: ExperimentMetadata, companion: str, maximum_bias: float) -> RecordBase:
    """Stage the owned intervention. The caller commits its receipt in the same batch."""
    change = plan.intervention
    if change.owner == "attention":
        _capture_previous(view, plan)
        assert change.focus is not None
        focus = FocusState.model_validate(change.focus.model_dump())
        focus.accept = True
        focus.fresh = False
        focus.strength = min(focus.strength, maximum_bias)
        return stage_focus(batch, view, instruction=focus.instruction, focus=focus.model_dump(mode="json"),
            companion_name=companion, experiment_id=experiment_id,
            expires_at=(vclock.utcnow() + timedelta(seconds=focus.seconds)).isoformat())
    from workflows.main_cognition.cycle_commits import revise_reflected_belief
    assert change.expected_record_id is not None
    belief = view.get(change.expected_record_id)
    assert isinstance(belief, BeliefRecord)
    return revise_reflected_belief(batch, belief, text=change.belief,
        evidence_ids=plan.hypotheses[0].supporting_ids, opposing_ids=plan.hypotheses[0].opposing_ids,
        experiment_id=experiment_id, companion_name=companion)


def _reverse(batch: WriteBatch, view: Snapshot, experiment_id: int, data: ExperimentMetadata,
             plan: ExperimentPlan, companion: str) -> RecordBase:
    """Restore the captured state as a new version without crossing a newer owner change."""
    effective_id = data.effective_record_id
    assert effective_id is not None
    batch.expect_revision(effective_id, view.revisions[effective_id])
    previous = data.previous
    if plan.intervention.owner == "attention":
        versions = focus_versions(view)
        if max(versions, default=None) != effective_id:
            raise GhostConflict("Newer attention advice prevents reversal of this experiment.")
        return stage_focus(batch, view, instruction=previous.focus.instruction or "Restore prior attention advice.",
            focus=previous.focus.model_dump(mode="json"), companion_name=companion, experiment_id=experiment_id,
            expires_at=previous.expires_at or vclock.utcnow().isoformat())
    from workflows.main_cognition.cycle_commits import revise_reflected_belief
    belief = view.get(effective_id)
    assert isinstance(belief, BeliefRecord)
    return revise_reflected_belief(batch, belief, text=previous.text,
        evidence_ids=previous.supporting_ids, opposing_ids=previous.opposing_ids,
        experiment_id=experiment_id, companion_name=companion)


def _validate_observations(view: Snapshot, experiment_id: int, data: ExperimentMetadata, decision: ReflectionDecision) -> list[int]:
    """Require later evidence and explicit consumption links. Return the guarded consumer IDs."""
    payload = context(view)
    grant = observation_grant(experiment_id, data, payload.records)
    _validate_outcome(decision, grant)
    return grant.consumer_ids


def _validate_outcome(decision: ReflectionDecision, grant: ObservationGrant) -> None:
    """Apply the same observation rules during model repair and host acceptance."""
    if decision.result != "unresolved" and not decision.observation_ids:
        raise ValueError("Missing evidence cannot support or contradict a prediction.")
    if decision.result != "unresolved" and not grant.consumer_ids:
        raise ValueError("No later decision consumed the experiment's effective version.")
    invalid = set(decision.observation_ids) - set(grant.observation_ids)
    if invalid:
        raise ValueError(f"Outcome IDs {sorted(invalid)} are not eligible later observations. "
                         f"Allowed observation IDs: {grant.observation_ids}. "
                         "An external outcome needs an explicit link to a consuming decision. "
                         "Original evidence can support a hypothesis but cannot establish its later outcome.")

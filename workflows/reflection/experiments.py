"""Commit experiment history beside the accepted change it describes.

The ledger uses existing Feature records and the MemoryStore checkpoint.
Every lifecycle step checks its source revisions. No helper commits or sends
an event. The private host owns cancellation, persistence, and publication.
"""

from pm import vclock

from pm.records.copying import pickle_copy
from datetime import datetime, timedelta

from pm.memory.snapshot import GhostConflict
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from workflows.cognitive_control.attention import focus_versions, stage_focus
from workflows.cognitive_control.schemas import FocusState
from workflows.reflection.schemas import EffectiveInputs, ExperimentPlan, ReflectionDecision

SCHEMA = "pm.reflection-experiment.v1"
OPEN_STATES = {"proposed", "applied", "awaiting_evidence", "evaluated"}


def experiments(view) -> list:
    """Return experiments in creation order, including failed and reversed work."""
    return sorted((item for item in view.all_knoxels.values()
                   if item.metadata.get("schema") == SCHEMA), key=lambda item: item.id)


def context(view) -> dict:
    """Select pending experiments and their exact sources from a frozen view.

    At most eight open experiments and four terminal experiments enter a call.
    The omitted count is explicit. Full histories remain in the checkpoint.
    """
    records = view.all_knoxels
    all_items = experiments(view)
    from pm.introspection.episodes import EpisodeIndex
    index = EpisodeIndex(view)
    episodes = [item.to_dict() for item in index.recent(limit=6)]
    selected = ([item for item in all_items if item.metadata["state"] in OPEN_STATES][:8]
                + [item for item in all_items if item.metadata["state"] not in OPEN_STATES][-4:])
    ids = set()
    for episode in episodes:
        ids.update(episode["inputs"])
        ids.update(episode["outputs"])
        for linked in episode["links"].values():
            ids.update(linked)
    for item in selected:
        ids.add(item.id)
        ids.update(int(key) for key in item.metadata["source_revisions"])
        ids.update(item.metadata.get("observation_ids", []))
        for key in (item.metadata.get("effective_record_id"), item.metadata["plan"]["intervention"]["expected_record_id"]):
            if key is not None:
                ids.add(key)
    focus = focus_versions(view)
    if focus:
        ids.add(max(focus))
    # These are retained execution records, not reconstructed explanations.
    recent = [item for item in records.values() if item.metadata.get("operation") == "action_selection"
              or item.metadata.get("origin") == "appraisal"
              or getattr(item, "feature_type", None) == FeatureType.Dialogue
              or isinstance(item, BeliefRecord) and item.active]
    ids.update(item.id for item in sorted(recent, key=lambda item: item.id)[-24:])
    # The newest main turns name their decision record, their subjective
    # experience, and their claimed stimuli. Read them typed.
    from pm.records.global_workspace import GwtTurnRecord
    turns = sorted((GwtTurnRecord.model_validate(raw) for raw in view.serialize_gwt().values()),
                   key=lambda turn: turn.tick_id)
    for turn in [turn for turn in turns if turn.slot_id == "main"][-8:]:
        ids.update(key for key in [turn.decision_id, turn.experience_id, *turn.stimulus_ids]
                   if key is not None and key in records)
    observations = [item for item in records.values()
                    if item.metadata.get("reflection_versions")
                    or getattr(item, "feature_type", None) == FeatureType.Dialogue
                    and (item.provenance is None or item.provenance.source_kind != SourceKind.Self)]
    newest = max((item.id for item in observations), default=0)
    due = any(item.metadata["state"] in {"proposed", "applied"}
              or item.metadata["state"] in {"awaiting_evidence", "evaluated"}
              and newest > item.metadata.get("reviewed_source_id", 0) for item in selected)
    rows = [source_payload(view, records[key]) for key in sorted(ids) if key in records]
    return {"experiments": [{"record_id": item.id, "revision": view.revisions[item.id],
                              "stale_source_ids": [int(key) for key, revision in item.metadata["source_revisions"].items()
                                                   if view.revisions.get(int(key)) != revision],
                              **pickle_copy(item.metadata),
                              "observation_grant": observation_grant(item.id, item.metadata, rows)} for item in selected],
            "omitted_experiments": len(all_items) - len(selected),
            "attention_head_id": max(focus) if focus else None,
            "review_due": due, "newest_observation_id": newest,
            "episodes": episodes,
            "episode_limits": ["The frozen record grant excludes managed-run receipts."],
            "records": rows}


def effective_inputs(view, focus: FocusState | None) -> EffectiveInputs:
    """Return effective experiment versions and original sources for main cognition.

    ``focus`` is the advice in effect (``active_focus``), or None. Expired or
    replaced attention advice contributes nothing. Inactive beliefs
    contribute nothing. Lifecycle interruption never pretends to undo a change.
    """
    from workflows.main_cognition.schemas import ReflectionGuidance
    records = view.all_knoxels
    versions, sources, guidance = {}, set(), []
    control_record_id = focus.control_record_id if focus is not None else None
    for item in experiments(view):
        effective_id = item.metadata.get("reversal_record_id") or item.metadata.get("effective_record_id")
        target = records.get(effective_id)
        if target is None:
            continue
        owner = item.metadata["plan"]["intervention"]["owner"]
        if owner == "attention" and control_record_id != effective_id:
            continue
        if owner == "belief" and (not isinstance(target, BeliefRecord) or not target.active):
            continue
        versions[str(item.id)] = effective_id
        sources.add(effective_id)
        if owner == "belief":
            sources.update(target.supporting_knoxel_ids)
            sources.update(target.contradicting_knoxel_ids)
        guidance.append(ReflectionGuidance(
            experiment_id=item.id, owner=owner, effective_record_id=effective_id,
            scope=item.metadata["plan"]["scope"],
            instruction=focus.instruction if owner == "attention" else target.content,
            status=item.metadata["state"]))
    return EffectiveInputs(versions=versions, source_ids=sorted(sources), guidance=guidance)


def source_payload(view, record) -> dict:
    """Expose original content, provenance, and execution metadata with a revision."""
    from pm.memory.references import referenced_ids
    from pm.memory.retrieval import evidence_role
    return {"knoxel_id": record.id, "revision": view.revisions[record.id],
            "tick_id": record.tick_id, "text": str(record.content),
            "kind": type(record).__name__, "metadata": pickle_copy(record.metadata),
            "reference_ids": sorted(referenced_ids(record)),
            "evidence_role": evidence_role(view, (record).as_payload()),
            "source": str(getattr(record, "source", "")),
            "provenance": record.provenance.model_dump(mode="json") if record.provenance else None,
            **({"belief_kind": record.belief_kind, "active": record.active,
                "supporting_ids": record.supporting_knoxel_ids,
                "opposing_ids": record.contradicting_knoxel_ids} if isinstance(record, BeliefRecord) else {})}


def observation_grant(experiment_id: int, data: dict, rows: list[dict]) -> dict:
    """Identify eligible later records without judging the prediction's meaning.

    Original sources and earlier episodes cannot become outcome evidence.
    An external report needs an explicit reference to a consuming decision.
    The model still assesses relevance, support, contradiction, and uncertainty.
    """
    if data.get("effective_record_id") is None:
        return {"consumer_ids": [], "observation_ids": []}
    consumers = {row["knoxel_id"] for row in rows
                 if row["tick_id"] > data["applied_tick"]
                 and row["metadata"].get("reflection_versions", {}).get(str(experiment_id)) == data["effective_record_id"]}
    original = {int(key) for key in data["source_revisions"]}
    eligible = {row["knoxel_id"] for row in rows if row["tick_id"] > data["applied_tick"]
                and row["knoxel_id"] not in original
                and (row["knoxel_id"] in consumers or row["evidence_role"] in {
                    "user_report", "runtime_event", "world_observation"}
                    and consumers.intersection(row["reference_ids"]))}
    return {"consumer_ids": sorted(consumers), "observation_ids": sorted(eligible)}


def _plan_ids(plan: ExperimentPlan) -> set[int]:
    """Collect record references without confusing an episode ID with a record ID."""
    ids = set()
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


def _episode_sources(plan: ExperimentPlan, payload: dict) -> set[int]:
    """Resolve a supplied A18 episode to its original inputs, outputs, and links."""
    episodes = [item for item in payload.get("episodes", []) if item["episode_id"] == plan.triggering_episode_id]
    if not episodes:
        raise ValueError("The triggering episode must name an episode_id from the supplied episodes, not a tick or source ID.")
    episode = next((item for item in episodes if item["kind"] != "gwt_turn"), episodes[0])
    return set(episode["inputs"]) | set(episode["outputs"]) | {key for values in episode["links"].values() for key in values}


def validate_decision(decision: ReflectionDecision, payload: dict) -> None:
    """Reject invented evidence or a lifecycle step outside the supplied grant."""
    known = {item["knoxel_id"] for item in payload.get("records", [])}
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
        experiment = next((item for item in payload.get("experiments", [])
                           if item["record_id"] == decision.experiment_id), None)
        if experiment is None:
            raise ValueError("The experiment is outside the frozen grant.")
        if set(decision.observation_ids) - known:
            raise ValueError("An evaluation cites evidence outside its frozen grant.")
        allowed_states = {"apply": {"proposed"}, "await": {"applied", "awaiting_evidence"},
            "evaluate": {"applied", "awaiting_evidence", "evaluated"}, "retain": {"evaluated"},
            "reverse": {"applied", "awaiting_evidence", "evaluated", "retained"},
            "revise": {"evaluated", "reversed", "interrupted"}, "interrupt": OPEN_STATES}
        _require(experiment, allowed_states[decision.action])
        if decision.action == "evaluate":
            grant = observation_grant(decision.experiment_id, experiment, payload["records"])
            _validate_outcome(decision, grant)
        if decision.action == "revise":
            validate_decision(ReflectionDecision(action="propose", plan=decision.plan, reason=decision.reason), payload)


def _event(data: dict, batch, action: str, reason: str, **facts) -> None:
    """Append a lifecycle receipt without replacing earlier predictions or results."""
    data["history"].append({"action": action, "reason": reason, "tick_id": batch.tick_id,
                            "at": vclock.utcnow().isoformat(), **facts})


def stage_decision(batch, view, decision: ReflectionDecision, payload: dict, *,
                   companion_name: str, maximum_focus_bias: float = 0.35,
                   revised_from_experiment_id: int | None = None):
    """Stage one experiment transition and any owned change in the same batch.

    ValueError rejects an invalid transition. GhostConflict rejects stale state.
    A cancelled caller can commit neither the intervention nor its receipt.
    Proposed predictions survive cancellation of a later application attempt.
    """
    validate_decision(decision, payload)
    if decision.action == "none":
        return None
    supplied = {item["knoxel_id"]: item["revision"] for item in payload["records"]}
    if decision.action == "propose":
        plan = decision.plan
        ids = _plan_ids(plan) | _episode_sources(plan, payload)
        for key in ids:
            batch.expect_revision(key, supplied[key])
        previous = _capture_previous(view, plan)
        data = {"schema": SCHEMA, "state": "proposed", "plan": plan.model_dump(mode="json"),
                "revised_from_experiment_id": revised_from_experiment_id,
                "source_revisions": {str(key): supplied[key] for key in sorted(ids)},
                "previous": previous, "effective_record_id": None,
                "observation_ids": [], "result": "unresolved", "history": []}
        _event(data, batch, "proposed", decision.reason)
        return new_feature(batch._ghost, plan.finding, FeatureType.StateUpdate,
            source=companion_name, source_kind=SourceKind.Self, changes=batch, metadata=data)
    record = view.get(decision.experiment_id)
    batch.expect_revision(record.id, supplied[record.id])
    data = pickle_copy(record.metadata)
    data["reviewed_source_id"] = payload.get("newest_observation_id", 0)
    action = decision.action
    plan = ExperimentPlan.model_validate(data["plan"])
    if action == "apply":
        _require(data, {"proposed"})
        for key, revision in data["source_revisions"].items():
            batch.expect_revision(int(key), revision)
        effective = _apply(batch, view, record.id, plan, data, companion_name, maximum_focus_bias)
        data.update(state="applied", effective_record_id=effective.id, applied_tick=batch.tick_id)
        _event(data, batch, "applied", decision.reason, effective_record_id=effective.id)
    elif action == "await":
        _require(data, {"applied", "awaiting_evidence"})
        data["state"] = "awaiting_evidence"
        _event(data, batch, "awaiting_evidence", decision.reason)
    elif action == "evaluate":
        _require(data, {"applied", "awaiting_evidence", "evaluated"})
        consumer_ids = _validate_observations(view, record.id, data, decision)
        for key in consumer_ids:
            batch.expect_revision(key, view.revisions[key])
        for key in decision.observation_ids:
            batch.expect_revision(key, supplied[key])
        data["observation_ids"] = sorted(set(data["observation_ids"]) | set(decision.observation_ids))
        data.update(state="evaluated", result=decision.result)
        _event(data, batch, "evaluated", decision.reason, result=decision.result,
               observation_ids=list(decision.observation_ids))
    elif action == "retain":
        _require(data, {"evaluated"})
        supported = {key for event in data["history"] if event["action"] == "evaluated"
                     and event["result"] == "supported" for key in event["observation_ids"]}
        if data["result"] != "supported" or len({view.get(key).tick_id for key in supported}) < 2:
            raise ValueError("Strategy retention needs support from at least two later observations.")
        data["state"] = "retained"
        _event(data, batch, "retained", decision.reason)
    elif action == "reverse":
        _require(data, {"applied", "awaiting_evidence", "evaluated", "retained"})
        restored = _reverse(batch, view, record.id, data, plan, companion_name)
        data.update(state="reversed", reversal_record_id=restored.id)
        _event(data, batch, "reversed", decision.reason, reversal_record_id=restored.id)
    elif action == "revise":
        _require(data, {"evaluated", "reversed", "interrupted"})
        if record.id in _plan_ids(decision.plan):
            raise ValueError("A revised hypothesis must cite original observations. Its failed parent is linked automatically.")
        replacement = stage_decision(batch, view, ReflectionDecision(action="propose", plan=decision.plan,
            reason=decision.reason), payload, companion_name=companion_name, maximum_focus_bias=maximum_focus_bias,
            revised_from_experiment_id=record.id)
        data.update(state="revised", replacement_experiment_id=replacement.id)
        _event(data, batch, "revised", decision.reason, replacement_experiment_id=replacement.id)
    elif action == "interrupt":
        _require(data, OPEN_STATES)
        data["state"] = "interrupted"
        _event(data, batch, "interrupted", decision.reason,
               effective_record_id=data["effective_record_id"])
    batch.replace(record.model_copy(update={"metadata": data}))
    return record.model_copy(update={"metadata": data})


def _require(data: dict, states: set[str]) -> None:
    if data["state"] not in states:
        raise ValueError(f"Experiment state {data['state']} does not permit this transition.")


def _capture_previous(view, plan: ExperimentPlan) -> dict:
    """Capture reversal data or reject a changed owner and an impermissible target."""
    change = plan.intervention
    if change.owner == "attention":
        versions = focus_versions(view)
        head = max(versions) if versions else None
        if change.expected_record_id != head:
            raise GhostConflict("Attention advice changed before the experiment proposal.")
        previous = view.get(head).metadata if head is not None else {}
        return {"record_id": head, "focus": pickle_copy(previous.get("focus", {})),
                "expires_at": previous.get("expires_at")}
    record = view.get(change.expected_record_id)
    from workflows.main_cognition.cycle_commits import subjective_belief
    if not subjective_belief(record):
        raise ValueError("The belief owner permits only active subjective interpretations.")
    return {"record_id": record.id, "text": record.content,
            "supporting_ids": list(record.supporting_knoxel_ids),
            "opposing_ids": list(record.contradicting_knoxel_ids)}


def _apply(batch, view, experiment_id, plan, data, companion, maximum_bias):
    """Stage the owned intervention. The caller commits its receipt in the same batch."""
    change = plan.intervention
    if change.owner == "attention":
        _capture_previous(view, plan)
        focus = change.focus.model_dump(mode="json")
        focus.update(accept=True, fresh=False, strength=min(focus["strength"], maximum_bias))
        return stage_focus(batch, view, instruction=focus["instruction"], focus=focus,
            companion_name=companion, experiment_id=experiment_id,
            expires_at=(vclock.utcnow() + timedelta(seconds=focus["seconds"])).isoformat())
    from workflows.main_cognition.cycle_commits import revise_reflected_belief
    return revise_reflected_belief(batch, view.get(change.expected_record_id), text=change.belief,
        evidence_ids=plan.hypotheses[0].supporting_ids, opposing_ids=plan.hypotheses[0].opposing_ids,
        experiment_id=experiment_id, companion_name=companion)


def _reverse(batch, view, experiment_id, data, plan, companion):
    """Restore the captured state as a new version without crossing a newer owner change."""
    effective_id = data["effective_record_id"]
    batch.expect_revision(effective_id, view.revisions[effective_id])
    previous = data["previous"]
    if plan.intervention.owner == "attention":
        versions = focus_versions(view)
        if max(versions, default=None) != effective_id:
            raise GhostConflict("Newer attention advice prevents reversal of this experiment.")
        return stage_focus(batch, view, instruction=previous.get("focus", {}).get("instruction", "Restore prior attention advice."),
            focus=previous.get("focus", {}), companion_name=companion, experiment_id=experiment_id,
            expires_at=previous.get("expires_at") or vclock.utcnow().isoformat())
    from workflows.main_cognition.cycle_commits import revise_reflected_belief
    return revise_reflected_belief(batch, view.get(effective_id), text=previous["text"],
        evidence_ids=previous["supporting_ids"], opposing_ids=previous["opposing_ids"],
        experiment_id=experiment_id, companion_name=companion)


def _validate_observations(view, experiment_id, data, decision):
    """Require later evidence and explicit consumption links. Return the guarded consumer IDs."""
    payload = context(view)
    grant = observation_grant(experiment_id, data, payload["records"])
    _validate_outcome(decision, grant)
    return grant["consumer_ids"]


def _validate_outcome(decision: ReflectionDecision, grant: dict) -> None:
    """Apply the same observation rules during model repair and host acceptance."""
    if decision.result != "unresolved" and not decision.observation_ids:
        raise ValueError("Missing evidence cannot support or contradict a prediction.")
    if decision.result != "unresolved" and not grant["consumer_ids"]:
        raise ValueError("No later decision consumed the experiment's effective version.")
    invalid = set(decision.observation_ids) - set(grant["observation_ids"])
    if invalid:
        raise ValueError(f"Outcome IDs {sorted(invalid)} are not eligible later observations. "
                         f"Allowed observation IDs: {grant['observation_ids']}. "
                         "An external outcome needs an explicit link to a consuming decision. "
                         "Original evidence can support a hypothesis but cannot establish its later outcome.")

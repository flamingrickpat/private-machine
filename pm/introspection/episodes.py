"""Queryable cognitive episodes built from records the runtime already keeps.

A18 does not install a second tracing framework. An episode is a read-side
view over committed records that already exist: workspace turns, private
thoughts, action schemas, beliefs, expectations, and managed run receipts.
A main turn (``pm.gwt-turn.v2``) names its claimed stimuli, its shard
changes, its subjective experience, and its decision record. The view adds the causal fields the records carry, and it
refuses to infer an execution fact from timestamp order.

A private loop contributes episodes in both of its shapes: a linear inner
monologue (``loop: "inner"``) and an executed operation of the inquiry graph
(``loop: "inquiry"``, kind ``inquiry_operation``). The graph is the live private
path, so a mind that thinks in the graph is only inspectable, and only able to
notice its own recent thinking, when its operations are episode rows too. The
private workspace turn of that step stays its own row. That turn leaves
``decision_id`` empty, and the field names a main-cycle ``StateUpdate`` rather
than a private decision, so the two rows cannot be merged without inventing a
causal fact: a turn row reports the frame the mind held, while the decision row
carries the evidence, the outcome, and the earlier nodes it answered.

An episode stays queryable when it is incomplete. A cancelled cycle, a
rejected proposal, or an unresolved expectation is a first-class episode with
an explicit outcome, not a missing row. A missing trace produces an explicit
evidence limit instead of a reconstructed mechanism.

Reading map: :class:`Episode` is the read-side event and its explicit links,
:class:`EpisodeIndex` builds and queries episodes from one committed state, and
:func:`answer_questions` fills the A18 inspector questions with the records
the episode cites.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Literal, TypeAlias, TypedDict, cast

from pydantic import BaseModel, ConfigDict

from pm.memory.references import referenced_ids
from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.records.base import RecordBase
from pm.records.cognitive_records import (
    BeliefRecord,
    BeliefShiftRecord,
    ExpectationRecord,
)
from pm.records.enums import FeatureType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.introspection.capabilities import CapabilityFindingRow, CapabilityOverlay

__all__ = [
    "Episode",
    "EpisodeAnswer",
    "EpisodeIndex",
    "EpisodeRow",
    "answer_questions",
    "recent_without_duplicate_inner_frames",
]


class EpisodeRow(TypedDict):
    """One serialized episode, with its explicit links kept separate."""

    episode_id: int
    kind: str
    loop: str
    tick_id: int
    operation: str
    owner: str
    summary: str
    outcome: str
    revision: int
    inputs: list[int]
    outputs: list[int]
    links: dict[str, list[int]]
    metadata: dict[str, object]


class CrossLoopLink(TypedDict):
    """One explicit main-to-inner influence pair, not a shared timestamp."""

    main_episode_id: int
    inner_episode_id: int
    via_record_id: int
    relation: Literal["evidence", "cause"]


class PresentRecordRow(TypedDict):
    """One cited record that this store still holds."""

    record_id: int
    type: str
    revision: int
    tick_id: int
    content: str


class MissingRecordRow(TypedDict):
    """One cited record that this store does not hold. ``limit`` says so."""

    record_id: int
    limit: str


class AlternativeRow(TypedDict):
    """One competing option that the episode recorded, with its disposition."""

    text: str
    disposition: str


class InquiryProgressRead(BaseModel):
    """One inquiry step's accepted progress claim, as its operation record keeps it.

    ``workflows.inner_thought.schemas.InquiryProgress`` writes these fields. The
    graph boundary has already rejected an unsupported progress claim, so the
    index reports what was accepted rather than re-judging it.
    """

    model_config = ConfigDict(extra="ignore")

    kind: Literal["new_evidence", "discrimination", "none"]
    reason: str
    meaningful_difference: bool


class InquiryRelationRead(BaseModel):
    """One explicit relation to an earlier graph node, with its stated reason."""

    model_config = ConfigDict(extra="ignore")

    target_record_id: int
    kind: str
    reason: str


class InquiryAlternativeRead(BaseModel):
    """One branch an inquiry step generated, as its operation record keeps it."""

    model_config = ConfigDict(extra="ignore")

    text: str
    disposition: str


class InquiryStepRead(BaseModel):
    """Read-side shape of the ``inquiry_step`` payload on an accepted operation.

    ``workflows.inner_thought.schemas.InquiryStep`` writes the complete payload.
    ``pm`` must not import ``workflows``, so this names only the fields the
    episode index reads from that persisted JSON. Every field is required: a
    missing one is a broken writer contract, not an episode with guessed fields.
    The evidence, the earlier nodes, the operation, and the reality label come
    from the record's typed :class:`~pm.records.base.RecordLineage` instead.
    """

    model_config = ConfigDict(extra="ignore")

    outcome: Literal["continue", "finding", "information_gap", "wait", "pause", "completed", "abandoned"]
    reason: str
    information_gap: str
    progress: InquiryProgressRead
    relations: tuple[InquiryRelationRead, ...]
    alternatives: tuple[InquiryAlternativeRead, ...]


class BeliefEvidenceRow(TypedDict):
    """One belief record with its supporting and opposing evidence ids."""

    record_id: int
    content: str
    supporting: list[int]
    contradicting: list[int]
    superseded_by_id: int | None
    active: bool


AnswerEvidence: TypeAlias = (PresentRecordRow | MissingRecordRow | AlternativeRow | BeliefEvidenceRow
                             | CrossLoopLink | CapabilityFindingRow)
"""One evidence row of an inspector answer."""


class EpisodeAnswer(TypedDict):
    """One inspector answer: what was observed, its evidence, and its limit."""

    answer: str
    evidence: Sequence[AnswerEvidence]
    limit: str | None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: How one inquiry operation's stated outcome reads as an episode outcome. A step
#: that continues leaves its question open, so the episode is unresolved; work an
#: inquiry gave up on is a rejected episode; every other terminal outcome states a
#: result. The operation's own word stays in ``metadata["inquiry_outcome"]``.
_INQUIRY_OUTCOMES = {
    "continue": "unresolved",
    "finding": "committed",
    "completed": "committed",
    "information_gap": "unresolved",
    "wait": "unresolved",
    "pause": "unresolved",
    "abandoned": "rejected",
}

#: Kinds anchored to a record rather than to a workspace turn. A record id and a
#: tick id can share a number, so :meth:`EpisodeIndex.by_id` tries these first.
_RECORD_ANCHOR_KINDS = ("thought", "inquiry_operation", "action", "run_receipt")


@dataclass(frozen=True)
class Episode:
    """One meaningful event, choice, question, or goal and its references.

    ``links`` maps an explicit relation name to the record ids it points at.
    The relation names are ``cause``, ``evidence``, ``output``, ``proposal``,
    ``dispatch``, and ``derivation``. ``outcome`` is one of ``committed``,
    ``idle``, ``rejected``, ``cancelled``, ``failed``, or ``unresolved``.
    """

    episode_id: int
    kind: str
    loop: str
    tick_id: int
    operation: str
    owner: str
    summary: str
    outcome: str
    revision: int = 1
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    links: dict[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def record_ids(self) -> set[int]:
        """Return every record this episode names as input, output, or link."""
        ids: set[int] = set(self.inputs) | set(self.outputs)
        for linked in self.links.values():
            ids.update(linked)
        return ids

    def to_dict(self) -> EpisodeRow:
        """Return the serialized episode for a package, a trace, or a belief."""
        return {
            "episode_id": self.episode_id,
            "kind": self.kind,
            "loop": self.loop,
            "tick_id": self.tick_id,
            "operation": self.operation,
            "owner": self.owner,
            "summary": self.summary,
            "outcome": self.outcome,
            "revision": self.revision,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "links": {name: list(values) for name, values in self.links.items()},
            "metadata": dict(self.metadata),
        }


def _ints(values: Iterable) -> tuple[int, ...]:
    """Return the distinct integer ids of ``values``, in order.

    A value that is not an integer (a missing or foreign id in a receipt)
    contributes nothing; the caller keeps the records it did resolve.
    """
    result: list[int] = []
    for value in values or ():
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(dict.fromkeys(result))


class EpisodeIndex:
    """Build and query episodes from a MemoryStore or frozen snapshot.

    The index reads committed state only. It performs no write and no model
    call. A caller that needs a later state builds a new index.
    """

    def __init__(self, store: MemoryStore | Snapshot) -> None:
        self.store: MemoryStore | Snapshot = store
        self._episodes: list[Episode] = []
        self._build()

    def _build(self) -> None:
        """Read every committed anchor kind once and sort the episodes."""
        store = self.store
        episodes: list[Episode] = []
        for tick, turn in store.gwt_turns.items():
            episodes.append(self._from_turn(tick, turn))
        inquiry_branches = _inquiry_branch_ids(store)
        for record in store.all_knoxels.values():
            # Only a Feature carries a feature type, and only the private loops
            # write inner monologue records. The private mind has two shapes: the
            # retained linear monologue and an executed inquiry operation.
            if isinstance(record, Feature) and record.feature_type is FeatureType.InnerMonologue:
                loop = str(record.metadata.get("loop", ""))
                if loop == "inner":
                    episodes.append(self._from_thought(record))
                elif loop == "inquiry":
                    branches = inquiry_branches.get(int(record.id), ())
                    episodes.append(self._from_inquiry_operation(record, branches))
        for record in store.all_knoxels.values():
            if isinstance(record, ActionSchemaRecord):
                episodes.append(self._from_action(record))
        for run_id, receipt in getattr(store, "managed_runs", {}).items():
            episodes.append(self._from_receipt(run_id, receipt))
        episodes.sort(key=lambda item: (item.tick_id, item.episode_id))
        self._episodes = episodes

    # ------------------------------------------------------------------
    # Anchor builders. Each reads only committed fields.

    def _from_turn(self, tick: int, turn: GwtTurnRecord) -> Episode:
        """Return the episode of one turn record.

        The claimed stimuli are the cause. The records that entered a shard
        are the evidence. The subjective experience and the decision record
        are the derivation. Every listed record is an output.
        """
        loop = turn.slot_id
        links = {"output": _ints(turn.knoxel_ids)}
        if turn.stimulus_ids:
            links["cause"] = _ints(turn.stimulus_ids)
        entered = [item.record_id for change in turn.shard_changes for item in change.entered]
        if entered:
            links["evidence"] = _ints(entered)
        derived = [key for key in (turn.experience_id, turn.decision_id) if key is not None]
        if derived:
            links["derivation"] = _ints(derived)
        outcome = "committed" if links["output"] else "idle"
        return Episode(
            episode_id=int(tick), kind="gwt_turn", loop=loop, tick_id=int(tick),
            operation="cognition_cycle" if loop == "main" else "private_cycle",
            owner="workflows.main_cognition" if loop == "main" else "workflows.inner_thought",
            summary=turn.summary,
            outcome=outcome, revision=self._revision(turn),
            inputs=links.get("cause", ()), outputs=links["output"], links=links,
            metadata={"shards": [change.key.model_dump(mode="json") for change in turn.shard_changes],
                      "codelet_runs": [run.model_dump(mode="json") for run in turn.trace.codelet_runs],
                      "structure_ids": list(turn.trace.structure_ids),
                      "causal_percept_ids": list(turn.trace.causal_percept_ids)},
        )

    def _from_thought(self, record: Feature) -> Episode:
        """Return the episode of one private thought record."""
        metadata = record.metadata or {}
        evidence = _ints(metadata.get("evidence_ids", ()))
        alternatives = [
            {"text": str(item.get("text", "")), "disposition": str(item.get("disposition", ""))}
            for item in metadata.get("alternatives", []) or []
        ]
        outcome = "unresolved" if metadata.get("continuation") == "continue" else "committed"
        links = {"evidence": evidence}
        alternatives_ids = _ints(item for item in metadata.get("alternative_ids", ()))
        if alternatives_ids:
            links["proposal"] = alternatives_ids
        return Episode(
            episode_id=int(record.id), kind="thought", loop="inner", tick_id=int(record.tick_id),
            operation=str(metadata.get("operation", "private_cycle")), owner="workflows.inner_thought",
            summary=str(record.content), outcome=outcome, revision=self._revision(record),
            inputs=evidence, outputs=(int(record.id),), links=links,
            metadata={"trigger": metadata.get("trigger", ""), "reality": metadata.get("reality", ""),
                      "alternatives": alternatives, "uncertainty": metadata.get("uncertainty")},
        )

    def _from_inquiry_operation(self, record: Feature, branch_ids: tuple[int, ...]) -> Episode:
        """Return the episode of one executed operation of the inquiry graph.

        The operation's lineage names its cited evidence, its earlier graph nodes,
        the operation, and its reality label, so those come from typed fields.
        The step's own judgement fields stay as persisted JSON under
        ``inquiry_step`` and are read through :class:`InquiryStepRead`. A record
        with no lineage, or a step payload missing a required field, is a broken
        contract rather than an episode with guessed fields.

        ``branch_ids`` are the competing branch records this operation created,
        found through their lineage. They are the operation's proposals, so they
        are both outputs and ``proposal`` links.
        """
        lineage = record.lineage
        if lineage is None or not lineage.episode_id:
            raise ValueError("An accepted inquiry operation requires its own lineage.")
        raw_step = record.metadata.get("inquiry_step")
        if not isinstance(raw_step, Mapping):
            raise ValueError("An accepted inquiry operation requires its inquiry_step payload.")
        step = InquiryStepRead.model_validate(raw_step)
        stated_outcome = step.outcome
        outcome = _INQUIRY_OUTCOMES[stated_outcome]
        evidence = _ints(lineage.source_event_ids)
        # The first step's parent is the nomination (or deliberate action) that
        # opened the episode. Later steps name the earlier graph nodes they
        # answered. Neither is the operation itself, so no self-reference is
        # filtered; a self-parent would be a broken writer contract.
        parents = _ints(lineage.parent_record_ids)
        links: dict[str, tuple[int, ...]] = {"evidence": evidence}
        if parents:
            links["cause"] = parents
        if branch_ids:
            links["proposal"] = branch_ids
        relations = [{"target_record_id": item.target_record_id, "kind": item.kind,
                      "reason": item.reason} for item in step.relations]
        alternatives = [{"text": item.text, "disposition": item.disposition}
                        for item in step.alternatives]
        return Episode(
            episode_id=int(record.id), kind="inquiry_operation", loop="inquiry",
            tick_id=int(record.tick_id), operation=lineage.operation, owner="workflows.inner_thought",
            summary=str(record.content), outcome=outcome, revision=self._revision(record),
            inputs=evidence, outputs=(int(record.id), *branch_ids), links=links,
            metadata={"graph_episode_id": lineage.episode_id, "sequence": int(lineage.sequence),
                      "reality": lineage.reality, "inquiry_outcome": stated_outcome,
                      "reason": step.reason,
                      "information_gap": step.information_gap,
                      "progress": step.progress.model_dump(mode="json"),
                      "relations": relations, "alternatives": alternatives},
        )

    def _from_action(self, record: ActionSchemaRecord) -> Episode:
        """Return the episode of one action schema, with its status as outcome."""
        links = {"evidence": _ints(record.intention_ids)}
        # The action follows from the decision record of its cycle.
        if record.lineage is not None and record.lineage.parent_record_ids:
            links["cause"] = _ints(record.lineage.parent_record_ids)
        status = record.status.value
        outcome = {"Executed": "committed", "Selected": "committed",
                   "Abandoned": "rejected", "Proposed": "unresolved"}.get(status, "unresolved")
        return Episode(
            episode_id=int(record.id), kind="action", loop="main", tick_id=int(record.tick_id),
            operation=f"action:{status}", owner="workflows.main_cognition",
            summary=str(record.content), outcome=outcome, revision=self._revision(record),
            inputs=links.get("evidence", ()), outputs=(int(record.id),), links=links,
            metadata={"expectation": record.expectation,
                      "execution_mode": record.execution_mode.value},
        )

    def _from_receipt(self, run_id: str, receipt: dict) -> Episode:
        """Return the episode of one managed-run receipt.

        The receipt keeps its own state and evidence; a run id that has no
        control version still gets one stable integer identity.
        """
        state = str(receipt.get("state", "unresolved"))
        outcome = {"completed": "committed", "failed": "failed", "cancelled": "cancelled",
                   "interrupted": "cancelled"}.get(state, "unresolved")
        events = list(receipt.get("events", []) or [])
        references = _ints(event.get("knoxel_id") for event in events)
        # A run id maps to a stable integer id, so the same run keeps one
        # identity across processes and restarts.
        run_key = int(hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:12], 16)
        return Episode(
            episode_id=int(receipt.get("control_version", 0)) or run_key,
            kind="run_receipt", loop="managed", tick_id=int(receipt.get("created_at", 0) or 0),
            operation=str(receipt.get("workflow_name", "managed_run")),
            owner="pm.runs.supervisor", summary=str(receipt.get("latest_event", "")) or state,
            outcome=outcome, revision=int(receipt.get("revision", 0)), outputs=tuple(references),
            links={"output": tuple(references)},
            metadata={"run_id": run_id, "state": state,
                      "evidence": list(receipt.get("evidence", []) or [])},
        )

    def _revision(self, record: RecordBase | GwtTurnRecord) -> int:
        """Return the acceptance revision of one record, or 1 when unrecorded.

        A workspace turn is not a knoxel and has no revision of its own, so a
        turn episode reports its first revision.
        """
        if isinstance(record, GwtTurnRecord):
            return 1
        return int(self.store.revisions.get(int(record.id), 1))

    # ------------------------------------------------------------------
    # Queries.

    def all(self) -> list[Episode]:
        """Return every episode in tick order."""
        return list(self._episodes)

    def recent(self, *, loop: str | None = None, limit: int = 20) -> list[Episode]:
        """Return the newest episodes, newest first, filtered by loop when named."""
        items = [item for item in self._episodes if loop is None or item.loop == loop]
        items.sort(key=lambda item: (item.tick_id, item.episode_id), reverse=True)
        return items[: max(1, limit)]

    def by_id(self, episode_id: int) -> Episode | None:
        """Return the record-anchored episode, or a workspace turn with that tick.

        A record id and a tick id can share a number. Record anchors (a linear
        thought, an inquiry operation, an action, a run receipt) take precedence,
        so a caller that names a record id never silently receives a turn episode.
        """
        target = int(episode_id)
        for kind in _RECORD_ANCHOR_KINDS + ("gwt_turn",):
            for item in self._episodes:
                if item.kind == kind and item.episode_id == target:
                    return item
        return None

    def by_tick(self, tick_id: int) -> list[Episode]:
        """Return every episode anchored to one tick."""
        return [item for item in self._episodes if item.tick_id == int(tick_id)]

    def by_operation(self, operation: str) -> list[Episode]:
        """Return every episode of one operation name."""
        return [item for item in self._episodes if item.operation == operation]

    def for_record(self, record_id: int) -> list[Episode]:
        """Return every episode that links to one record id."""
        target = int(record_id)
        found = []
        for item in self._episodes:
            if target in item.inputs or target in item.outputs:
                found.append(item)
                continue
            if any(target in values for values in item.links.values()):
                found.append(item)
        return found

    # ------------------------------------------------------------------
    # Cross-loop influence. Timestamp order is not used as a cause.

    def cross_loop_links(self) -> list[CrossLoopLink]:
        """Return explicit main/inner influence pairs.

        A pair is recorded only when the main loop's evidence or cause links
        name a private-loop episode's output record: a linear inner monologue or
        an inquiry operation and its branches. A shared timestamp is never a link.
        """
        inner_outputs = {record_id: item for item in self._episodes
                         if item.loop in {"inner", "inquiry"} for record_id in item.outputs}
        pairs: list[CrossLoopLink] = []
        for item in self._episodes:
            if item.loop != "main":
                continue
            referenced = set(item.inputs) | set(item.links.get("cause", ()))
            for record_id in sorted(referenced & set(inner_outputs)):
                pairs.append(CrossLoopLink(
                    main_episode_id=item.episode_id,
                    inner_episode_id=inner_outputs[record_id].episode_id,
                    via_record_id=record_id,
                    relation="evidence" if record_id in item.inputs else "cause",
                ))
        return pairs


def recent_without_duplicate_inner_frames(index: EpisodeIndex, *, limit: int = 20) -> list[Episode]:
    """Return recent episodes without an inner frame row that repeats a private episode.

    One private cycle commits two rows: the thought or graph operation and the
    inner workspace turn that projects it. The turn's outputs hold the private
    record id, so the link is explicit. A companion-facing projection reads the
    private episode once; owner tools keep both rows through
    :meth:`EpisodeIndex.recent` and :meth:`EpisodeIndex.all`.
    """
    recent = index.recent(limit=limit)
    private_ids = {item.episode_id for item in recent if item.loop in {"inner", "inquiry"}}
    kept: list[Episode] = []
    for item in recent:
        if item.kind == "gwt_turn" and item.loop == "inner" and private_ids.intersection(item.outputs):
            continue
        kept.append(item)
    return kept


def _inquiry_branch_ids(store: MemoryStore | Snapshot) -> dict[int, tuple[int, ...]]:
    """Map each executed inquiry operation to the branch records it created.

    A generative inquiry step writes its competing branches as separate ``Thought``
    records whose lineage parent is that operation. The operation record does not
    name them, so the index follows lineage instead of pairing them by time order.
    """
    branches: dict[int, list[int]] = {}
    for record in store.all_knoxels.values():
        if not isinstance(record, Feature) or record.feature_type is not FeatureType.Thought:
            continue
        lineage = record.lineage
        if lineage is None or lineage.operation != "inquiry_alternative" or not lineage.parent_record_ids:
            continue
        for parent_id in lineage.parent_record_ids:
            branches.setdefault(int(parent_id), []).append(int(record.id))
    return {key: tuple(dict.fromkeys(values)) for key, values in branches.items()}


def _question(store: MemoryStore | Snapshot, episode: Episode,
              record_ids: Iterable[int]) -> list[PresentRecordRow | MissingRecordRow]:
    """Describe the named records with their revision, or mark a missing id."""
    rows: list[PresentRecordRow | MissingRecordRow] = []
    for record_id in _ints(record_ids):
        record = store.get_knoxel_by_id(record_id)
        if record is None:
            rows.append(MissingRecordRow(record_id=record_id, limit="record is not present in this store"))
            continue
        rows.append(PresentRecordRow(
            record_id=record_id,
            type=type(record).__name__,
            revision=int(store.revisions.get(record_id, 1)),
            tick_id=int(record.tick_id),
            content=str(record.content)[:400],
        ))
    return rows


def answer_questions(store: MemoryStore | Snapshot, episode: Episode, *,
                     overlay: CapabilityOverlay | None = None) -> dict[str, EpisodeAnswer]:
    """Answer the A18 inspector questions from retained records.

    Every answer names its evidence records. A question with no retained
    evidence returns an explicit limit. The function never reconstructs an
    execution fact from a plausible explanation.
    """
    links = episode.links
    answers: dict[str, EpisodeAnswer] = {}

    answers["information_available"] = {
        "answer": "The frozen input view is the cited evidence set with its revisions.",
        "evidence": _question(store, episode, episode.inputs),
        "limit": None if episode.inputs else "no input records were retained",
    }

    producer_ids = tuple(links.get("cause", ()))
    answers["cause"] = {
        "answer": "Recorded producer and cause links name the source event.",
        "evidence": _question(store, episode, producer_ids),
        "limit": None if producer_ids else "no producer or cause link was retained",
    }

    # Both private builders write this metadata entry as ``AlternativeRow`` entries,
    # and no other writer touches it, so the narrow cast states that contract.
    alternatives = cast("list[AlternativeRow]", episode.metadata.get("alternatives", []))
    answers["alternatives"] = {
        "answer": "Recorded nominations and dispositions name the competing options.",
        "evidence": alternatives,
        "limit": None if alternatives else "no competing alternative was retained",
    }

    outputs = tuple(links.get("output", ())) + tuple(links.get("dispatch", ()))
    answers["changed_after"] = {
        "answer": "The committed output and dispatch references show the observed change.",
        "evidence": _question(store, episode, outputs),
        "limit": None if outputs else "no output or dispatch reference was retained",
    }

    # Retrieval and inference stay distinct: a retrieval source is a record,
    # a generated reading is labeled by its reality or interpretation fields.
    answers["retrieved_or_inferred"] = {
        "answer": "Retrieved records keep their ids; generated readings stay labeled.",
        "evidence": _question(store, episode, links.get("evidence", ())),
        "limit": "a generated interpretation is attributed, not verified" if episode.metadata.get("reality") else None,
    }

    if overlay is not None:
        answers["capabilities"] = {
            "answer": "The capability overlay lists each state with its own evidence.",
            "evidence": [finding.to_dict() for finding in overlay.findings.values()],
            "limit": None,
        }
        planned = [finding.to_dict() for finding in overlay.findings.values()
                   if finding.strongest in {"intended", "implemented", "unavailable", "uncertain"}]
        answers["planned_or_unobserved"] = {
            "answer": "These capabilities have no observed execution evidence.",
            "evidence": planned,
            "limit": None if planned else None,
        }

    cross = [pair for pair in EpisodeIndex(store).cross_loop_links()
             if pair["main_episode_id"] == episode.episode_id]
    answers["private_thought_to_action"] = {
        "answer": "An explicit main-to-inner link shows influence; timestamp order is not a cause.",
        "evidence": cross,
        "limit": None if cross else "no explicit cross-loop link was recorded",
    }

    belief_ids = _ints(links.get("evidence", ()))
    beliefs: list[BeliefEvidenceRow] = []
    for record_id in belief_ids:
        record = store.get_knoxel_by_id(record_id)
        if isinstance(record, BeliefRecord):
            beliefs.append(BeliefEvidenceRow(record_id=record_id, content=record.content,
                                             supporting=list(record.supporting_knoxel_ids),
                                             contradicting=list(record.contradicting_knoxel_ids),
                                             superseded_by_id=record.superseded_by_id,
                                             active=record.active))
    answers["belief_change"] = {
        "answer": "A belief change cites supporting and opposing records and its revision history.",
        "evidence": beliefs,
        "limit": None if beliefs else "this episode did not change a belief",
    }
    return answers

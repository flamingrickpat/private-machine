"""Queryable cognitive episodes built from records the runtime already keeps.

A18 does not install a second tracing framework. An episode is a read-side
view over committed records that already exist: workspace turns, attention
frames, private thoughts, action schemas, beliefs, expectations, and managed
run receipts. The view adds the causal fields the records carry, and it
refuses to infer an execution fact from timestamp order.

An episode stays queryable when it is incomplete. A cancelled cycle, a
rejected proposal, or an unresolved expectation is a first-class episode with
an explicit outcome, not a missing row. A missing trace produces an explicit
evidence limit instead of a reconstructed mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Iterable

from pm.memory.references import referenced_ids
from pm.records.cognitive_records import (
    AttentionFrameRecord,
    BeliefRecord,
    BeliefShiftRecord,
    ExpectationRecord,
)
from pm.records.enums import FeatureType
from pm.records.memory_records import ActionSchemaRecord

__all__ = [
    "Episode",
    "EpisodeIndex",
    "answer_questions",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: Record kinds that carry a decision and therefore anchor an episode.
_ANCHOR_OPERATIONS = {
    "main_cognition": "cognition_cycle",
    "inner_thought": "private_cycle",
}


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
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    result = []
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

    def __init__(self, store):
        self.store = store
        self._episodes: list[Episode] = []
        self._build()

    def _build(self) -> None:
        store = self.store
        episodes: list[Episode] = []
        if hasattr(store, "gwt_turns"):
            turns = store.gwt_turns
        else:
            from pm.records.global_workspace import GwtTurnRecord
            turns = {tick: GwtTurnRecord.model_validate(raw) for tick, raw in store.serialize_gwt().items()}
        for tick, turn in turns.items():
            episodes.append(self._from_turn(tick, turn))
        for record in store.all_knoxels.values():
            if getattr(record, "feature_type", None) == FeatureType.InnerMonologue:
                metadata = record.metadata or {}
                if metadata.get("loop") == "inner":
                    episodes.append(self._from_thought(record))
        for record in store.all_knoxels.values():
            if isinstance(record, ActionSchemaRecord):
                episodes.append(self._from_action(record))
        for run_id, receipt in getattr(store, "managed_runs", {}).items():
            episodes.append(self._from_receipt(run_id, receipt))
        episodes.sort(key=lambda item: (item.tick_id, item.episode_id))
        self._episodes = episodes

    # ------------------------------------------------------------------
    # Anchor builders. Each reads only committed fields.

    def _from_turn(self, tick: int, turn) -> Episode:
        loop = str(getattr(turn, "slot_id", "main") or "main")
        links = {"output": _ints(getattr(turn, "knoxel_ids", ()))}
        if getattr(turn, "attention_frame_id", None) is not None:
            links["cause"] = (int(turn.attention_frame_id),)
        frame = self.store.get_knoxel_by_id(turn.attention_frame_id) if turn.attention_frame_id else None
        operation = "cognition_cycle"
        if isinstance(frame, AttentionFrameRecord):
            links["evidence"] = _ints(frame.focus_knoxel_ids)
            operation = f"frame:{frame.control_mode}"
        outcome = "committed" if links["output"] else "idle"
        return Episode(
            episode_id=int(tick), kind="gwt_turn", loop=loop, tick_id=int(tick),
            operation=operation, owner="workflows.main_cognition" if loop == "main" else "workflows.inner_thought",
            summary=str(getattr(turn, "summary", "")),
            outcome=outcome, revision=self._revision(turn),
            inputs=links.get("evidence", ()), outputs=links["output"], links=links,
        )

    def _from_thought(self, record) -> Episode:
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

    def _from_action(self, record: ActionSchemaRecord) -> Episode:
        links = {"evidence": _ints(getattr(record, "intention_ids", ()))}
        if record.attention_frame_id is not None:
            links["cause"] = (int(record.attention_frame_id),)
        status = str(getattr(record.status, "value", record.status))
        outcome = {"Executed": "committed", "Selected": "committed",
                   "Abandoned": "rejected", "Proposed": "unresolved"}.get(status, "unresolved")
        return Episode(
            episode_id=int(record.id), kind="action", loop="main", tick_id=int(record.tick_id),
            operation=f"action:{status}", owner="workflows.main_cognition",
            summary=str(record.content), outcome=outcome, revision=self._revision(record),
            inputs=links.get("evidence", ()), outputs=(int(record.id),), links=links,
            metadata={"expectation": getattr(record, "expectation", ""),
                      "execution_mode": str(getattr(record.execution_mode, "value", record.execution_mode))},
        )

    def _from_receipt(self, run_id: str, receipt: dict) -> Episode:
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

    def _revision(self, record) -> int:
        return int(self.store.revisions.get(int(getattr(record, "id", -1)), 1))

    # ------------------------------------------------------------------
    # Queries.

    def all(self) -> list[Episode]:
        return list(self._episodes)

    def recent(self, *, loop: str | None = None, limit: int = 20) -> list[Episode]:
        items = [item for item in self._episodes if loop is None or item.loop == loop]
        items.sort(key=lambda item: (item.tick_id, item.episode_id), reverse=True)
        return items[: max(1, limit)]

    def by_id(self, episode_id: int) -> Episode | None:
        """Return the record-anchored episode, or a workspace turn with that tick.

        A record id and a tick id can share a number. Record anchors
        (thought, action, receipt) take precedence, so a caller that names a
        record id never silently receives a turn episode.
        """
        target = int(episode_id)
        for kind in ("thought", "action", "run_receipt", "gwt_turn"):
            for item in self._episodes:
                if item.kind == kind and item.episode_id == target:
                    return item
        return None

    def by_tick(self, tick_id: int) -> list[Episode]:
        return [item for item in self._episodes if item.tick_id == int(tick_id)]

    def by_operation(self, operation: str) -> list[Episode]:
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

    def cross_loop_links(self) -> list[dict]:
        """Return explicit main/inner influence pairs.

        A pair is recorded only when the main loop's evidence or cause links
        name an inner-loop episode's output record. A shared timestamp is
        never a link.
        """
        inner_outputs = {record_id: item for item in self._episodes
                         if item.loop == "inner" for record_id in item.outputs}
        pairs = []
        for item in self._episodes:
            if item.loop != "main":
                continue
            referenced = set(item.inputs) | set(item.links.get("cause", ()))
            for record_id in sorted(referenced & set(inner_outputs)):
                pairs.append({
                    "main_episode_id": item.episode_id,
                    "inner_episode_id": inner_outputs[record_id].episode_id,
                    "via_record_id": record_id,
                    "relation": "evidence" if record_id in item.inputs else "cause",
                })
        return pairs


def _question(store, episode: Episode, record_ids: Iterable[int]) -> list[dict]:
    """Describe the named records with their revision, or mark a missing id."""
    rows = []
    for record_id in _ints(record_ids):
        record = store.get_knoxel_by_id(record_id)
        if record is None:
            rows.append({"record_id": record_id, "limit": "record is not present in this store"})
            continue
        rows.append({
            "record_id": record_id,
            "type": type(record).__name__,
            "revision": int(store.revisions.get(record_id, 1)),
            "tick_id": int(getattr(record, "tick_id", -1)),
            "content": str(getattr(record, "content", ""))[:400],
        })
    return rows


def answer_questions(store, episode: Episode, *, overlay=None) -> dict:
    """Answer the A18 inspector questions from retained records.

    Every answer names its evidence records. A question with no retained
    evidence returns an explicit limit. The function never reconstructs an
    execution fact from a plausible explanation.
    """
    links = episode.links
    answers: dict[str, dict] = {}

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

    alternatives = episode.metadata.get("alternatives", [])
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
    beliefs = []
    for record_id in belief_ids:
        record = store.get_knoxel_by_id(record_id)
        if isinstance(record, BeliefRecord):
            beliefs.append({"record_id": record_id, "content": record.content,
                            "supporting": list(record.supporting_knoxel_ids),
                            "contradicting": list(record.contradicting_knoxel_ids),
                            "superseded_by_id": record.superseded_by_id,
                            "active": record.active})
    answers["belief_change"] = {
        "answer": "A belief change cites supporting and opposing records and its revision history.",
        "evidence": beliefs,
        "limit": None if beliefs else "this episode did not change a belief",
    }
    return answers

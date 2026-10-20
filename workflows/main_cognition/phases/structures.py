"""Structures phase: build structures from the CSM.

Two structure builders run after the codelets (plan section "Structures"):

1. :class:`TopicalCoalitionBuilder` (pure Python) ports
   ``_build_structures_get_coalitions`` from
   ``private-machine-head/pm/ghosts/ghost_lida.py``. It clusters the CSM items
   by embedding with agglomerative clustering (cosine distance, average
   linkage). It builds one *hard* coalition for each cluster and one
   *balanced* coalition for each cluster. A balanced coalition gets records
   of every type in turn, so each coalition holds a spread of types.
2. :class:`StructureBinder` (one story-framed model call) reads the CSM view
   and the recent broadcasts. It binds codelet percepts to memories and
   narratives with a stated reason. A structure can bind conflicting content.

Each structure names a target shard. Attention rates the structures for
each shard and can place their records in another shard.

The structures are drafts with pending ids. The commit stages one
``Structure`` record for each draft and writes the ids into
``CycleState.structure_ids``. The current stimuli and the shard tails are
not part of any structure: the prompt already shows them, so attention
cannot place them as items.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable

from pydantic import create_model, model_validator

from pm.constants import (
    COALITION_ITEMS_PER_CLUSTER,
    COALITION_MAX_CLUSTERS,
    STRUCTURE_BINDER_MAX,
    STRUCTURE_VIEW_TOKENS,
)
from pm.context_render.turns import format_content
from pm.records.codelet_records import StructureRelation
from pm.records.mental_state import compute_attention_bias
from pm.records.situational_model import SituationalModel, cosine
from pm.records.workspace import SELF_SHARD, ShardKey

from workflows.main_cognition.csm_view import Embedder, record_tokens, render_story
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import BinderAnswer, RecordView, StructureDraft, StructuresResult
from workflows.main_cognition.story_prefix import StoryPrefix, resolve_target

#: Returns the embedding of a record id.
EmbeddingOf = Callable[[int], list[float]]

COALITION = "topical_coalition"
BINDER = "structure_binder"


def average_linkage(vectors: list[list[float]], clusters: int) -> list[int]:
    """Cluster ``vectors`` into ``clusters`` groups and return one label for each vector.

    Agglomerative clustering with the cosine distance ``max(0, 1 - cos)``
    and average linkage, like ``AgglomerativeClustering(metric="precomputed",
    linkage="average")`` in the old code. The Lance-Williams update keeps the
    cluster distances exact. Ties merge the pair with the lowest indexes, so
    the result is repeatable. The cost is cubic in the number of vectors.
    The CSM holds at most ``CSM_CAPACITY`` items, so this stays fast.

    Labels are numbered by the first vector of each cluster, from 0.
    """
    count = len(vectors)
    members: dict[int, list[int]] = {index: [index] for index in range(count)}
    distance: dict[tuple[int, int], float] = {}
    for left in range(count):
        for right in range(left + 1, count):
            distance[(left, right)] = max(0.0, 1.0 - cosine(vectors[left], vectors[right]))
    while len(members) > max(1, clusters):
        (left, right), _ = min(distance.items(), key=lambda entry: (entry[1], entry[0]))
        size_left, size_right = len(members[left]), len(members[right])
        for other in members:
            if other in (left, right):
                continue
            to_left = distance[(min(left, other), max(left, other))]
            to_right = distance[(min(right, other), max(right, other))]
            distance[(min(left, other), max(left, other))] = (
                (size_left * to_left + size_right * to_right) / (size_left + size_right))
            del distance[(min(right, other), max(right, other))]
        del distance[(left, right)]
        members[left].extend(members.pop(right))
    labels = [0] * count
    for label, group in enumerate(sorted(members.values(), key=min)):
        for index in group:
            labels[index] = label
    return labels


def _centroid(vectors: list[list[float]]) -> list[float]:
    return [math.fsum(values) / len(vectors) for values in zip(*vectors)]


def _theme(records: dict[int, RecordView], record_ids: list[int]) -> str:
    """Return the old theme summary: the start of the first three records."""
    return "; ".join(records[record_id].content[:30] for record_id in record_ids[:3])


def _time_order(records: dict[int, RecordView], record_ids: list[int]) -> list[int]:
    """Return the ids in world-time order. Records without a time come last, in id order."""
    return sorted(record_ids, key=lambda record_id: (records[record_id].timestamp is None,
                                                     records[record_id].timestamp or 0, record_id))


def _affect(csm: SituationalModel, record_ids: list[int]) -> tuple[float, float]:
    """Return the mean valence and salience of the percepts among ``record_ids``."""
    items = [csm.items[record_id] for record_id in record_ids
             if record_id in csm.items and csm.items[record_id].salience is not None]
    if not items:
        return 0.0, 0.0
    return (round(math.fsum(item.valence or 0.0 for item in items) / len(items), 4),
            round(math.fsum(item.salience or 0.0 for item in items) / len(items), 4))


def _target(csm: SituationalModel, record_ids: list[int]) -> ShardKey:
    """Return the most common shard hint of the records, or the self shard."""
    hints = Counter(csm.items[record_id].shard_hint for record_id in record_ids
                    if record_id in csm.items and csm.items[record_id].shard_hint is not None)
    return hints.most_common(1)[0][0] if hints else SELF_SHARD


class TopicalCoalitionBuilder:
    """Build balanced and hard coalitions from the CSM. Port of ``_build_structures_get_coalitions``.

    The number of clusters is one for each ``COALITION_ITEMS_PER_CLUSTER``
    items (the old value 8), at least 1 and at most
    ``COALITION_MAX_CLUSTERS``. The old code filled the hard coalitions with
    the same round-robin assignment as the balanced ones. Here a hard
    coalition holds exactly one cluster, as the plan describes.
    """

    def __init__(self, records: dict[int, RecordView], embedding_of: EmbeddingOf, rng: random.Random):
        self.records = records
        self.embedding_of = embedding_of
        self.rng = rng

    def kind_of(self, record_id: int) -> str:
        """Return the type group of a record: the feature type of a Feature, else the record class."""
        view = self.records[record_id]
        return view.feature_type if view.record_class == "Feature" else view.record_class

    def build(self, record_ids: list[int], csm: SituationalModel) -> tuple[list[StructureDraft], int]:
        """Return the coalitions of ``record_ids`` and the number of clusters.

        Each draft has a pending id of 0. The caller assigns pending ids.
        The records of each coalition are in world-time order.
        """
        ids = sorted(dict.fromkeys(record_ids))
        if not ids:
            return [], 0
        clusters = min(COALITION_MAX_CLUSTERS, max(1, len(ids) // COALITION_ITEMS_PER_CLUSTER))
        vectors = {record_id: self.embedding_of(record_id) for record_id in ids}
        labels = average_linkage([vectors[record_id] for record_id in ids], clusters)
        cluster_map: dict[int, list[int]] = {}
        for record_id, label in zip(ids, labels, strict=True):
            cluster_map.setdefault(label, []).append(record_id)
        centroids = {label: _centroid([vectors[record_id] for record_id in members])
                     for label, members in cluster_map.items()}

        # Balanced coalitions: shuffle, group by type, and give each record of
        # a type group to the nearest cluster that has no record of that group
        # yet. When every cluster has one, the round starts again.
        shuffled = list(ids)
        self.rng.shuffle(shuffled)
        type_map: dict[str, list[int]] = {}
        for record_id in shuffled:
            type_map.setdefault(self.kind_of(record_id), []).append(record_id)
        balanced: dict[int, list[int]] = {label: [] for label in cluster_map}
        for members in type_map.values():
            available = list(cluster_map)
            for record_id in members:
                best = min(available, key=lambda label: (1.0 - cosine(centroids[label], vectors[record_id]), label))
                balanced[best].append(record_id)
                available.remove(best)
                if not available:
                    available = list(cluster_map)

        drafts: list[StructureDraft] = []
        for kind, groups in (("balanced", balanced), ("hard", cluster_map)):
            for label in sorted(groups):
                members = _time_order(self.records, groups[label])
                if not members:
                    continue
                valence, salience = _affect(csm, members)
                types = sorted({self.kind_of(record_id) for record_id in members})
                reason = (f"balanced coalition near topic cluster {label}: records of the types {', '.join(types)}"
                          if kind == "balanced" else f"hard coalition: the records of topic cluster {label}")
                drafts.append(StructureDraft(builder=COALITION, kind=kind, theme=_theme(self.records, members),
                                             reason=reason, record_ids=members, target_shard=_target(csm, members),
                                             valence=valence, salience=salience))
        return drafts, len(cluster_map)


class StructureBinder:
    """Bind codelet percepts to memories and narratives with one story-framed call.

    The call reads the shared story prefix, then the CSM view with ids, the
    recent broadcasts, and the instruction. An id outside the view, or a
    relation between records outside its structure, is invalid output and gets
    a repair turn. An unknown target is corrected in place instead.
    """

    def __init__(self, work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts):
        self.work = work
        self.prefix = prefix
        self.prompts = prompts

    def instruction(self, view_ids: list[int]) -> str:
        """Return the last turn of the call."""
        broadcasts = "\n".join(f"- {item.summary}" for item in self.work.state.broadcasts if item.summary.strip())
        return format_content(self.prefix.fill(self.prompts.user_template),
                              view=render_story(self.work.records, view_ids, with_id=True) or "(nothing)",
                              broadcasts=broadcasts or "(nothing yet)",
                              max_structures=str(STRUCTURE_BINDER_MAX),
                              targets=", ".join(f'"{label}"' for label in self.prefix.targets)).text

    def schema(self, view_ids: list[int]) -> type[BinderAnswer]:
        """Return the answer model with the checks of this view.

        An unknown ``target`` is not a repair case. The caller resolves it, so
        one wrong word does not repeat the whole story prefix in a repair turn.
        """
        allowed = set(view_ids)

        def check(answer):
            if len(answer.structures) > STRUCTURE_BINDER_MAX:
                raise ValueError(f"Write at most {STRUCTURE_BINDER_MAX} structures.")
            for structure in answer.structures:
                unknown = sorted(set(structure.record_ids) - allowed)
                if unknown:
                    raise ValueError(f"record_ids {unknown} are not in the MIND list. Use only ids from that list.")
                for relation in structure.relations:
                    if {relation.source_id, relation.target_id} - set(structure.record_ids):
                        raise ValueError("A relation can link only ids of its own structure's record_ids.")
            return answer

        return create_model("BinderAnswerChecked", __base__=BinderAnswer,
                            __validators__={"check": model_validator(mode="after")(check)})

    def bind(self, context, view_ids: list[int]) -> list[StructureDraft]:
        """Run the call and return one draft for each bound structure."""
        if not view_ids:
            return []
        answer = complete(context, self.work, self.prefix.ask(self.instruction(view_ids)), self.schema(view_ids))
        csm = self.work.state.csm
        drafts = []
        for structure in answer.structures:
            members = _time_order(self.work.records, list(dict.fromkeys(structure.record_ids)))
            valence, salience = _affect(csm, members)
            drafts.append(StructureDraft(
                builder=BINDER, kind="binding", theme=structure.theme.strip(), reason=structure.reason.strip(),
                record_ids=members, target_shard=self.prefix.targets[resolve_target(structure.target, self.prefix.targets)[0]],
                relations=[StructureRelation(source_id=item.source_id, relation=item.relation,
                                             target_id=item.target_id) for item in structure.relations],
                valence=valence, salience=salience))
        return drafts


def structure_candidates(work: CycleWork) -> list[int]:
    """Return the CSM record ids that structures can hold.

    The current stimuli and the tail records are left out: the shared prefix
    already shows them.
    """
    shown = {item.knoxel_id for item in work.request.stimuli} | set(work.state.workspace.tail_ids())
    return [record_id for record_id in work.state.csm.items if record_id not in shown]


def binder_view(work: CycleWork, embedder: Embedder, candidates: list[int], rng: random.Random) -> list[int]:
    """Return the CSM view of the binder: the items most relevant to the situation.

    The query is the embedding of the new stimuli and the recent dialogue.
    The percepts of this cycle get the precursor bonus, so the binder sees
    the impulses that it must bind. The view holds at most
    ``STRUCTURE_VIEW_TOKENS`` tokens.
    """
    request = work.request
    situation = embedder.text(render_story(
        work.records, [*request.recent_dialogue_ids[-4:], *(item.knoxel_id for item in request.stimuli)],
        with_id=False) or "(nothing happens)")
    view = SituationalModel(items={record_id: work.state.csm.items[record_id] for record_id in candidates})
    fresh = [record_id for record_id in candidates if record_id < 0]
    return view.snapshot_for(situation, embedder.record, lambda record_id: record_tokens(work.records, record_id),
                             attention_bias=compute_attention_bias(work.state.mental_state),
                             max_tokens=STRUCTURE_VIEW_TOKENS, rng=rng, precursor_ids=fresh)


def build(context, work: CycleWork) -> StructuresResult:
    """Run both builders and return their structures with pending ids."""
    embedder = Embedder(context, work.records)
    rng = random.Random(f"{work.request.run_id}:structures")
    candidates = structure_candidates(work)
    embedder.prefetch_records(candidates)
    coalitions, clusters = TopicalCoalitionBuilder(work.records, embedder.record, rng).build(candidates, work.state.csm)
    view_ids = binder_view(work, embedder, candidates, rng)
    bound = StructureBinder(work, StoryPrefix(work), phase_prompts("structures")).bind(context, view_ids)
    result = StructuresResult(structures=[*coalitions, *bound], view_ids=view_ids, clusters=clusters,
                              calls=1 if view_ids else 0)
    for draft in result.structures:
        draft.record_id = work.allocate_pending_id()
    return result


def run(context, work: CycleWork) -> CycleWork:
    work.structures = build(context, work)
    work.state.structure_ids = [draft.record_id for draft in work.structures.structures]
    coalitions = sum(1 for draft in work.structures.structures if draft.builder == COALITION)
    work.receipt("structures", f"built {coalitions} coalitions from {work.structures.clusters} topic clusters and "
                               f"{len(work.structures.structures) - coalitions} bound structures")
    return work


__all__ = [
    "BINDER",
    "COALITION",
    "StructureBinder",
    "TopicalCoalitionBuilder",
    "average_linkage",
    "binder_view",
    "build",
    "run",
    "structure_candidates",
]

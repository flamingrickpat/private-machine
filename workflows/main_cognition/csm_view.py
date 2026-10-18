"""Embed, render, and fill the current situational model (CSM) of one cycle.

The CSM holds record ids. The phases render those records as story lines
and embed them through the ``embed_texts`` tool of the run. The host grants
that tool. The embedding service caches vectors by text, so the same record
text gets the same vector in every cycle. Prompts never show embeddings.

The retrieve phase calls :func:`enter_retrieval`. It adds the current
stimuli, the last dialogue turns, the memory candidates, the narratives, and
the active goals to the CSM with the endorser ``retrieval`` (plan "What
enters the CSM"). Prune keeps the items that the CSM requirements of
``workspace.yaml`` protect (:func:`csm_protected`).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pm.constants import (
    CSM_CANDIDATE_ACTIVATION,
    CSM_DIALOGUE_ACTIVATION,
    CSM_GOAL_ACTIVATION,
    CSM_NARRATIVE_ACTIVATION,
    CSM_SELECTED_MEMORY_ACTIVATION,
    CSM_STIMULUS_ACTIVATION,
)
from pm.context_render.turns import Role, Turn
from pm.llm.context_window import estimate_tokens
from pm.records.base import persisted_type_name
from pm.records.enums import FeatureType
from pm.records.situational_model import SituationalItem
from pm.records.workspace import WorkspaceRequirementConfig, load_workspace_requirements

from workflows.main_cognition.cycle_work import CycleWork, provisional_tick
from workflows.main_cognition.schemas import RecordView
from workflows.main_cognition.workspace_selection import MatchContext, protected_ids

#: The endorser name of retrieval entries.
RETRIEVAL = "retrieval"
#: The content requirements of the shards and the CSM.
WORKSPACE_FILE = Path(__file__).resolve().with_name("workspace.yaml")

_EMPTY_TOKENS = estimate_tokens("", [Turn(Role.User, "")])[0]


class Embedder:
    """Embed record views and texts through the ``embed_texts`` tool.

    One instance serves one phase. It caches vectors by text, so one text
    costs one tool call per phase. ``records`` is the record view map of the
    cycle. The embedder reads it at each call, so views that the phase adds
    later are found too. A record id without a view raises KeyError: that is
    a bug in the phase that added the id.
    """

    def __init__(self, context, records: dict[int, RecordView]):
        self._tool = context.runtime.tools["embed_texts"]
        self._records = records
        self._cache: dict[str, list[float]] = {}

    def prefetch(self, texts: Iterable[str]) -> None:
        """Embed every text that is not cached yet, in one tool call."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._tool(missing), strict=True))

    def text(self, text: str) -> list[float]:
        """Return the embedding of ``text``."""
        self.prefetch([text])
        return self._cache[text]

    def record(self, record_id: int) -> list[float]:
        """Return the embedding of the story line of one record."""
        return self.text(self._records[record_id].story_line())

    def prefetch_records(self, record_ids: Iterable[int]) -> None:
        """Embed the story lines of many records in one tool call."""
        self.prefetch(self._records[record_id].story_line() for record_id in record_ids)


def _entity_facts(record, record_class: str) -> tuple[list[int], str]:
    """Return the entity ids that a record is about and its category label.

    The record classes name their entity in different fields. A class that
    names no entity returns an empty list. Entity id 0 means "unknown" in
    every class, so it never counts.
    """
    if record_class == "Narrative":
        return [record.entity_id] if record.entity_id else [], record.narrative_type.value
    if record_class in {"SemanticRecord", "DeclarativeFactRecord"}:
        return [record.subject_entity_id] if record.subject_entity_id else [], record.record_type.value
    if record_class == "BeliefRecord":
        return list(record.about_entity_ids), record.belief_kind
    if record_class in {"Feature", "Stimulus", "Intention"}:
        return [record.entity_id] if record.entity_id else [], ""
    return [], ""


def record_view(record) -> RecordView:
    """Return the view of one memory store record. The host and the agent tool call it.

    A dialogue turn names its speaker. Another feature names its source and
    its feature type, so a private thought never reads like speech. A
    codelet percept names its codelet. A narrative names its type and the
    entity it is about. Other records name their type. The view also holds
    the facts that the workspace requirements match (see ``RecordView``).
    """
    kind = persisted_type_name(type(record))
    record_class = type(record).__name__
    entity_ids, category = _entity_facts(record, record_class)
    source = ""
    feature_type = ""
    if record_class == "Feature":
        feature_type = record.feature_type.value
        source = record.source if record.feature_type == FeatureType.Dialogue else f"{record.source} ({feature_type})"
    elif record_class == "CodeletPercept":
        source = record.codelet_name
    elif record_class == "Narrative":
        source = f"narrative ({category}) about {record.target_name}"
    return RecordView(record_id=record.id, kind=kind, source=source, content=record.content,
                      timestamp=record.timestamp_world_begin, record_class=record_class,
                      entity_ids=entity_ids, category=category, feature_type=feature_type)


def story_tokens(text: str) -> int:
    """Return the estimated token count of one story text, without chat framing."""
    return estimate_tokens("", [Turn(Role.User, text)])[0] - _EMPTY_TOKENS


def record_tokens(records: dict[int, RecordView], record_id: int) -> int:
    """Return the token count of the story line of one record, without its id.

    The workspace budgets and the tails count tokens with this function.
    """
    return story_tokens(story_line(records[record_id], with_id=False))


def workspace_requirements() -> WorkspaceRequirementConfig:
    """Read the content requirements of ``workspace.yaml``. An invalid file raises."""
    return load_workspace_requirements(WORKSPACE_FILE)


def match_context(work: CycleWork) -> MatchContext:
    """Return the cycle facts that the requirement selectors read."""
    return MatchContext(
        self_entity_id=work.request.self_entity_id,
        group_entity_ids=[int(group.key.ref) for group in work.state.stimulus_groups if group.key.kind == "entity"],
        stimulus_ids=[item.knoxel_id for item in work.request.stimuli])


def csm_protected(work: CycleWork) -> set[int]:
    """Return the CSM record ids that the CSM requirements protect from prune.

    At least the current stimuli, the narratives of the self, and the
    narratives of the group entities stay (plan "CSM requirements").
    """
    return protected_ids(work.state.csm, workspace_requirements().csm, work.records, match_context(work))


def story_line(view: RecordView, *, with_id: bool) -> str:
    """Render one record as a story line: time, speaker or kind, and content.

    ``with_id`` puts ``[id N]`` in front, so a model can cite the record.
    """
    when = f"({view.timestamp:%Y-%m-%d %H:%M}) " if view.timestamp is not None else ""
    prefix = f"[id {view.record_id}] " if with_id else ""
    return f"{prefix}{when}{view.story_line()}"


def render_story(records: dict[int, RecordView], record_ids: Iterable[int], *, with_id: bool) -> str:
    """Render records in world-time order. Records without a time come last, in input order."""
    views = [records[record_id] for record_id in dict.fromkeys(record_ids)]
    timed = sorted((view for view in views if view.timestamp is not None), key=lambda view: view.timestamp)
    untimed = [view for view in views if view.timestamp is None]
    return "\n".join(story_line(view, with_id=with_id) for view in [*timed, *untimed])


def enter_retrieval(work: CycleWork, embedder: Embedder) -> list[int]:
    """Add the retrieval candidates of this cycle to the CSM. Return the removed ids.

    Entries and their start activation:

    - each current stimulus: ``CSM_STIMULUS_ACTIVATION``, with the shard of
      its stimulus group as hint;
    - each of the last dialogue turns: ``CSM_DIALOGUE_ACTIVATION``;
    - each memory that the retrieve phase selected:
      ``CSM_SELECTED_MEMORY_ACTIVATION``;
    - each other memory candidate: ``CSM_CANDIDATE_ACTIVATION``;
    - each narrative of the self and of the group entities:
      ``CSM_NARRATIVE_ACTIVATION``;
    - each active goal: ``CSM_GOAL_ACTIVATION``.

    An item that the CSM already holds gets a boost instead (see
    ``SituationalModel.add_or_boost``). Then the CSM is pruned. Prune keeps
    the items that the CSM requirements protect (:func:`csm_protected`).
    """
    request = work.request
    csm = work.state.csm
    tick = provisional_tick(work)
    stimulus_ids = [item.knoxel_id for item in request.stimuli]
    selected = set(work.context.candidate_ids)
    entries: list[SituationalItem] = []
    for group in work.state.stimulus_groups:
        entries.extend(SituationalItem(record_id=record_id, entered_tick=tick, last_boost_tick=tick,
                                       activation=CSM_STIMULUS_ACTIVATION, shard_hint=group.key)
                       for record_id in group.stimulus_ids)
    entries.extend(SituationalItem(record_id=record_id, entered_tick=tick, last_boost_tick=tick,
                                   activation=CSM_DIALOGUE_ACTIVATION)
                   for record_id in request.recent_dialogue_ids)
    entries.extend(SituationalItem(record_id=item.knoxel_id, entered_tick=tick, last_boost_tick=tick,
                                   activation=CSM_SELECTED_MEMORY_ACTIVATION if item.knoxel_id in selected
                                   else CSM_CANDIDATE_ACTIVATION)
                   for item in request.memory_candidates if item.knoxel_id not in stimulus_ids)
    entries.extend(SituationalItem(record_id=record_id, entered_tick=tick, last_boost_tick=tick,
                                   activation=CSM_NARRATIVE_ACTIVATION)
                   for record_id in request.narrative_ids)
    entries.extend(SituationalItem(record_id=goal.knoxel_id, entered_tick=tick, last_boost_tick=tick,
                                   activation=CSM_GOAL_ACTIVATION)
                   for goal in request.active_goal_records)
    embedder.prefetch_records([*csm.items, *(entry.record_id for entry in entries)])
    for entry in entries:
        csm.add_or_boost(entry, RETRIEVAL, embedder.record)
    return csm.prune(csm_protected(work))


__all__ = ["Embedder", "RETRIEVAL", "WORKSPACE_FILE", "csm_protected", "enter_retrieval", "match_context",
           "record_tokens", "record_view", "render_story", "story_line", "story_tokens", "workspace_requirements"]

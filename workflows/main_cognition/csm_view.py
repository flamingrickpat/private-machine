"""Embed, render, and fill the current situational model (CSM) of one cycle.

The CSM holds record ids. The phases render those records as story lines
and embed them through the ``embed_texts`` tool of the run. The host grants
that tool. The story line itself comes from
:mod:`pm.context_render.story`, which the memory inspector shares.
 Loaded record views carry the vectors of their central knoxels.
Prompts never show embeddings.

The retrieve phase calls :func:`enter_retrieval`. It adds the current
stimuli, the last dialogue turns, the memory candidates, the narratives, and
the active goals to the CSM with the endorser ``retrieval`` (plan "What
enters the CSM"). Prune keeps the items that the CSM requirements of
``workspace.yaml`` protect (:func:`csm_protected`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from pm.constants import (
    CSM_CANDIDATE_ACTIVATION,
    CSM_DIALOGUE_ACTIVATION,
    CSM_EXPECTATION_ACTIVATION,
    CSM_GOAL_ACTIVATION,
    CSM_NARRATIVE_ACTIVATION,
    CSM_SELECTED_MEMORY_ACTIVATION,
    CSM_STIMULUS_ACTIVATION,
)
# The story rendering lives in pm, so the memory inspector renders with it.
# These names stay importable from here for the phases.
from pm_workflows.python_role import RoleContext

from pm.context_render.story import (
    RecordView, record_tokens, record_view, render_story, story_line, story_tokens,
)
from pm.records.situational_model import SituationalItem
from pm.records.workspace import WorkspaceRequirementConfig, load_workspace_requirements

from workflows.main_cognition.cycle_work import CycleWork, provisional_tick
from workflows.main_cognition.workspace_selection import MatchContext, protected_ids

#: The endorser name of retrieval entries.
RETRIEVAL = "retrieval"
#: The endorser name of open expectations.
EXPECTATION = "expectation"
#: The content requirements of the shards and the CSM.
WORKSPACE_FILE = Path(__file__).resolve().with_name("workspace.yaml")


class Embedder:
    """Embed record views and texts through the ``embed_texts`` tool.

    One instance serves one phase. Records retain their own vectors.
    Temporary query vectors belong to the caller. ``records`` holds the
    typed record views of the cycle. The embedder reads it at each call, so views that the phase adds
    later are found too. A record id without a view raises KeyError: that is
    a bug in the phase that added the id.
    """

    def __init__(self, context: RoleContext, records: dict[int, RecordView]) -> None:
        self._tool: Callable[[list[str]], list[list[float]]] = context.runtime.tools["embed_texts"]
        self._records = records

    def prefetch(self, texts: Iterable[str]) -> dict[str, list[float]]:
        """Return vectors for a group of query texts. The caller owns the results."""
        texts = list(dict.fromkeys(texts))
        return dict(zip(texts, self._tool(texts), strict=True)) if texts else {}

    def text(self, text: str) -> list[float]:
        """Return the embedding of ``text``."""
        return self._tool([text])[0]

    def record(self, record_id: int) -> list[float]:
        """Return the stored vector of one record. Embed its content only when missing."""
        record = self._records[record_id]
        if not record.embedding:
            record.embedding = self.text(record.content)
        return record.embedding

    def prefetch_records(self, record_ids: Iterable[int]) -> None:
        """Fill missing vectors on record views in one tool call."""
        records = [self._records[key] for key in dict.fromkeys(record_ids) if not self._records[key].embedding]
        if records:
            for record, vector in zip(records, self._tool([record.content for record in records]), strict=True):
                record.embedding = vector


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


def enter_retrieval(work: CycleWork, embedder: Embedder, expectation_ids: Iterable[int] = ()) -> list[int]:
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
    - each active goal: ``CSM_GOAL_ACTIVATION``;
    - each open expectation of ``expectation_ids``:
      ``CSM_EXPECTATION_ACTIVATION``, with the endorser ``expectation``, so
      attention can see what the companion still waits for.

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
    expected = [SituationalItem(record_id=record_id, entered_tick=tick, last_boost_tick=tick,
                                activation=CSM_EXPECTATION_ACTIVATION) for record_id in expectation_ids]
    embedder.prefetch_records([*csm.items, *(entry.record_id for entry in [*entries, *expected])])
    for entry in entries:
        csm.add_or_boost(entry, RETRIEVAL, embedder.record)
    for entry in expected:
        csm.add_or_boost(entry, EXPECTATION, embedder.record)
    return csm.prune(csm_protected(work))


__all__ = ["EXPECTATION", "Embedder", "RETRIEVAL", "WORKSPACE_FILE", "csm_protected", "enter_retrieval", "match_context",
           "record_tokens", "record_view", "render_story", "story_line", "story_tokens", "workspace_requirements"]

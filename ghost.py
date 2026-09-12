"""The shared in-memory Ghost for the Spamuel prototype.

A small :class:`BaseGhost` subclass with no LLM attachment. It owns a lock and
the synchronized write path every module uses, plus the GWT turn records the
main cognition loop publishes.
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from pm_core.ghost.ghost_base import BaseGhost
from pm_core.model.ghost_state import CycleStateModel, GhostConfig
from pm_core.model.gwt import GwtTurnRecord
from pm_core.persist.dataprovider_yaml import load_ghost_state_from_yaml


def serialize_knoxel(knoxel: Any) -> dict:
    """Return a knoxel in the registered ``__type__`` / ``data`` format."""
    return {
        "__type__": knoxel.__class__.__name__,
        "data": knoxel.model_dump(mode="json", serialize_as_any=True),
    }


class Ghost(BaseGhost):
    """The prototype memory object. No embedding, LLM, or disk export."""

    def __init__(self, config: GhostConfig):
        super().__init__(None, config)
        self.lock = RLock()
        self.gwt_turns: dict[int, GwtTurnRecord] = {}

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def from_state(cls, state: CycleStateModel) -> "Ghost":
        config = GhostConfig(
            user_name=state.user_name,
            companion_name=state.companion_name,
            timestamp_format=state.timestamp_format,
            character_card_story=state.character_card_story,
            character_card_assistant=state.character_card_assistant,
            supported_capabilities=state.supported_capabilities,
            unsupported_capabilities=state.unsupported_capabilities,
            capability_notes=state.capability_notes,
            personality_profile=state.personality_profile,
        )
        ghost = cls(config)
        ghost.states = list(state.cycles)
        ghost.all_knoxels = {knoxel.id: knoxel for knoxel in state.knoxels}
        ghost.current_knoxel_id = max((k.id for k in state.knoxels), default=0)
        ghost.current_tick_id = max(
            [max((c.tick_id for c in state.cycles), default=0)]
            + [max((k.tick_id for k in state.knoxels), default=0)],
        )
        ghost.rebuild_knoxel_indexes()
        return ghost

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Ghost":
        state = load_ghost_state_from_yaml(Path(path))
        return cls.from_state(state)

    # ------------------------------------------------------------- mutation

    def store_knoxel(self, knoxel: Any) -> Any:
        """Add one knoxel under the shared lock; assigns id and tick."""
        with self.lock:
            return self.add_knoxel(knoxel, generate_embedding=False)

    def create_bash_environment(self, mode: str = "read", visible_ids=None):
        from working_target import GhostBashWrapper

        return GhostBashWrapper(self, mode=mode, visible_ids=visible_ids)


__all__ = ["Ghost", "serialize_knoxel"]
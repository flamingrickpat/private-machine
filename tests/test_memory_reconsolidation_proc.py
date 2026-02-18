from types import SimpleNamespace

from pm.ghosts.procedures.memory_reconsolidation import MemoryReconsolidationProc


class _Consolidator:
    def __init__(self):
        self.calls = 0

    def consolidate_memory_if_needed(self):
        self.calls += 1


class _Ghost:
    def __init__(self, with_consolidator: bool = True):
        self.current_tick_id = 7
        self.all_knoxels = {}
        self.all_features = []
        self.all_declarative_facts = []
        self.all_narratives = []
        self.all_episodic_memories = []
        self.all_intentions = []
        self.all_graph_nodes = []
        self.all_graph_edges = []
        self.all_concepts = []
        self.story_context_embedding_cache = {"x": [0.1]}
        self.memory_consolidator = _Consolidator() if with_consolidator else None
        self.config = SimpleNamespace(companion_name="Companion", user_name="User")


def test_memory_recon_proc_calls_consolidator_and_persists_payload():
    ghost = _Ghost(with_consolidator=True)
    out = MemoryReconsolidationProc.run(ghost)
    assert out["ok"] is True
    assert out["skipped"] is False
    assert ghost.memory_consolidator.calls == 1
    assert isinstance(ghost.memory_recon_last, dict)
    assert isinstance(ghost.memory_recon_history, list)
    assert ghost.story_context_embedding_cache == {}


def test_memory_recon_proc_handles_missing_consolidator():
    ghost = _Ghost(with_consolidator=False)
    out = MemoryReconsolidationProc.run(ghost)
    assert out["ok"] is False
    assert out["skipped"] is True
    assert out["reason"] == "missing_memory_consolidator"


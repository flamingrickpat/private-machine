import json
from pathlib import Path

from pm.data_structures import ClusterType, MemoryClusterKnoxel, Stimulus, StimulusType
from pm.ghosts.base_ghost import BaseGhost, GhostConfig
from pm.ghosts.knoxel_trace import (
    finalize_knoxel_trace_session,
    set_knoxel_trace_phase,
    start_knoxel_trace_session,
    trace_knoxel_flow,
)
from pm.ghosts.procedures.pam import PamProc


class _LLMStub:
    def get_embedding(self, text):
        t = (text or "").lower()
        if "music" in t:
            return [1.0, 0.0, 0.0]
        if "coding" in t:
            return [0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.0]


def test_pam_subprocedures_are_traced_with_memory_inputs(tmp_path: Path):
    ghost = BaseGhost(_LLMStub(), GhostConfig())
    ghost.current_tick_id = 4
    ghost.enable_knoxel_trace = True
    ghost.knoxel_trace_output_dir = str(tmp_path)

    m1 = MemoryClusterKnoxel(
        content="User likes music and headphones.",
        cluster_type=ClusterType.Topical,
        level=100,
        embedding=[1.0, 0.0, 0.0],
    )
    m2 = MemoryClusterKnoxel(
        content="User works on coding projects.",
        cluster_type=ClusterType.Topical,
        level=100,
        embedding=[0.0, 1.0, 0.0],
    )
    ghost.add_knoxel(m1, generate_embedding=False)
    ghost.add_knoxel(m2, generate_embedding=False)

    ghost.primary_stimulus = Stimulus(
        content="Let's talk about music gear.",
        source="user",
        stimulus_type=StimulusType.UserMessage,
    )
    ghost.add_knoxel(ghost.primary_stimulus, generate_embedding=False)

    start_knoxel_trace_session(ghost)
    set_knoxel_trace_phase(ghost, "pam")

    trace_knoxel_flow(name="PamProc.run", phase="pam")(PamProc.run)(ghost)
    paths = finalize_knoxel_trace_session(ghost)

    assert paths is not None
    data = json.loads(Path(paths["cytoscape_json"]).read_text(encoding="utf-8"))
    calls = data["trace"]["calls"]
    names = {c["name"] for c in calls}
    assert "PamProc._build_query_embedding" in names
    assert "PamProc._collect_candidates" in names
    assert "PamProc._rank_candidates" in names
    assert "PamProc._emit_recall_features" in names

    emit_call = next(c for c in calls if c["name"] == "PamProc._emit_recall_features")
    assert m1.id in emit_call["input_knoxels"] or m2.id in emit_call["input_knoxels"]

    # Memory derivation edges are present for at least one recall feature.
    edges = data["elements"]["edges"]
    assert any(e["data"].get("relation") == "derived_from_memory" for e in edges)


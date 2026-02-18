import json
from pathlib import Path

from pm.data_structures import Feature, FeatureType
from pm.ghosts.base_ghost import BaseGhost, GhostConfig
from pm.ghosts.knoxel_trace import (
    finalize_knoxel_trace_session,
    set_knoxel_trace_phase,
    start_knoxel_trace_session,
    trace_knoxel_flow,
)


class _LLMStub:
    def get_embedding(self, text):
        return [0.0, 0.1, 0.2]


def test_knoxel_trace_writes_cytoscape_and_logs_additions(tmp_path: Path):
    ghost = BaseGhost(_LLMStub(), GhostConfig())
    ghost.current_tick_id = 9
    ghost.enable_knoxel_trace = True
    ghost.knoxel_trace_output_dir = str(tmp_path)

    start_knoxel_trace_session(ghost)
    set_knoxel_trace_phase(ghost, "demo_phase")

    base = Feature(
        content="base thought",
        feature_type=FeatureType.Thought,
        source="seed",
        interlocus=-1,
        causal=True,
    )
    ghost.add_knoxel(base)

    @trace_knoxel_flow(name="DemoProc.run", phase="demo_phase")
    def _proc(g):
        # force read tracking
        _ = g.get_knoxel_by_id(base.id)
        f = Feature(
            content="new thought",
            feature_type=FeatureType.Thought,
            source="demo",
            interlocus=-1,
            causal=False,
        )
        g.add_knoxel(f)
        return f

    @trace_knoxel_flow(name="WorkspaceProc.run", phase="workspace")
    def _proc2(g):
        a = g.get_knoxel_by_id(base.id)
        b = g.get_knoxel_by_id(base.id + 1)
        _ = [x for x in [a, b] if x is not None]
        return None

    _proc(ghost)
    _proc2(ghost)
    paths = finalize_knoxel_trace_session(ghost)

    assert paths is not None
    assert paths["cytoscape_json"].endswith("_cytoscape.json")
    assert ghost.last_knoxel_trace_cytoscape_json

    data = json.loads(Path(paths["cytoscape_json"]).read_text(encoding="utf-8"))
    assert data["format"] == "cytoscape"
    assert data["meta"]["tick"] == 9
    assert data["meta"]["event_count"] >= 1

    nodes = data["elements"]["nodes"]
    edges = data["elements"]["edges"]
    assert any(n["data"].get("kind") == "call" for n in nodes)
    assert any(n["data"].get("kind") == "knoxel" for n in nodes)
    assert any(n.get("classes", "").startswith("proxy") for n in nodes)
    assert any(n["data"].get("kind") == "tag_group" for n in nodes)
    assert any(e["data"].get("relation") == "add_event" for e in edges)
    assert any(e["data"].get("relation") == "exec_next" for e in edges)
    assert any(e["data"].get("relation") == "input_proxy" for e in edges)
    assert any(e["data"].get("relation") == "tag_group" for e in edges)

    # base feature should carry semantic tags from read/use.
    base_node = next(n for n in nodes if n["data"].get("kind") == "knoxel" and n["data"].get("knoxel_id") == base.id)
    assert any("workspace_basis" == t for t in base_node["data"].get("trace_tags", []))

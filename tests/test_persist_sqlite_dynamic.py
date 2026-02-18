import json
import sqlite3
from pathlib import Path

from pm.data_structures import Action, ActionType, Feature, FeatureType, Intention
from pm.ghosts.base_ghost import BaseGhost, GhostConfig, GhostState
from pm.ghosts.persist_sqlite import PersistSqlite


class _LLMStub:
    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]


def _mk_ghost():
    return BaseGhost(_LLMStub(), GhostConfig())


def test_sqlite_dynamic_roundtrip_json_fields(tmp_path: Path):
    db_path = tmp_path / "ghost_dynamic.db"
    ghost = _mk_ghost()
    ghost.current_tick_id = 5
    ghost.states.append(GhostState(tick_id=5))

    expectation = Intention(
        content="User expects concise answer",
        internal=False,
        urgency=0.7,
        affective_valence=0.3,
        incentive_salience=0.6,
        fulfilment=0.0,
    )
    ghost.add_knoxel(expectation, generate_embedding=False)

    action = Action(
        content="Reply with two bullet points",
        action_type=ActionType.Reply,
        generated_expectation_ids=[expectation.id],
    )
    ghost.add_knoxel(action, generate_embedding=False)

    feature = Feature(
        content="Need to keep it short and clear",
        feature_type=FeatureType.Thought,
        source="test",
        interlocus=-1.0,
        metadata={"source": "unit", "scores": [0.5, 0.7]},
        embedding=[0.11, 0.22, 0.33],
        causal=True,
    )
    ghost.add_knoxel(feature, generate_embedding=False)

    PersistSqlite(ghost).save_state_sqlite(str(db_path))

    # Ensure human-readable JSON persisted in TEXT columns.
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT embedding, metadata FROM feature WHERE id = ?", (feature.id,))
    row = cur.fetchone()
    assert isinstance(row[0], str)
    assert isinstance(row[1], str)
    assert json.loads(row[0]) == [0.11, 0.22, 0.33]
    assert json.loads(row[1])["source"] == "unit"
    conn.close()

    loaded = _mk_ghost()
    ok = PersistSqlite(loaded).load_state_sqlite(str(db_path))
    assert ok is True

    loaded_feature = loaded.get_knoxel_by_id(feature.id)
    loaded_action = loaded.get_knoxel_by_id(action.id)
    assert loaded_feature.embedding == [0.11, 0.22, 0.33]
    assert loaded_feature.metadata["scores"] == [0.5, 0.7]
    assert loaded_action.generated_expectation_ids == [expectation.id]
    assert loaded.states and loaded.states[-1].tick_id == 5


def test_sqlite_dynamic_loader_tolerates_extra_columns(tmp_path: Path):
    db_path = tmp_path / "ghost_dynamic_extra.db"
    ghost = _mk_ghost()
    ghost.current_tick_id = 2
    ghost.states.append(GhostState(tick_id=2))
    feature = Feature(
        content="extra column test",
        feature_type=FeatureType.Thought,
        source="test",
        interlocus=-1.0,
        causal=True,
    )
    ghost.add_knoxel(feature, generate_embedding=False)
    PersistSqlite(ghost).save_state_sqlite(str(db_path))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("ALTER TABLE feature ADD COLUMN debug_extra TEXT;")
    cur.execute("UPDATE feature SET debug_extra = 'hello' WHERE id = ?;", (feature.id,))
    conn.commit()
    conn.close()

    loaded = _mk_ghost()
    ok = PersistSqlite(loaded).load_state_sqlite(str(db_path))
    assert ok is True
    loaded_feature = loaded.get_knoxel_by_id(feature.id)
    assert loaded_feature is not None
    assert loaded_feature.content == "extra column test"


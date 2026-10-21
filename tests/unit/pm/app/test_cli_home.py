"""Resolve ``--home`` options and keep accepted state across a restart."""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from pm.app.character_home import CharacterHome
from pm.app.cli import build_from_plan, launch_plan
from pm.constants import WEB_CHAT_PORT
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.enums import FeatureType, InterlocusType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import new_feature
from tests.unit.pm.app.test_web_chat import make_home

SEED = Path("seed.yaml")


def rejected(argv) -> bool:
    """Return True when argparse rejects ``argv``."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.suppress(SystemExit):
        launch_plan(argv, SEED)
        return False
    return True


class LaunchPlanTests(unittest.TestCase):
    def test_home_supplies_checkpoint_runtime_and_persona(self):
        home = Path(tempfile.gettempdir()) / "pm-home"
        plan = launch_plan(["--home", str(home), "--config", "models.yaml", "--cognition", "dummy"], SEED)
        self.assertEqual(plan.home, home.resolve())
        self.assertEqual(plan.checkpoint, home.resolve() / "ghost.pickle")
        self.assertEqual(plan.runtime_root, home.resolve() / "runtime")
        self.assertIsNone(plan.seed)
        self.assertEqual((plan.config, plan.cognition, plan.web, plan.web_port), ("models.yaml", "dummy", True, WEB_CHAT_PORT))
        self.assertEqual(launch_plan(["--home", str(home), "--web-port", "9000"], SEED).web_port, 9000)

    def test_home_excludes_seed_checkpoint_ephemeral_and_runtime_root(self):
        for extra in (["--seed", "x.yaml"], ["--checkpoint", "g.json"], ["--ephemeral"], ["--runtime-root", "r"]):
            self.assertTrue(rejected(["--home", "h", *extra]), extra)

    def test_terminal_mode_keeps_its_defaults(self):
        plan = launch_plan([], SEED)
        self.assertEqual((plan.home, plan.seed, plan.web), (None, SEED, False))
        self.assertEqual(plan.checkpoint, Path.cwd() / ".runtime" / "ghost.pickle")
        plan = launch_plan(["--runtime-root", "r", "--ephemeral", "--web"], SEED)
        self.assertEqual((plan.runtime_root, plan.checkpoint, plan.web), (Path("r"), None, True))

    def test_missing_ghost_json_stops_the_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = launch_plan(["--home", tmp], SEED)
            with self.assertRaises(FileNotFoundError):
                build_from_plan(plan, assembly=None)


class RestartTests(unittest.TestCase):
    def test_restart_keeps_ticks_history_and_previous_state(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        store = make_home(tmp / "home").open_store()
        message = new_feature(store, "hello", FeatureType.Dialogue, source="Jesse", interlocus=InterlocusType.Public)
        # One accepted cycle, committed like the dummy main cognition does it.
        batch = store.begin_batch("cycle-1", new_tick=True)
        tick = batch.tick_id
        batch.set_gwt(GwtTurnRecord(tick_id=tick, stimulus_ids=[message.id], knoxel_ids=[message.id], summary="s"),
                      CycleState(tick_id=tick, stimulus_groups=[StimulusGroup(stimulus_ids=[message.id])]))
        store.commit(batch)
        reply = new_feature(store, "hi!", FeatureType.Dialogue, source="Sunny", interlocus=InterlocusType.Public)

        reopened = CharacterHome(tmp / "home").open_store()
        self.assertEqual(reopened.current_tick_id, store.current_tick_id)
        self.assertEqual(reopened.current_tick_id, tick)
        self.assertEqual(sorted(reopened.gwt_turns), [tick])
        self.assertEqual(reopened.states[-1], store.states[-1])
        self.assertEqual(reopened.get_knoxel_by_id(reply.id).content, "hi!")
        # The next tick continues after the saved one and never reuses it.
        self.assertEqual(reopened.begin_batch("cycle-2", new_tick=True).tick_id, tick + 1)


if __name__ == "__main__":
    unittest.main()

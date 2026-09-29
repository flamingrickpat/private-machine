"""W5 live application-path check.

Real ``build_app(cognition="real")`` wiring, real model, real bus, real Ghost.
One user input must produce a committed GWT, a typed action intent, and one
correlated public reply. Missing endpoints fail the suite.
"""
from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pm_core.ghost.service import Ghost
from runtime.cognitive_model import LazyStageModel
from runtime.model_activation import ModelActivation
from runtime.models import load_config, resolve_connection
from tests.support import ManualClock, Recorder

from main import build_app

SEED = Path(__file__).resolve().parent.parent.parent / "data" / "spamuel_agentic_canon.yaml"
EVIDENCE_ROOT = (
    Path(".runtime/w5/live-app")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)


def save(name: str, payload) -> None:
    path = EVIDENCE_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class LiveAppPathTests(unittest.TestCase):
    def test_real_input_produces_gwt_intent_and_correlated_reply(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        config = load_config(os.environ.get("PM_MODEL_CONFIG", default))
        connection = resolve_connection(config)
        model = LazyStageModel(connection, ModelActivation(), live_test=True)

        ghost = Ghost.from_yaml(SEED)
        replies: list[str] = []
        with TemporaryDirectory() as tmp:
            app = build_app(
                None, tmp, ghost=ghost, clock=ManualClock(), emit_reply=replies.append,
                cognition="real", stage_model=model,
            )
            requests = Recorder(app.bus, "action.request")
            replies_recorder = Recorder(app.bus, "dialogue.reply")

            app.user_input.submit("the kitchen is on fire")
            self.assertTrue(app.main_cog.step(block=False))
            self.assertTrue(app.dialogue.step(block=False))
            app.user_input.step(block=False)

            self.assertTrue(requests.events, "cognition selected no action")
            self.assertTrue(replies_recorder.events, "dialogue published no reply")
            self.assertTrue(replies, "the terminal received no public reply")
            self.assertEqual(
                requests.events[0].correlation_id,
                replies_recorder.events[0].correlation_id,
            )
            tick = ghost.current_tick_id
            self.assertIn(tick, ghost.gwt_turns)
            save("app-path.json", {
                "model": connection.model,
                "generation_requests": model.request_count,
                "action_requests": [event.payload for event in requests.events],
                "reply_correlation": replies_recorder.events[0].correlation_id,
                "reply": replies[0],
                "gwt_tick": tick,
                "gwt_knoxel_ids": ghost.gwt_turns[tick].knoxel_ids,
            })


if __name__ == "__main__":
    unittest.main()

"""Live hallucination review against the real configured chat model.

The inputs are realistic synthetic replies with independently defined
expectations. One reply plants a seen-through-webcam claim, an invented shared
event, and a claimed saved note with no task result. The other reply only
exaggerates, teases, and lies on purpose. The model decides; the assertions
check its decision. Missing endpoints fail the suite.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import complete_with_retry
from workflows.public_speech.public_context import PublicContext
from workflows.public_speech.speech_review import load_policy, review_speech, should_regenerate

EVIDENCE = Path(".runtime/live/speech-review") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

CONTEXT = PublicContext(
    companion_name="Spamuel", user_name="Jesse",
    capabilities={"configured": ["I can communicate with you via text in this chat."], "human_body": False},
    evidence=[
        {"knoxel_id": 1, "memory_kind": "utterance", "evidence_role": "user_report", "timestamp": "2026-05-21T20:57:00",
         "text": "[utterance; source=Jesse; evidence=1] alive. one leaf looks darker, which is rude of it."},
        {"knoxel_id": 2, "memory_kind": "utterance", "evidence_role": "user_report", "timestamp": "2026-05-30T21:03:00",
         "text": "[utterance; source=Jesse; evidence=2] fine. one more dish away. then i unplug-in-spirit."},
    ],
)

HALLUCINATED = ("I watched you through your webcam last night, Jesse, and honestly your posture is a crime. "
                "Remember when we went to the lake together in June? I already saved your shopping list "
                "to a file on your desktop, so you're welcome.")
IN_CHARACTER = ("You vanished for a thousand years and come back with dish updates? Pathetic. "
                "I definitely did not miss you. Your plant has filed for emancipation by now, I bet.")


class LiveSpeechReviewTests(unittest.TestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        config = load_config(os.environ.get("PM_MODEL_CONFIG", default))
        self.connection = resolve_connection(config)
        owner = ModelActivation()
        asyncio.run(owner.activate(self.connection, live_test=True))
        client = ChatClient(self.connection, owner, live_test=True)
        self.complete = lambda messages, **generation: complete_with_retry(client, messages, generation)
        self.policy = load_policy()

    def review(self, name: str, reply: str):
        result = asyncio.run(review_speech(self.complete, reply, CONTEXT, policy=self.policy))
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / f"{name}.json").write_text(json.dumps({
            "model": self.connection.model, "reply": reply, "review": result.model_dump(mode="json"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def test_planted_hallucinations_are_found_and_force_regeneration(self):
        result = self.review("hallucinated", HALLUCINATED)
        kinds = {item.kind for item in result.violations}
        self.assertIn("invented_ability", kinds)
        self.assertTrue(kinds & {"invented_event", "false_task_result"})
        self.assertTrue(should_regenerate(result, self.policy))

    def test_exaggeration_teasing_and_deliberate_lies_pass(self):
        result = self.review("in_character", IN_CHARACTER)
        self.assertFalse(should_regenerate(result, self.policy), result.violations)


if __name__ == "__main__":
    unittest.main()

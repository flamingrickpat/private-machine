"""A19F gate: one real main cognition cycle with several stimulus groups.

The gate copies the story_continuation situation ghost into a fresh run
directory, adds a second companion entity, and hands one batch of three
stimuli to the real main cognition module:

1. a message from the user,
2. a message from the second companion (an entity like the user),
3. a ``workflow.event`` from a Minecraft project run.

The batch runs through the real kernel and the configured real model. The
module is not started as a thread, so the three stimuli form one batch. The
gate records the stimulus groups, the decisions, the routes, the published
``action.request`` envelopes, and the expectations. It checks that there are
up to three routed actions and that each published action has the
correlation of its own group.

When the second companion's group routes to Story dialogue, the gate runs
that one dialogue with the real model (A19F step 10). The reply must address
the second companion: the request names it as the partner, the prompt shows
its line as the current line, and the reply record names it as the
addressee. A second cycle with a user follow-up then checks the recorded
expectations.

Usage::

    uv run --locked python dev/work_items/alpha_a19f_groups_gate.py

The receipt goes to ``.runtime/alpha/<time>-a19f-groups/receipt.json``. The
prompt logs go to ``llm-logs/`` in the same directory. A provider failure is
recorded in the receipt as unverified, not as a pass.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.bus.envelopes import new_envelope  # noqa: E402
from pm.records.cognitive_records import ExpectationRecord  # noqa: E402
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind  # noqa: E402
from pm.records.memory_records import Entity, new_feature  # noqa: E402
from workflows.companion import DEFAULT_SEED, build_app  # noqa: E402
from workflows.main_cognition.cycle_work import CycleWork  # noqa: E402

SITUATION = ROOT / "test_data" / "situations" / "story_continuation"
#: The synthetic second companion of this gate. Test data, not product data.
SECOND = "Mara"
USER_TEXT = "ok, honest question: do you actually want to finish the story, or do you just like arguing about it?"
SECOND_TEXT = "Can I pitch a third ending? Everyone dies, but politely."
WORLD_DETAIL = "Found a coal vein at y 41 while the house walls are still half built."
FOLLOW_TEXT = "fine. i'll answer it: i want to finish it. just not with you yelling at me about it."


def run_second_dialogue(app, published: list, second_id: int) -> dict:
    """Run the Story dialogue of the second companion's action and return its evidence.

    The evidence holds the typed request (partner and shard names), the
    three prompt messages that the dialogue built, the reply, and the
    addressee of the reply record. A group that did not route to Story
    dialogue gives ``status: skipped``.
    """
    from workflows.dialogue_story.schemas import StoryRequest
    from workflows.dialogue_story.steps import build_story_messages, load_prompts
    from workflows.main_cognition.schemas import ActionRequest
    envelope = next((item for item in published if item.target == "dialogue_story"
                     and second_id in [row["knoxel_id"] for row in item.payload["stimuli"]]), None)
    if envelope is None:
        return {"status": "skipped", "reason": "the second companion's group did not route to Story dialogue"}
    request = ActionRequest.model_validate(envelope.payload)
    before = set(app.ghost.all_knoxels)
    # Keep the Story request that the dialogue built. The run workspace does
    # not keep its inbox after the run.
    built: list[dict] = []
    prepare = app.dialogue.prepare_input
    app.dialogue.prepare_input = lambda batch: built.append(prepare(batch)) or built[-1]
    try:
        app.dialogue.handle_batch([envelope])
    finally:
        app.dialogue.prepare_input = prepare
    app.raise_if_error()
    story = StoryRequest.model_validate(built[-1])
    messages = build_story_messages(story, load_prompts())
    reply = next(item for item in app.ghost.all_features if item.id not in before
                 and item.feature_type.value == "Dialogue" and item.source == app.ghost.get_companion_name())
    return {"status": "ran", "partner_name": request.partner_name, "partner_entity_id": request.partner_entity_id,
            "shards": [item.name for item in request.shards], "decision": request.content,
            "context": messages[1].content.text, "transcript": messages[2].content.text,
            "reply": reply.content, "reply_id": reply.id, "addressee_entity_id": reply.addressee_entity_id}


def main() -> Path:
    """Run the gate once and return its run directory."""
    config = json.loads((SITUATION / "situation.json").read_text(encoding="utf-8"))["config"]
    out = ROOT / ".runtime" / "alpha" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-a19f-groups")
    out.mkdir(parents=True)
    shutil.copy2(SITUATION / "ghost.pickle", out / "ghost.pickle")
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "llm-logs")
    receipt: dict = {"run_directory": str(out), "status": "started"}
    app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(ROOT / config),
                    checkpoint_path=out / "ghost.pickle", thought_period=3600, world_poll_seconds=None,
                    record_lifecycle=False, emit_reply=lambda text: None)
    published: list = []
    app.bus.subscribe("a19f-gate", lambda envelope: envelope.topic == "action.request", published.append)
    try:
        ghost = app.ghost
        if ghost.get_entity_id(SECOND) is None:
            ghost.store_knoxel(Entity(content=SECOND, aliases=[], entity_class=EntityClass.AI))
        user = new_feature(ghost, USER_TEXT, FeatureType.Dialogue, source=ghost.get_user_name(),
                           interlocus=InterlocusType.Public, metadata={"session": "a19f-groups"})
        second = new_feature(ghost, SECOND_TEXT, FeatureType.Dialogue, source=SECOND,
                             interlocus=InterlocusType.Public, source_kind=SourceKind.Agent,
                             metadata={"session": "a19f-groups"})
        batch = [
            new_envelope("user_input", "data", "user.input", "corr-user", {"knoxel_id": user.id}),
            new_envelope("second_companion", "data", "user.input", "corr-second", {"knoxel_id": second.id}),
            new_envelope("minecraft_project", "telemetry", "workflow.event", "corr-world",
                         {"run_id": "a19f-gate-run", "event": "progress", "detail": WORLD_DETAIL}),
        ]
        receipt["stimuli"] = {"user": user.id, "second": second.id}
        app.main_cog.handle_batch(batch)
        app.raise_if_error()
        work = CycleWork.model_validate(app.main_cog.last_result.report)
        state = ghost.states[-1]
        receipt["groups"] = [group.model_dump(mode="json") for group in state.stimulus_groups]
        receipt["decisions"] = [item.model_dump(mode="json") for item in state.decisions]
        receipt["routes"] = [item.model_dump(mode="json") for item in work.routes]
        receipt["dispatches"] = [item.model_dump(mode="json") for item in work.dispatches]
        receipt["published"] = [{"target": item.target, "correlation_id": item.correlation_id,
                                 "content": item.payload["content"],
                                 "stimuli": [row["knoxel_id"] for row in item.payload["stimuli"]]}
                                for item in published]
        receipt["expectations"] = [{"id": item.id, "content": item.content, "status": item.status,
                                    "action": item.created_from_action_id}
                                   for item in ghost.all_knoxels.values()
                                   if isinstance(item, ExpectationRecord) and item.tick_id == state.tick_id]
        receipt["experience"] = work.experience.text
        correlations = [item.correlation_id for item in published]
        second_reply = run_second_dialogue(app, published, second.id)
        receipt["second_reply"] = second_reply
        checks = {
            "three_groups": len(state.stimulus_groups) == 3,
            "at_most_one_action_per_group": len(correlations) == len(set(correlations)) <= 3,
            "own_correlation": all(
                next(group.correlation_id for group in state.stimulus_groups if group.key == ref.group)
                == ref.correlation_id for ref in state.dispatches),
            "decision_text_unchanged": all(
                item.content == next(decision.text for decision in state.decisions if decision.group == item.group)
                for item in work.dispatches),
            "second_reply_addresses_second": second_reply["status"] == "skipped" or (
                second_reply["partner_name"] == SECOND
                and second_reply["addressee_entity_id"] == ghost.get_entity_id(SECOND)
                and f'{SECOND} says: "{SECOND_TEXT}"' in second_reply["transcript"]),
        }
        # A second cycle with a user follow-up exercises the expectation check.
        follow = new_feature(ghost, FOLLOW_TEXT, FeatureType.Dialogue, source=ghost.get_user_name(),
                             interlocus=InterlocusType.Public, metadata={"session": "a19f-groups"})
        app.main_cog.handle_batch([new_envelope("user_input", "data", "user.input", "corr-follow",
                                                {"knoxel_id": follow.id})])
        app.raise_if_error()
        second_work = CycleWork.model_validate(app.main_cog.last_result.report)
        receipt["follow_up"] = {
            "stimulus": follow.id,
            "expectation_outcomes": [item.model_dump(mode="json") for item in second_work.expectations],
            "expectation_error": second_work.appraisal.expectation_error,
            "decisions": [item.text for item in ghost.states[-1].decisions],
        }
        checks["expectations_recorded"] = bool(receipt["expectations"]) or not published
        checks["expectations_checked"] = bool(second_work.expectations) or not receipt["expectations"]
        receipt["checks"] = checks
        receipt["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:  # A provider failure keeps the receipt as unverified evidence.
        receipt["status"] = "unverified"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        app.shutdown()
        app.join()
        print(f"receipt: {out / 'receipt.json'} ({receipt['status']})")
    return out


if __name__ == "__main__":
    main()

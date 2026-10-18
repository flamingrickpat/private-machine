"""A19F step 7 and 8: groups, in-character decisions, routing, expectations, and the experience.

No live model. The real manifest, the real phases, the real host, and the
real MemoryStore commit run with the scripted model of ``cycle_support``.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureSubtype, FeatureType
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from pm.records.mental_state import FullMentalState
from pm.records.workspace import SELF_SHARD, ShardKey
from pm.runs.workflow_module import PROJECT_ROOT
from tests.support import ManualClock, Recorder
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel, last_user_text, scripted_tools
from workflows.dialogue_story.host import StoryDialogue
from workflows.main_cognition.expectations import outcome_of
from workflows.main_cognition.host import MainCogLoop
from workflows.main_cognition.phases.experience import continuation, load_experience_prompts, seed_phrases
from workflows.main_cognition.phases.route import body_lease, resolve_conflicts
from workflows.main_cognition.schemas import ActionDispatch, DecisionAnswer, DecisionCheck, OpenExpectation

MANIFEST = PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md"
STORY_MANIFEST = PROJECT_ROOT / "workflows" / "dialogue_story" / "dialogue_story.workflow.md"

USER_DECISION = "He has dodged choosing four times. Call him a coward for it, then admit that you like it unfinished."
OTHER_DECISION = "She interrupts again. Tell her to wait her turn, with a grin."
WORLD_DECISION = "The mine can wait. Ignore the report for now and keep your eyes on the conversation."


def ghost_with_entities() -> MemoryStore:
    """Return a store with entity records for the companion, the user, and a second companion."""
    ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
    for name, kind in (("Spamuel", EntityClass.Self), ("Jesse", EntityClass.Human), ("Mara", EntityClass.AI)):
        ghost.store_knoxel(Entity(content=name, aliases=[], entity_class=kind))
    return ghost


def say(ghost, speaker: str, text: str) -> int:
    """Store one dialogue line of ``speaker`` and return its id."""
    return ghost.store_knoxel(Feature(content=text, feature_type=FeatureType.Dialogue, causal=True,
                                      override_source=speaker)).id


def module_for(ghost, model: ScriptedModel, tmp: str) -> tuple[MainCogLoop, BusSystem, list]:
    bus = BusSystem()
    errors: list[BaseException] = []
    module = MainCogLoop("main_cog", bus, errors.append, ManualClock(), None,
                         working_target=MemoryBashTarget(ghost, "read"), workflow=MANIFEST,
                         tools_factory=scripted_tools(model), runtime_root=Path(tmp) / "runtime", ghost=ghost,
                         workflow_catalogue=[{"name": "dialogue_story", "description": "speak"},
                                             {"name": "minecraft", "description": "game body"}])
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors


def decide_by_part(messages) -> dict:
    """Decide by the part of the story that the instruction names."""
    text = last_user_text(messages)
    part = re.search(r"This choice is about (.+?)\. The new events", text).group(1)
    if part == "Jesse":
        return {"variants": [USER_DECISION]}
    if part == "Mara":
        return {"variants": [OTHER_DECISION]}
    return {"variants": [WORLD_DECISION]}


def route_reply(messages) -> dict:
    """Route the world decision to Ignore, and every other decision to a spoken reply."""
    if WORLD_DECISION in last_user_text(messages):
        return {"action_type": "Ignore", "workflow_name": ""}
    return {"action_type": "Reply", "workflow_name": "dialogue_story"}


def user(stimulus_id: int, correlation: str) -> Envelope:
    return Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                    correlation_id=correlation, payload={"knoxel_id": stimulus_id})


class GroupDecisionTests(unittest.TestCase):
    def test_one_decision_for_each_group_and_each_dispatch_carries_its_group(self):
        ghost = ghost_with_entities()
        jesse = say(ghost, "Jesse", "Fine, I still won't pick an ending.")
        mara = say(ghost, "Mara", "Can I add a third ending?")
        model = ScriptedModel({"decide": decide_by_part, "route": route_reply})
        batch = [user(jesse, "c-jesse"),
                 Envelope(id=new_id(), source="mara", kind="data", topic="user.input", correlation_id="c-mara",
                          payload={"knoxel_id": mara}),
                 Envelope(id=new_id(), source="minecraft_project", kind="telemetry", topic="workflow.event",
                          correlation_id="c-world", payload={"run_id": "run-1", "event": "progress",
                                                             "detail": "found coal"})]
        with TemporaryDirectory() as tmp:
            module, bus, errors = module_for(ghost, model, tmp)
            requests = Recorder(bus, "action.request")
            module.handle_batch(batch)
            self.assertEqual(errors, [])
        self.assertEqual(model.calls.count("decide"), 3)
        self.assertEqual(model.calls.count("route"), 3)
        state = ghost.states[-1]
        keys = [group.key for group in state.stimulus_groups]
        jesse_id, mara_id = ghost.get_entity_id("Jesse"), ghost.get_entity_id("Mara")
        self.assertEqual(keys, [ShardKey(kind="entity", ref=str(jesse_id)), ShardKey(kind="entity", ref=str(mara_id)),
                                ShardKey(kind="workflow", ref="minecraft_project")])
        self.assertEqual([item.text for item in state.decisions], [USER_DECISION, OTHER_DECISION, WORLD_DECISION])
        # The world decision routed to Ignore, so two groups act, each with its own correlation.
        self.assertEqual([(item.correlation_id, item.payload["content"]) for item in requests.events],
                         [("c-jesse", USER_DECISION), ("c-mara", OTHER_DECISION)])
        self.assertEqual([item.payload["stimuli"][0]["knoxel_id"] for item in requests.events], [jesse, mara])
        self.assertEqual([(ref.group, ref.correlation_id) for ref in state.dispatches],
                         [(keys[0], "c-jesse"), (keys[1], "c-mara")])
        actions = {item.id: item for item in ghost.all_knoxels.values() if isinstance(item, ActionSchemaRecord)}
        for ref, event in zip(state.dispatches, requests.events, strict=True):
            action = actions[ref.action_id]
            self.assertEqual(action.content, event.payload["content"])
            self.assertEqual(action.metadata["dispatch_correlation"], ref.correlation_id)
            self.assertEqual(action.metadata["group"], ref.group.model_dump(mode="json"))

    def test_an_insignificant_group_gets_no_decision(self):
        ghost = ghost_with_entities()
        jesse = say(ghost, "Jesse", "ok")
        mara = say(ghost, "Mara", "Can I add a third ending?")
        model = ScriptedModel({"decide": decide_by_part, "route": route_reply,
                               "triage": {"items": [{"knoxel_id": jesse, "category": "Insignificant"},
                                                    {"knoxel_id": mara, "category": "Significant"}]}})
        batch = [user(jesse, "c-jesse"), Envelope(id=new_id(), source="mara", kind="data", topic="user.input",
                                                  correlation_id="c-mara", payload={"knoxel_id": mara})]
        with TemporaryDirectory() as tmp:
            module, bus, errors = module_for(ghost, model, tmp)
            requests = Recorder(bus, "action.request")
            module.handle_batch(batch)
            self.assertEqual(errors, [])
        state = ghost.states[-1]
        self.assertEqual([group.insignificant for group in state.stimulus_groups], [True, False])
        self.assertEqual(model.calls.count("decide"), 1)
        self.assertEqual([item.text for item in state.decisions], [OTHER_DECISION])
        self.assertEqual([item.correlation_id for item in requests.events], ["c-mara"])

    def test_the_decision_text_reaches_the_dialogue_prompt_unchanged(self):
        ghost = ghost_with_entities()
        jesse = say(ghost, "Jesse", "Fine, I still won't pick an ending.")
        model = ScriptedModel({"decide": decide_by_part, "route": route_reply})
        prompts: list[str] = []

        async def story_complete(messages, **generation):
            prompts.append("\n".join(turn["content"] if isinstance(turn, dict) else turn.content.text
                                     for turn in messages))
            return 'Spamuel says: "Coward."'

        async def public_complete(messages, **generation):
            return '{"judgment": "accept", "reason": "grounded", "false_task_result": false, "findings": []}'

        with TemporaryDirectory() as tmp:
            module, bus, errors = module_for(ghost, model, tmp)
            dialogue = StoryDialogue("dialogue", bus, errors.append, ManualClock(), None,
                                     working_target=MemoryBashTarget(ghost, "read"), workflow=STORY_MANIFEST,
                                     tools_factory=lambda environment, batch: {"story_complete": story_complete,
                                                                               "public_complete": public_complete},
                                     runtime_root=Path(tmp) / "story", ghost=ghost)
            bus.subscribe(dialogue.module_id, dialogue.accepts, dialogue.deliver)
            module.handle_batch([user(jesse, "c-jesse")])
            self.assertTrue(dialogue.step(block=False))
            self.assertEqual(errors, [])
        self.assertEqual(len(prompts), 1)
        self.assertIn(f"What Spamuel decided to do with this reply: {USER_DECISION}", prompts[0])


class ExperienceTests(unittest.TestCase):
    def test_the_subjective_experience_enters_the_self_tail_and_the_later_prompts(self):
        ghost = ghost_with_entities()
        jesse = say(ghost, "Jesse", "Fine, I still won't pick an ending.")
        model = ScriptedModel({"experience": " and I am so done with this dance.", "decide": decide_by_part,
                               "route": route_reply})
        with TemporaryDirectory() as tmp:
            module, bus, errors = module_for(ghost, model, tmp)
            module.handle_batch([user(jesse, "c-jesse")])
            self.assertEqual(errors, [])
        experiences = [item for item in ghost.all_features if item.feature_type == FeatureType.SubjectiveExperience]
        self.assertEqual(len(experiences), 1)
        experience = experiences[0]
        self.assertTrue(experience.content.endswith(
            "I am so done with this dance. Now my mind turns to Jesse's words: 'Fine, I still won't pick an ending.'."))
        state = ghost.states[-1]
        self.assertIn(experience.id, state.workspace.self_shard.tail_ids)
        self.assertEqual(state.broadcasts[-1].experience_id, experience.id)
        self.assertEqual(ghost.gwt_turns[state.tick_id].experience_id, experience.id)
        self.assertIn(experience.id, ghost.gwt_turns[state.tick_id].shard_changes[0].tail_ids)
        # The goals and decide calls read the experience in the shared story prefix.
        for phase in ("goals", "decide"):
            story = model.messages[model.calls.index(phase)][1].content.text
            self.assertIn("I am so done with this dance.", story)

    def test_the_continuation_follows_the_seed_with_one_space(self):
        self.assertEqual(continuation("the word hits me", "I feel it...\nAnd right now,"), " the word hits me")
        self.assertEqual(continuation("I feel it... and more", "I feel it..."), " and more")
        self.assertEqual(continuation(" already spaced", "seed"), " already spaced")
        self.assertEqual(continuation("no seed", ""), "no seed")

    def test_seed_phrases_name_the_dominant_feeling_and_the_pressing_need(self):
        state = FullMentalState()
        state.state_emotions.boredom = 0.85
        state.state_needs.creative_expression = 0.15
        state.state_cognition.mental_aperture = -0.6
        phrases = seed_phrases(state, load_experience_prompts().seed_phrases)
        self.assertIn("My focus becomes narrow, zoning in on one thing...", phrases)
        self.assertIn("A feeling of boredom is intensely present...", phrases)
        self.assertIn("I feel a desperate need for creative expression right now...", phrases)


class ExpectationTests(unittest.TestCase):
    def test_an_expectation_is_recorded_for_a_routed_action_and_checked_later(self):
        ghost = ghost_with_entities()
        first = say(ghost, "Jesse", "Fine, I still won't pick an ending.")
        model = ScriptedModel({"decide": decide_by_part, "route": route_reply,
                               "anticipate": {"expectations": [{"content": "Jesse finally picks an ending.",
                                                                "affective_valence": 0.8, "urgency": 0.4}]}})
        with TemporaryDirectory() as tmp:
            module, bus, errors = module_for(ghost, model, tmp)
            module.handle_batch([user(first, "c-1")])
            self.assertEqual(errors, [])
            expectation = next(item for item in ghost.all_knoxels.values() if isinstance(item, ExpectationRecord))
            action = next(item for item in ghost.all_knoxels.values() if isinstance(item, ActionSchemaRecord))
            self.assertEqual(expectation.status, "pending")
            self.assertEqual(expectation.created_from_action_id, action.id)
            self.assertEqual(expectation.about_entity_ids, [ghost.get_entity_id("Jesse")])
            self.assertEqual(expectation.evidence_ids, [first])
            self.assertEqual(expectation.metadata["decision"], USER_DECISION)

            # The next message partly fulfills it. The check moves the fulfillment halfway
            # toward 0.9, so the expectation stays open and enters the CSM.
            model.answers["expectation_check"] = {"fulfillments": [
                {"expectation_id": expectation.id, "fulfillment_score": 0.9, "directness_score": 0.5}]}
            # A different new expectation, so the open one is not superseded.
            model.answers["anticipate"] = {"expectations": [{"content": "He takes the dragon ending back.",
                                                             "affective_valence": -0.3, "urgency": 0.2}]}
            second = say(ghost, "Jesse", "Okay, maybe the dragon one. Maybe.")
            module.handle_batch([user(second, "c-2")])
            self.assertEqual(errors, [])
        checked = ghost.get_knoxel_by_id(expectation.id)
        self.assertEqual(checked.status, "pending")
        self.assertAlmostEqual(checked.fulfillment, 0.45)
        self.assertEqual(checked.metadata["last_check"]["observation_ids"], [second])
        appraisal = [item for item in ghost.all_features if item.feature_subtype == FeatureSubtype.Appraisal][-1]
        self.assertAlmostEqual(appraisal.metadata["expectation_error"], 0.45 * 0.8 * 0.6, places=4)
        check_prompt = last_user_text(model.messages[model.calls.index("expectation_check")])
        self.assertIn(f"[id {expectation.id}] Jesse finally picks an ending.", check_prompt)
        item = ghost.states[-1].csm.items[expectation.id]
        self.assertIn("expectation", item.endorsers)

    def test_a_nearly_equal_new_expectation_supersedes_the_open_one(self):
        ghost = ghost_with_entities()
        model = ScriptedModel({"decide": decide_by_part, "route": route_reply,
                               "anticipate": {"expectations": [{"content": "Jesse finally picks an ending.",
                                                                "affective_valence": 0.8, "urgency": 0.4}]}})
        with TemporaryDirectory() as tmp:
            module, bus, errors = module_for(ghost, model, tmp)
            module.handle_batch([user(say(ghost, "Jesse", "No ending yet."), "c-1")])
            module.handle_batch([user(say(ghost, "Jesse", "Still no ending."), "c-2")])
            self.assertEqual(errors, [])
        records = sorted((item for item in ghost.all_knoxels.values() if isinstance(item, ExpectationRecord)),
                         key=lambda item: item.id)
        self.assertEqual([item.status for item in records], ["superseded", "pending"])
        self.assertEqual(records[0].observation_ids, [records[1].id])

    def test_outcome_rules(self):
        wanted = OpenExpectation(record_id=1, content="a yes", expected_valence=0.8)
        supported = outcome_of(wanted, 1.0, 1.0, [5])
        self.assertEqual((supported.status, supported.fulfillment), ("supported", 1.0))
        self.assertGreater(supported.valence_impact, 0.0)
        self.assertLess(supported.fear_impact, 0.0)
        contradicted = outcome_of(wanted.model_copy(update={"fulfillment": 0.5}), 0.0, 0.9, [5])
        self.assertEqual(contradicted.status, "contradicted")
        self.assertLess(contradicted.valence_impact, 0.0)
        self.assertGreater(contradicted.fear_impact, 0.0)
        unrelated = outcome_of(wanted, 0.5, 0.0, [5])
        self.assertEqual((unrelated.status, unrelated.change, unrelated.valence_impact), ("pending", 0.0, 0.0))


class ConflictTests(unittest.TestCase):
    GROUPS = [StimulusGroup(key=ShardKey(kind="entity", ref="1"), category="Moderate"),
              StimulusGroup(key=ShardKey(kind="workflow", ref="minecraft"), category="Critical"),
              StimulusGroup(key=SELF_SHARD, category="Significant")]

    def dispatch(self, action_id: str, group: ShardKey, workflow: str = "dialogue_story", **arguments):
        return ActionDispatch(action_id=action_id, workflow_name=workflow, action_type="ToolCall",
                              content=action_id, arguments=arguments, group=group, status="dispatched")

    def test_two_actions_cannot_claim_the_same_body_lease(self):
        talk = self.dispatch("talk", self.GROUPS[0].key, "minecraft")
        mine = self.dispatch("mine", self.GROUPS[1].key, "minecraft_project")
        project = self.dispatch("project", SELF_SHARD, "self_management", operation="start",
                                workflow_name="minecraft")
        resolved = resolve_conflicts([talk, mine, project], self.GROUPS)
        # The critical world group keeps the body; the others are rejected.
        self.assertEqual([item.status for item in resolved], ["rejected", "dispatched", "rejected"])
        self.assertIn("body lease minecraft already claimed by mine", resolved[0].rejection_reason)
        self.assertEqual(body_lease(self.dispatch("update", SELF_SHARD, "minecraft", run_id="run-1")), "")
        self.assertEqual(body_lease(self.dispatch("read", SELF_SHARD, "dialogue_assistant", minecraft_read="state")), "")

    def test_one_group_gets_at_most_one_action(self):
        first = self.dispatch("first", SELF_SHARD)
        second = self.dispatch("second", SELF_SHARD)
        other = self.dispatch("other", self.GROUPS[0].key)
        resolved = resolve_conflicts([first, second, other], self.GROUPS)
        self.assertEqual([item.status for item in resolved], ["dispatched", "rejected", "dispatched"])
        self.assertIn("already acts through first", resolved[1].rejection_reason)

    def test_a_dispatch_that_was_not_released_claims_nothing(self):
        queued = self.dispatch("queued", self.GROUPS[1].key, "minecraft").model_copy(update={"status": "queued"})
        later = self.dispatch("later", self.GROUPS[0].key, "minecraft")
        resolved = resolve_conflicts([queued, later], self.GROUPS)
        self.assertEqual([item.status for item in resolved], ["queued", "dispatched"])


class DecisionTextTests(unittest.TestCase):
    CHECK = DecisionCheck(workflow_names=["dialogue_story", "minecraft_project"]).model_dump()

    def test_planner_text_and_system_words_are_rejected(self):
        for text in ("Generate an in-character reply to Jesse.", "Validate the mutual agreement.",
                     "Respond to Jesse's preference.", "Route this to dialogue_story.", "Start a ToolCall."):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                DecisionAnswer.model_validate({"variants": [text]}, context=self.CHECK)

    def test_an_in_character_decision_and_an_empty_choice_are_valid(self):
        self.assertEqual(DecisionAnswer.model_validate({"variants": [USER_DECISION]}, context=self.CHECK).variants,
                         [USER_DECISION])
        self.assertEqual(DecisionAnswer.model_validate({"variants": []}, context=self.CHECK).variants, [])
        with self.assertRaises(ValidationError):
            DecisionAnswer.model_validate({"variants": ["one", "two"]}, context=self.CHECK)


if __name__ == "__main__":
    unittest.main()

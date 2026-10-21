"""Assistant dialogue contract tests.

The completion tool is a deterministic scripted sequence at the tool boundary.
This tests the real message builder, tool loop, denial, and receipt logic. Live
model behavior is covered separately in ``tests/live``.
"""
from __future__ import annotations

import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_workflows.runtime import WorkflowRuntime

from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.minecraft_body.mcp_client import build_minecraft_tools
from workflows.dialogue_assistant import steps
from workflows.dialogue_assistant.schemas import (
    AssistantMessage,
    AssistantReport,
    AssistantRequest,
    ToolDescriptor,
    ToolReceipt,
)
from pm.memory.bash_view import MemoryBashEnvironment, MemoryBashTarget

MANIFEST = PROJECT_ROOT / "workflows" / "dialogue_assistant" / "dialogue_assistant.workflow.md"


def example_request(**overrides) -> AssistantRequest:
    values = dict(
        companion_name="Spamuel",
        partner_name="Jesse",
        character_card="An acerbic, loyal companion.",
        abilities="- Cannot see, cannot move.",
        time="An ordinary evening.",
        prior_dialogue=[
            AssistantMessage(role="user", content="hey"),
            AssistantMessage(role="assistant", content="oh, you again"),
        ],
        current_user_message="did you start that task?",
        memories=["*(Fact: the task list is empty)*"],
        workspace="idle",
        tools=[
            ToolDescriptor(name="managed_start", description="Start a run."),
            ToolDescriptor(name="managed_status", description="Read a run."),
        ],
        tool_grant=["managed_start", "managed_status"],
    )
    values.update(overrides)
    return AssistantRequest(**values)


def completion(*messages) -> dict:
    return {"choices": [{"message": dict(messages[0])}]}


class MessageTests(unittest.TestCase):
    def test_roles_are_system_prior_then_user(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual(
            [message.role for message in messages],
            ["system", "user", "assistant", "user", "user"],
        )
        self.assertIn("Spamuel", messages[0].content)
        self.assertIn("Cannot see", messages[0].content)
        self.assertEqual("did you start that task?", messages[-1].content)

    def test_tool_schemas_are_function_shaped(self):
        schemas = steps.tool_schemas(example_request())
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["function"]["name"], "managed_start")

    def test_prior_tool_structure_is_preserved(self):
        request = example_request(prior_dialogue=[
            AssistantMessage(role="assistant", content="", tool_calls=[
                {"id": "c1", "function": {"name": "managed_status", "arguments": "{}"}},
            ]),
            AssistantMessage(role="tool", content='{"state": "completed"}', tool_call_id="c1"),
        ])
        messages = steps.build_messages(request, steps.load_prompts())
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].tool_calls[0]["function"]["name"], "managed_status")
        self.assertEqual(messages[2].role, "tool")
        self.assertEqual(messages[2].tool_call_id, "c1")

    def test_wire_message_drops_empty_tool_fields(self):
        plain = steps.wire_message(AssistantMessage(role="user", content="hi"))
        self.assertEqual(plain, {"role": "user", "content": "hi"})
        call = steps.wire_message(AssistantMessage(role="assistant", content="", tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "managed_list", "arguments": "{}"}},
        ]))
        self.assertIn("tool_calls", call)
        self.assertEqual(call["content"], "")
        result = steps.wire_message(AssistantMessage(role="tool", content="[]", tool_call_id="c1"))
        self.assertEqual(result, {"role": "tool", "content": "[]", "tool_call_id": "c1"})

    def test_wire_messages_never_contains_empty_tool_calls(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        for message in steps.wire_messages(messages):
            self.assertNotEqual(message.get("tool_calls"), [])


class ClaimTests(unittest.TestCase):
    def test_management_claim_needs_a_receipt(self):
        unsupported = steps.unsupported_management_claims(
            "I started the task for you.", [],
        )
        self.assertEqual(unsupported, ["started the task"])
        supported = steps.unsupported_management_claims(
            "I started the task for you.",
            [ToolReceipt(tool_call_id="c", name="managed_start", executed=True)],
        )
        self.assertEqual(supported, [])

    def test_denied_receipt_does_not_support_a_claim(self):
        denied = ToolReceipt(
            tool_call_id="c", name="managed_start", executed=False, error="denied",
        )
        self.assertIn(
            "started the task",
            steps.unsupported_management_claims("I started the task.", [denied]),
        )


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".runtime/w6-direct-tests")
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls: list[tuple] = []

    def _run(self, script, request: AssistantRequest, extra_tools: dict | None = None):
        responses = list(script)

        async def assistant_complete(messages, tools=None, **generation):
            return responses.pop(0)

        direct = {
            "managed_status": lambda run_id: {"run_id": run_id, "state": "completed"},
            "managed_terminate": lambda run_id: {"forbidden": True},
            "managed_start": lambda **kwargs: {"run_id": "new", "state": "accepted"},
            "managed_list": lambda: [],
            "managed_update": lambda run_id, instructions, request_id="": {"run_id": run_id},
            "managed_answer": lambda run_id, question_id, answer: {"run_id": run_id},
            "ghost_get": lambda knoxel_id: {},
            "ghost_search": lambda query: [],
            "ghost_gwt": lambda tick_id: {},
        }
        direct.update(extra_tools or {})
        # Keep the new manifest grant connected to the real client. These
        # retained dialogue fixtures do not execute the Minecraft read.
        minecraft = build_minecraft_tools()["minecraft_call"]
        direct["minecraft_observe"] = lambda include_image=False: minecraft("minecraft_observe", {"include_image": include_image})
        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=threading.Event(),
            tools={"assistant_complete": assistant_complete, "public_complete": assistant_complete, **direct},
            propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, self.root, namespace="assistant")
        result = runner.invoke(WorkflowRequest("assistant-1", request.model_dump(mode="json")), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        return AssistantReport.model_validate(result.report)

    def test_allowed_tool_runs_and_result_reaches_the_transcript(self):
        request = example_request(current_user_message="status?")
        report = self._run([
            completion({
                "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "managed_status", "arguments": '{"run_id": "r1"}'},
                }],
            }),
            completion({"content": "The run is completed, per the record."}),
        ], request)
        self.assertEqual(len(report.tool_receipts), 1)
        self.assertTrue(report.tool_receipts[0].executed)
        tool_message = next(m for m in report.messages if m.role == "tool")
        self.assertIn("completed", tool_message.content)
        self.assertEqual(report.final_text, "The run is completed, per the record.")

    def test_denied_tool_does_not_run(self):
        request = example_request(tool_grant=["managed_status"])
        report = self._run([
            completion({
                "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "managed_terminate", "arguments": '{"run_id": "r1"}'},
                }],
            }),
            completion({"content": "I did not do that."}),
        ], request)
        receipt = report.tool_receipts[0]
        self.assertFalse(receipt.executed)
        self.assertIn("not granted", receipt.error)
        self.assertEqual(report.final_text, "I did not do that.")

    def test_tool_turn_budget_ends_with_a_text_result(self):
        request = example_request(max_tool_turns=2)
        tool_call = {
            "content": "",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "managed_status", "arguments": '{"run_id": "r1"}'},
            }],
        }
        report = self._run([
            completion(tool_call),
            completion(tool_call),
            completion({"content": "I stopped looping and answered."}),
        ], request)
        self.assertEqual(report.final_text, "I stopped looping and answered.")
        # Only the first, allowed call produced a receipt.
        self.assertEqual(len(report.tool_receipts), 1)


class AssistantModuleTests(unittest.TestCase):
    def _module(self, root, responses, *, text_override=None):
        from pm.bus.bus import BusSystem
        from pm.bus.envelopes import Envelope
        from workflows.dialogue_assistant.host import AssistantDialogue
        from pm.memory.store import MemoryStore
        from pm.records.cycle_state import PersonaConfig
        from pm.runs.supervisor import ManagedRunSupervisor
        from pm.runs.workflow_spec import WorkflowSpec
        from workflows.dialogue_assistant.host import build_assistant_tools

        ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        supervisor = ManagedRunSupervisor(
            ghost, runtime_root=root / "runtime",
            workflows={
                "local_task": WorkflowSpec(
                    PROJECT_ROOT / "workflows" / "local_task" / "local_task.workflow.md",
                    namespace="local_task",
                ),
            },
            clock=lambda: 1.0,
            task_root=root / "tasks",
        )
        script = list(responses)

        class FakeModel:
            def build_async_tool(self, *, cancel=None):
                async def review(messages, **generation):
                    # Boundary fixture: the agentic review finds nothing.
                    return '{"judgment": "accept", "reason": "grounded", "false_task_result": false, "findings": []}'
                return review

            def build_tool_completion(self, *, cancel=None):
                async def complete(messages, tools=None, **generation):
                    return script.pop(0)

                return complete

        bus = BusSystem()
        module = AssistantDialogue(
            "dialogue", bus, self.fail, lambda: 1.0, None,
            working_target=MemoryBashTarget(ghost, mode="read"),
            workflow=MANIFEST,
            tools_factory=build_assistant_tools(FakeModel(), supervisor),
            runtime_root=root / "runtime", ghost=ghost,
        )
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        return module, bus, ghost, supervisor

    def test_empty_speech_keeps_completed_tool_receipts_without_replay(self):
        import json
        from pm.bus.envelopes import Envelope
        from tests.support import Recorder

        with TemporaryDirectory() as tmp:
            module, bus, ghost, supervisor = self._module(Path(tmp), [
                completion({"content": "", "tool_calls": [{"id": "list-once", "type": "function",
                    "function": {"name": "managed_list", "arguments": "{}"}}]}),
                *[completion({"content": ""}) for _ in range(8)],
            ])
            receipts = Recorder(bus, "action.receipt")
            module.mailbox.put(Envelope("empty", "test", "data", "action.request", "empty", {},
                                       target="dialogue_assistant"))
            self.assertTrue(module.step())
            failure = next(event for event in receipts.events if event.payload["status"] == "generation_failed")
            evidence = json.loads(Path(failure.payload["artifact_path"]).read_text(encoding="utf-8"))
            tools = json.loads(evidence["files"]["scratch/tool-receipts.json"])
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0]["tool_call_id"], "list-once")
            self.assertTrue(tools[0]["executed"])
            self.assertEqual(ghost.all_features, [])
            supervisor.join_all(timeout=10)

    def test_accepted_reply_is_committed_and_published(self):
        from pm.bus.envelopes import Envelope
        from tests.support import Recorder

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, bus, ghost, supervisor = self._module(root, [
                completion({"content": "Fine. Nothing to report."}),
            ])
            replies = Recorder(bus, "dialogue.reply")
            module.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-1",
                {"action_id": "action-0", "content": "chat", "arguments": {}},
                target="dialogue_assistant",
            ))
            self.assertTrue(module.step())
            self.assertEqual(module.last_result.status, WorkflowStatus.Completed)
            self.assertEqual(len(replies.events), 1)
            knoxel = ghost.get_knoxel_by_id(replies.events[0].payload["knoxel_id"])
            self.assertEqual(knoxel.content, "Fine. Nothing to report.")
            supervisor.join_all(timeout=10)

    def test_unsupported_management_claim_is_rejected(self):
        from pm.bus.envelopes import Envelope
        from tests.support import Recorder

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, bus, ghost, supervisor = self._module(root, [
                completion({"content": "I started the task already."}),
            ])
            receipts = Recorder(bus, "action.receipt")
            module.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-1",
                {"action_id": "action-0", "content": "chat", "arguments": {}},
                target="dialogue_assistant",
            ))
            with self.assertRaisesRegex(RuntimeError, "unsupported management claim"):
                module.step()
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "rejected")
            self.assertIn("unsupported management claim", receipts.events[0].payload["rejection_reason"])
            # Rejection records telemetry and leaves companion speech untouched.
            import pm.records.enums as enums
            dialogue = [k for k in ghost.all_features if k.feature_type == enums.FeatureType.Dialogue]
            self.assertEqual(len(dialogue), 0)
            supervisor.join_all(timeout=10)


if __name__ == "__main__":
    unittest.main()

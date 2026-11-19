"""Module-level tool factory for Assistant dialogue.

The application supplies one lazily-activated model, one managed-run
supervisor, and the scoped MemoryStore tools. The same management functions are
registered on the scoped MCP server supplied by ``mcp_services_factory``, so
the role can call them over MCP or directly with one implementation.
"""
from __future__ import annotations

from pm import vclock

from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import dataclass
from fastmcp import FastMCP, Client
from pydantic import JsonValue, TypeAdapter
from pm_workflows.working_target import WorkingEnvironment

from pm.host.mcp_listeners import McpServerDescriptor
from pm.memory.write_batch import WriteBatch
from workflows.dialogue_assistant.runtime import AssistantRuntime
from workflows.dialogue_assistant.schemas import McpEndpoint
from pm.memory.store_tools import build_mcp
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.management_api import management_mcp
from pm.bus.envelopes import Envelope
from pm.context_render.turns import Role
from workflows.dialogue_assistant.schemas import AssistantMessage, AssistantReport, AssistantRequest
from workflows.dialogue_assistant.steps import ToolExecutionError, load_prompts, unsupported_management_claims
from workflows.dialogue_assistant.tools import reply_tools
from workflows.main_cognition.schemas import ActionRequest
from workflows.public_speech.public_context import DialogueTurn, ability_lines, time_lines, PromptNames
from workflows.public_speech.public_dialogue import PublicDialogue

if TYPE_CHECKING:
    from pm.runs.supervisor import ManagedRunSupervisor


@dataclass(frozen=True, slots=True)
class AssistantMcpServices:
    supervisor: ManagedRunSupervisor
    minecraft_endpoint: str | None

    async def observe_minecraft(self, include_image: bool = False) -> JsonValue:
        if self.minecraft_endpoint is None:
            raise ToolExecutionError("No Minecraft body is configured for this character.")
        from workflows.minecraft_body.lease import BODY_LEASES
        with BODY_LEASES.command(self.minecraft_endpoint):
            async with Client(self.minecraft_endpoint) as client:
                result = await client.call_tool("minecraft_observe", {"include_image": False})
            if result.is_error:
                raise ToolExecutionError(f"Minecraft observation failed: {result}")
            return TypeAdapter[JsonValue](JsonValue).validate_python(result.structured_content if result.structured_content is not None else result.data)

    def services(self, environment: WorkingEnvironment) -> dict[str, FastMCP]:
        if not isinstance(environment, MemoryBashEnvironment):
            raise TypeError("Assistant tools require the invocation's frozen memory environment.")
        body = FastMCP("assistant_body")
        body.tool(name="minecraft_observe")(self.observe_minecraft)
        return {"workflow_management": management_mcp(self.supervisor),
                "ghost": build_mcp(environment), "body": body}


class AssistantDialogue(PublicDialogue):
    """Real Assistant dialogue. Tool-capable session with a separate contract.

    It accepts typed actions targeting ``dialogue_assistant``. On an accepted
    reply it commits one public dialogue feature and publishes
    ``dialogue.reply``. A management claim without an executed tool receipt is
    rejected instead of committed.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_assistant"

    def create_runtime(self, environment: WorkingEnvironment, proposal_batch: WriteBatch | None,
                       descriptors: dict[str, McpServerDescriptor] | None) -> AssistantRuntime:
        base = super().create_runtime(environment, proposal_batch, None)
        endpoints = tuple(McpEndpoint(url=descriptor["url"], type=descriptor["type"])
                          for descriptor in descriptors.values()) if descriptors is not None else ()
        return AssistantRuntime(environment=base.environment, stop_event=base.stop_event, tools=base.tools,
            propagate_python_errors=base.propagate_python_errors, agent_model=base.agent_model,
            child_factory=base.child_factory, classification_models=base.classification_models,
            classification_session=base.classification_session, model_session=base.model_session,
            embedding_service=base.embedding_service, managed_boundary=base.managed_boundary,
            tool_call_config=base.tool_call_config, memory_store=base.memory_store,
            memory_context=base.memory_context, mcp_endpoints=endpoints)

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict[str, object]:
        """Build the typed Assistant request from the typed action request.

        The reply answers the partner of the request. The prior chat window,
        the shard story, and the evidence come from the workspace shards of
        the cycle that chose the reply (``reply_context``). A terminal task
        receipt requests a report of existing work. It does not authorize
        another execution of the original user instruction.
        """
        assert self.ghost is not None
        action_request = ActionRequest.model_validate(batch[0].payload)
        action = (self.ghost.get_knoxel_by_id(action_request.knoxel_id)
                  if action_request.knoxel_id is not None else None)
        intent = str(action.content) if action is not None else action_request.content
        current_lines, planned, public = self.prepare_reply(action_request)
        current = "\n".join(item.text for item in current_lines)
        tools = reply_tools(terminal_receipt=bool(action_request.task_receipt))
        names = PromptNames(self.ghost.get_companion_name(), public.partner_name)
        request = AssistantRequest(
            task_result_supplied=bool(action_request.task_receipt),
            companion_name=self.ghost.get_companion_name(),
            partner_name=public.partner_name,
            character_card=self.ghost.get_simple_character_story_block(),
            abilities="\n".join(ability_lines(public)),
            time="\n".join(time_lines(vclock.utcnow().isoformat(), planned.dialogue, load_prompts(), names)),
            feelings=public.feelings,
            prior_dialogue=self._prior_dialogue(current, planned.dialogue, public.partner_name),
            current_user_message=current,
            memories=public.memories,
            workspace=public.workspace,
            tools=tools,
            tool_grant=[tool.name for tool in tools],
            selected_action=intent,
            public_context=public,
            minecraft_read=action_request.arguments.minecraft_read,
        )
        from pm.records.media import IMAGE_TAG, RecordImage
        texts = [request.current_user_message, request.workspace, request.selected_action]
        texts.extend(item.text for item in public.evidence)
        texts.extend(public.memories)
        if public.temporal_history is not None:
            texts.extend(item.content for item in public.temporal_history.items)
        texts.extend(public.goals)
        texts.extend(item.text for item in planned.dialogue)
        images: list[RecordImage] = []
        image_ids = sorted({match.group(1) for text in texts for match in IMAGE_TAG.finditer(text)})
        for image_id in image_ids:
            image = self.ghost.find_image(image_id)
            if image is None:
                raise ValueError(f"Prompt references missing image {image_id}.")
            images.append(image.model_copy(deep=True))
        request.images = tuple(images)
        return request.model_dump(mode="json")

    def _prior_dialogue(self, current: str, turns: Sequence[DialogueTurn],
                        partner_name: str) -> list[AssistantMessage]:
        """Turn the speech of the transcript into user and assistant chat messages.

        A line of the partner becomes a user message. A line of the companion
        becomes an assistant message. Private thoughts stay out.
        """
        assert self.ghost is not None
        turns = [turn for turn in turns if turn.kind == "speech"]
        messages: list[AssistantMessage] = []
        for index, turn in enumerate(turns):
            is_last = index == len(turns) - 1
            if is_last and turn.speaker == partner_name and turn.text == current:
                continue
            role = Role.Assistant if turn.speaker == self.ghost.get_companion_name() else Role.User
            messages.append(AssistantMessage(role=role, content=turn.text))
        return messages

    def publish_output(self, report: dict[str, object], batch: list[Envelope]) -> None:
        parsed = AssistantReport.model_validate(report)
        assert self._public_context is not None
        unsupported = (unsupported_management_claims(parsed.final_text, parsed.tool_receipts)
                       if not bool(self._public_context.task_receipt) else [])
        if not parsed.accepted or unsupported:
            reason = parsed.rejection_reason or (
                "unsupported management claim: " + ", ".join(unsupported)
            )
            self.reject_reply(batch, reason)
            return
        review = parsed.speech_review
        self.accept_reply(parsed.final_text, batch,
                          tool_receipts=[item.model_dump(mode="json") for item in parsed.tool_receipts],
                          review_reason=parsed.review_reason,
                          review_judgment=review.judgment if review is not None else "")

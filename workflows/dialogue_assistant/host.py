"""Module-level tool factory for Assistant dialogue.

The application supplies one lazily-activated model, one managed-run
supervisor, and the scoped MemoryStore tools. The same management functions are
registered on the scoped MCP server supplied by ``mcp_services_factory``, so
the role can call them over MCP or directly with one implementation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable
from functools import partial

from pm.memory.store_tools import build_mcp, build_tools
from pm.runs.management_api import build_management_tools, management_mcp
from pm.bus.envelopes import Envelope
from workflows.dialogue_assistant.schemas import AssistantMessage, AssistantReport, AssistantRequest
from workflows.dialogue_assistant.steps import ToolExecutionError, load_prompts, unsupported_management_claims
from workflows.dialogue_assistant.tools import reply_tools
from workflows.main_cognition.schemas import ActionRequest
from workflows.public_speech.public_context import ability_lines, time_lines
from workflows.public_speech.public_dialogue import PublicDialogue, _clip


def build_assistant_tools(model, supervisor, *, cancel_provider=None, minecraft_endpoint=None) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for Assistant dialogue."""

    def factory(environment, batch) -> dict:
        tools = {"assistant_complete": model.build_tool_completion()}
        tools["public_complete"] = model.build_async_tool()
        if cancel_provider is not None:
            cancel = cancel_provider()
            tools["assistant_complete"] = partial(tools["assistant_complete"], cancel=cancel)
            tools["public_complete"] = partial(tools["public_complete"], cancel=cancel)
        tools.update(build_management_tools(supervisor))
        tools.update(build_tools(environment, batch))
        if minecraft_endpoint is None:
            # The manifest always grants this tool name, so the role needs a
            # callable. Without a body it fails as a tool receipt, and the
            # assistant cannot claim an inventory that it never read.
            def minecraft_observe(include_image: bool = False):
                """Report that this character has no configured body."""
                raise ToolExecutionError(
                    "No Minecraft body is configured for this character. "
                    "Add the minecraft section to the model config.")
            tools["minecraft_observe"] = minecraft_observe
            return tools
        from workflows.minecraft_body.mcp_client import build_minecraft_tools
        from workflows.minecraft_body.lease import BODY_LEASES
        endpoint = minecraft_endpoint
        minecraft = build_minecraft_tools(endpoint)
        def minecraft_observe(include_image: bool = False):
            """Read current numeric state. The public text model gets no pixels."""
            with BODY_LEASES.command(endpoint):
                result = minecraft["minecraft_call"]("minecraft_observe", {"include_image": False})
                owner = BODY_LEASES.owner(endpoint)
                active = supervisor.receipts().get(owner)
            return {**{key: value for key, value in result.items() if key != "images"},
                    "body_owner": owner, "managed_run": active}
        tools["minecraft_observe"] = minecraft_observe
        return tools

    return factory


def build_assistant_mcp_factory(supervisor) -> Callable[[object], dict]:
    """Return ``mcp_services_factory(environment)`` for one Assistant run."""

    def factory(environment) -> dict:
        return {
            "workflow_management": management_mcp(supervisor),
            "ghost": build_mcp(environment),
        }

    return factory


__all__ = ["build_assistant_mcp_factory", "build_assistant_tools"]


class AssistantDialogue(PublicDialogue):
    """Real Assistant dialogue. Tool-capable session with a separate contract.

    It accepts typed actions targeting ``dialogue_assistant``. On an accepted
    reply it commits one public dialogue feature and publishes
    ``dialogue.reply``. A management claim without an executed tool receipt is
    rejected instead of committed.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_assistant"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        """Build the typed Assistant request from the typed action request.

        The reply answers the partner of the request. The prior chat window,
        the shard story, and the evidence come from the workspace shards of
        the cycle that chose the reply (``reply_context``). A terminal task
        receipt requests a report of existing work. It does not authorize
        another execution of the original user instruction.
        """
        action_request = ActionRequest.model_validate(batch[0].payload)
        action = (self.ghost.get_knoxel_by_id(action_request.knoxel_id)
                  if action_request.knoxel_id is not None else None)
        intent = str(action.content) if action is not None else action_request.content
        current_lines, planned, public = self.prepare_reply(action_request)
        current = "\n".join(item.text for item in current_lines)
        tools = reply_tools(terminal_receipt=bool(action_request.task_receipt))
        names = {"companion_name": self.ghost.get_companion_name(), "partner_name": public.partner_name}
        request = AssistantRequest(
            companion_name=self.ghost.get_companion_name(),
            partner_name=public.partner_name,
            character_card=self.ghost.get_simple_character_story_block(),
            abilities="\n".join(ability_lines(public)),
            time="\n".join(time_lines(datetime.utcnow().isoformat(), planned.dialogue, load_prompts(), names)),
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
        return request.model_dump(mode="json")

    def _prior_dialogue(self, current: str, turns, partner_name: str) -> list[AssistantMessage]:
        """Turn the speech of the transcript into user and assistant chat messages.

        A line of the partner becomes a user message. A line of the companion
        becomes an assistant message. Private thoughts stay out.
        """
        turns = [turn for turn in turns if turn.kind == "speech"]
        messages: list[AssistantMessage] = []
        for index, turn in enumerate(turns):
            is_last = index == len(turns) - 1
            if is_last and turn.speaker == partner_name and turn.text == current:
                continue
            role = "assistant" if turn.speaker == self.ghost.get_companion_name() else "user"
            messages.append(AssistantMessage(role=role, content=_clip(turn.text, 1500)))
        return messages

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        parsed = AssistantReport.model_validate(report)
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        unsupported = (unsupported_management_claims(parsed.final_text, parsed.tool_receipts)
                       if not self._public_context.task_receipt else [])
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

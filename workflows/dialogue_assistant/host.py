"""Module-level tool factory for Assistant dialogue.

The application supplies one lazily-activated model, one managed-run
supervisor, and the scoped MemoryStore tools. The same management functions are
registered on the scoped MCP server supplied by ``mcp_services_factory``, so
the role can call them over MCP or directly with one implementation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from pm.memory.store_tools import build_mcp, build_tools
from pm.runs.management_api import build_management_tools, management_mcp
from pm.bus.envelopes import Envelope
from workflows.dialogue_assistant.schemas import AssistantMessage, AssistantReport, AssistantRequest
from workflows.dialogue_assistant.steps import load_prompts, unsupported_management_claims
from workflows.dialogue_assistant.tools import reply_tools
from workflows.public_speech.public_context import ability_lines, time_lines
from workflows.public_speech.public_dialogue import PublicDialogue, _clip


def build_assistant_tools(model, supervisor) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for Assistant dialogue."""

    def factory(environment, batch) -> dict:
        tools = {"assistant_complete": model.build_tool_completion()}
        tools["public_complete"] = model.build_async_tool()
        tools.update(build_management_tools(supervisor))
        tools.update(build_tools(environment, batch))
        from workflows.minecraft_body.mcp_client import build_minecraft_tools, DEFAULT_MCP_URL
        from workflows.minecraft_body.lease import BODY_LEASES
        import os
        endpoint = os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)
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
        payload = batch[0].payload
        action = self.ghost.get_knoxel_by_id(payload["knoxel_id"]) if "knoxel_id" in payload else None
        intent = str(action.content) if action is not None else str(payload.get("content", ""))
        current = "\n".join(item["text"] for item in self.user_stimuli(payload))
        # The "assistant" context plan picks the prior chat window and the
        # memory lanes from the cycle's workspace (context_plans.yaml).
        planned = self.plan_context(payload, "assistant")
        public = self.public_context(payload, planned)
        # A terminal receipt requests a report of existing work. It does not
        # authorize another execution of the original user instruction.
        tools = reply_tools(terminal_receipt=bool(payload.get("task_receipt")))
        names = {"companion_name": self.ghost.get_companion_name(), "user_name": self.ghost.get_user_name()}
        request = AssistantRequest(
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            character_card=self.ghost.get_simple_character_story_block(),
            abilities="\n".join(ability_lines(public)),
            time="\n".join(time_lines(datetime.utcnow().isoformat(), planned.dialogue, load_prompts(), names)),
            feelings=public.feelings,
            prior_dialogue=self._prior_dialogue(current, planned.dialogue),
            current_user_message=current,
            memories=public.memories,
            workspace=public.workspace,
            tools=tools,
            tool_grant=[tool.name for tool in tools],
            selected_action=intent,
            public_context=public,
            minecraft_read=payload.get("arguments", {}).get("minecraft_read", ""),
        )
        return request.model_dump(mode="json")

    def _prior_dialogue(self, current: str, turns) -> list[AssistantMessage]:
        """Turn the planned speech window into user and assistant chat messages."""
        turns = [turn for turn in turns if turn.kind == "speech"]
        messages: list[AssistantMessage] = []
        for index, turn in enumerate(turns):
            is_last = index == len(turns) - 1
            if is_last and turn.speaker == self.ghost.get_user_name() and turn.text == current:
                continue
            role = "user" if turn.speaker == self.ghost.get_user_name() else "assistant"
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

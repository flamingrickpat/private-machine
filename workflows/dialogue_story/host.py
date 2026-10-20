"""Module-level tool factory for the real Story dialogue module.

The application supplies one lazily-activated model. This factory is called on
the module thread when a run starts, so the model is activated then and the
role receives an async completion tool bound to the already-active client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable
from functools import partial
from pm.bus.envelopes import Envelope
from workflows.dialogue_story.schemas import StoryReport, StoryRequest
from workflows.main_cognition.schemas import ActionRequest
from workflows.public_speech.public_dialogue import PublicDialogue


def build_story_tools(model, *, cancel_provider=None) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` for Story dialogue."""

    def factory(environment, batch) -> dict:
        complete = model.build_async_tool()
        if cancel_provider is not None:
            complete = partial(complete, cancel=cancel_provider())
        return {"story_complete": complete, "public_complete": complete}

    return factory


__all__ = ["build_story_tools"]


class StoryDialogue(PublicDialogue):
    """Real Story generation module. Accepts typed actions targeting it."""

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_story"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        """Build the typed Story request from the typed action request.

        The reply answers the partner of the request: the user or another
        entity. The intent is the decision text of the stored action. The
        transcript, the shard story, and the evidence come from the
        workspace shards of the cycle that chose the reply (``reply_context``).
        Partner lines that arrived after that cycle join the current message,
        so the reply answers the newest line and not an older one.
        """
        action_request = ActionRequest.model_validate(batch[0].payload)
        action = (self.ghost.get_knoxel_by_id(action_request.knoxel_id)
                  if action_request.knoxel_id is not None else None)
        intent = str(action.content) if action is not None else action_request.content
        spoken, planned, public = self.prepare_reply(action_request)
        request = StoryRequest(
            companion_name=self.ghost.get_companion_name(),
            partner_name=public.partner_name,
            character_card_story=self.ghost.get_simple_character_story_block(),
            supported_capabilities=[str(item) for item in self.ghost.ghost_config.supported_capabilities],
            unsupported_capabilities=[str(item) for item in self.ghost.ghost_config.unsupported_capabilities],
            capability_notes=[str(item) for item in self.ghost.ghost_config.capability_notes],
            now=datetime.utcnow().isoformat(),
            prior_dialogue=planned.dialogue,
            current_user_message="\n".join(item.text for item in spoken),
            current_message_id=spoken[-1].knoxel_id if spoken else None,
            feelings=public.feelings,
            regulation=public.regulation,
            goals=list(public.goals),
            memories=public.memories,
            workspace=public.workspace,
            selected_action=intent,
            public_context=public,
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        story = StoryReport.model_validate(report)
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        review = story.speech_review
        if not story.accepted:
            from workflows.public_speech.speech_review import block_reason
            reason = block_reason(review) if review is not None else "false task result"
            self.reject_reply(batch, reason)
            return
        self.accept_reply(story.candidate, batch,
                          review_reason=story.review_reason,
                          review_judgment=review.judgment if review is not None else "")

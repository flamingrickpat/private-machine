"""Module-level tool factory for the real Story dialogue module.

The application supplies one lazily-activated model. This factory is called on
the module thread when a run starts, so the model is activated then and the
role receives an async completion tool bound to the already-active client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable
from pm.bus.envelopes import Envelope
from workflows.dialogue_story.schemas import StoryReport, StoryRequest
from workflows.public_speech.public_dialogue import PublicDialogue, _recent_public_dialogue


def build_story_tools(model) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` for Story dialogue."""

    def factory(environment, batch) -> dict:
        complete = model.build_async_tool()
        return {"story_complete": complete, "public_complete": complete}

    return factory


__all__ = ["build_story_tools"]


# Verbatim prior dialogue window for Story generation.
STORY_DIALOGUE_TURNS = 24
STORY_TURN_CHARACTERS = 1200


class StoryDialogue(PublicDialogue):
    """Real Story generation module. Accepts typed actions targeting it."""

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_story"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        action = self.ghost.get_knoxel_by_id(payload["knoxel_id"]) if "knoxel_id" in payload else None
        intent = str(action.content) if action is not None else str(payload.get("content", ""))
        stimuli = [
            item for item in payload.get("stimuli", [])
            if str(item.get("text", "")).strip()
        ]
        current = str(stimuli[-1]["text"]) if stimuli else intent
        public = self.public_context(payload)
        current_ids = [int(item["knoxel_id"]) for item in stimuli if "knoxel_id" in item]
        request = StoryRequest(
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            character_card_story=self.ghost.get_simple_character_story_block(),
            supported_capabilities=[str(item) for item in self.ghost.ghost_config.supported_capabilities],
            unsupported_capabilities=[str(item) for item in self.ghost.ghost_config.unsupported_capabilities],
            capability_notes=[str(item) for item in self.ghost.ghost_config.capability_notes],
            now=datetime.utcnow().isoformat(),
            # Verbatim prior speech anchors the voice. Until the context
            # planner budgets it, keep a generous fixed window.
            prior_dialogue=_recent_public_dialogue(self.ghost, STORY_DIALOGUE_TURNS, exclude_ids=set(current_ids),
                                                   max_chars=STORY_TURN_CHARACTERS),
            current_user_message=current or intent or "(no message)",
            current_message_id=current_ids[-1] if current_ids else None,
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
        if not story.accepted:
            reason = "; ".join(violation.reason for violation in story.ontology.violations)
            reason = reason or "; ".join(story.ontology.fact_conflicts) or "ontology boundary"
            self.reject_reply(batch, reason)
            return
        self.accept_reply(story.candidate, batch)

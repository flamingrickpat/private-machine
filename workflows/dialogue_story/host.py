"""Run selected Story reply actions through generation, review, and publication.

``StoryDialogue`` receives an approved action from main cognition and freezes
its current partner lines, shard state, and retrieved memory into ``StoryRequest``.
The runtime supplies the conservative input ceiling. ``context.py`` fits complete
generation and review prompts; ``steps.py`` runs those model calls and records
their selections. If the current exchange cannot fit, the public host records
the failure and leaves it pending. Unexpected errors propagate.

The tool factory binds the application's model and cancellation event on the
module thread. The shared ``PublicDialogue`` base owns source guards, durable
utterance commits, and public delivery. A model draft has no public side effect
until that base accepts and commits the reviewed result.
"""

from __future__ import annotations

from workflows.public_speech.public_context import order_dialogue

from pm import vclock
from pm.memory.reconsolidation.assessment import MemoryWorkReader

from collections.abc import Sequence

from pydantic import JsonValue

from pm.bus.envelopes import Envelope
from workflows.dialogue_story.schemas import DialogueTurn, StoryGuidance, StoryReport, StoryRequest
from workflows.dialogue_story.guidance import accepted_story_guidance, retain_following_dialogue
from workflows.public_speech.context_planning import PlannedContext
from workflows.main_cognition.schemas import ActionRequest
from workflows.public_speech.public_dialogue import PublicDialogue

class StoryDialogue(PublicDialogue):
    """Real Story generation module. Accepts typed actions targeting it."""

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_story"
    _story_guidance: StoryGuidance | None = None
    memory_work_reader: MemoryWorkReader | None = None

    def reply_context(self, request: ActionRequest, late_ids: Sequence[int] = ()) -> PlannedContext:
        """Keep the original main frame and all speech that followed its snapshot."""
        assert self.ghost is not None
        planned = super().reply_context(request, list(late_ids))
        self._story_guidance = accepted_story_guidance(self.ghost, request)
        if self._story_guidance is not None:
            planned = retain_following_dialogue(self.ghost, request, planned, self._story_guidance)
        return planned

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict[str, JsonValue]:
        """Build the typed Story request from the typed action request.

        The reply answers the partner of the request: the user or another
        entity. The intent is the decision text of the stored action. The
        transcript, the shard story, and the evidence come from the
        workspace shards of the cycle that chose the reply (``reply_context``).
        Partner lines that arrived after that cycle join the current message,
        so the reply answers the newest line and not an older one.
        """
        assert self.ghost is not None
        action_request = ActionRequest.model_validate(batch[0].payload)
        action = (self.ghost.get_knoxel_by_id(action_request.knoxel_id)
                  if action_request.knoxel_id is not None else None)
        intent = str(action.content) if action is not None else action_request.content
        spoken, planned, public = self.prepare_reply(action_request)
        if self._story_guidance is not None:
            public.historical_sources = list(self._story_guidance.sources)
        spoken_ids = {item.knoxel_id for item in spoken}
        prior_dialogue = [turn for turn in planned.dialogue if turn.knoxel_id not in spoken_ids]
        # Coalesced inputs keep their own identities and timestamps. Joining
        # them under the last input's id can place an earlier line on the wrong
        # side of the main frame or repeat it alongside retrieved dialogue.
        with self.ghost.lock:
            for item in spoken[:-1]:
                source = self.ghost.snapshot({item.knoxel_id}).get(item.knoxel_id)
                prior_dialogue.append(DialogueTurn(speaker=public.partner_name, text=item.text,
                    knoxel_id=item.knoxel_id, timestamp=source.timestamp_world_begin.isoformat()))
        prior_dialogue = order_dialogue(prior_dialogue)
        request = StoryRequest(
            companion_name=self.ghost.get_companion_name(),
            partner_name=public.partner_name,
            character_card_story=self.ghost.get_simple_character_story_block(),
            supported_capabilities=[str(item) for item in self.ghost.ghost_config.supported_capabilities],
            unsupported_capabilities=[str(item) for item in self.ghost.ghost_config.unsupported_capabilities],
            capability_notes=[str(item) for item in self.ghost.ghost_config.capability_notes],
            now=vclock.utcnow().isoformat(),
            prior_dialogue=prior_dialogue,
            current_user_message=spoken[-1].text if spoken else "",
            current_message_id=spoken[-1].knoxel_id if spoken else None,
            feelings=public.feelings,
            regulation=public.regulation,
            goals=list(public.goals),
            memories=public.memories,
            workspace=public.workspace,
            selected_action=intent,
            guidance=self._story_guidance,
            public_context=public,
            input_budget_tokens=self.context_budget,
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

    def publish_output(self, report: dict[str, object], batch: list[Envelope]) -> None:
        story = StoryReport.model_validate(report)
        if self.memory_work_reader is not None:
            for attempt in story.attempts:
                if attempt.history_observation is not None:
                    self.memory_work_reader.observe_story(attempt.history_observation)
        review = story.speech_review
        if not story.accepted:
            from workflows.public_speech.speech_review import block_reason
            reason = block_reason(review) if review is not None else "false task result"
            self.reject_reply(batch, reason)
            return
        self.accept_reply(story.candidate, batch,
                          review_reason=story.review_reason,
                          review_judgment=review.judgment if review is not None else "")

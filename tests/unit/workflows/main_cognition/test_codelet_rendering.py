import unittest
from datetime import datetime

from pm.constants import PROMPT_MAX_IMAGES
from pm.context_render.prompt import RenderSettings
from pm.context_render.story import RecordView
from pm.context_render.turns import Media, Text
from pm.llm.multimodal import encode_messages
from pm.records.media import image_from_bytes
from tests.unit.pm.llm.test_multimodal import png
from workflows.main_cognition.codelet_registry import CodeletDefinition
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases.codelets import CodeletFrame, CodeletJob
from workflows.main_cognition.phases.model import phase_prompts
from workflows.main_cognition.story_prefix import StoryPrefix
from workflows.public_speech.situational_guidance import SituationalGuidance


class CodeletRenderingTests(unittest.TestCase):
    def test_codelet_view_images_reach_transport_once(self) -> None:
        shared = image_from_bytes(png(1))
        observation = image_from_bytes(png(2))
        work = CycleWork(request=CycleRequest(character_card=shared.tag, images=(shared, observation)))
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        frame = CodeletFrame(StoryPrefix(work, settings=settings), phase_prompts("codelets"))
        job = CodeletJob(view=f"{shared.tag} {observation.tag} [image:missing]")
        turns = frame.ask(CodeletDefinition(name="Observation"), job)
        messages = encode_messages(turns)["messages"]
        urls: list[str] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "image_url":
                        urls.append(part["image_url"]["url"])
        self.assertEqual(urls, [shared.data_url(), observation.data_url()])
        view_text = "".join(part.text for part in turns[-1].content.parts if isinstance(part, Text))
        self.assertIn("[image: unavailable]", view_text)

    def test_new_view_image_displaces_oldest_story_image_at_limit(self) -> None:
        images = tuple(image_from_bytes(png(index)) for index in range(PROMPT_MAX_IMAGES + 1))
        story = " ".join(image.tag for image in images[:-1])
        work = CycleWork(request=CycleRequest(character_card=story, images=images))
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        frame = CodeletFrame(StoryPrefix(work, settings=settings), phase_prompts("codelets"))
        turns = frame.ask(CodeletDefinition(name="Observation"), CodeletJob(view=images[-1].tag))
        encode_messages(turns)
        attached = [part.name for turn in turns for part in turn.content.parts if isinstance(part, Media)]
        self.assertEqual(attached, [image.id for image in images[1:]])

    def test_shared_story_keeps_guidance_sources_for_renderer_capture(self) -> None:
        situation = SituationalGuidance(as_of_utc=datetime(2026, 6, 9, 5), companion_name="Spamuel")
        work = CycleWork(request=CycleRequest(situational_guidance=[situation]))
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        story = StoryPrefix(work, settings=settings)
        self.assertEqual(story.render_input.situations, (situation,))
        turns = story.ask_with_shared_evidence("Observe the scene.")
        self.assertEqual(turns[:-1], story.turns)
        self.assertIn("World time: 2026-06-09T05:00 UTC.", turns[1].content.text)

    def test_rendering_a_view_reuses_the_selected_workspace_records(self) -> None:
        observation = image_from_bytes(png(1))
        work = CycleWork(request=CycleRequest(images=(observation,)))
        older = RecordView(record_id=7, kind="Feature", content="A torch lights the doorway.")
        newer = RecordView(record_id=8, kind="Feature", content=older.content)
        work.records[older.record_id] = older
        work.records[newer.record_id] = newer
        work.state.workspace.self_shard.tail_ids = [older.record_id, newer.record_id]
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        story = StoryPrefix(work, settings=settings)
        frame = CodeletFrame(story, phase_prompts("codelets"))

        turns = frame.ask(CodeletDefinition(name="Observation"), CodeletJob(view=observation.tag))

        self.assertIn("[source 8]", story.turns[1].content.text)
        self.assertNotIn("[source 7]", story.turns[1].content.text)
        self.assertEqual(turns[:-1], story.turns)
        images = [part.name for part in turns[-1].content.parts if isinstance(part, Media)]
        self.assertEqual(images, [observation.id])

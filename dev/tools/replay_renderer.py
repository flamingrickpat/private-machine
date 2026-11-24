from __future__ import annotations

from dataclasses import replace
import argparse
import pickle
from pathlib import Path
from typing import cast

from pm.context_render.prompt import RendererInput, RenderedPrompt, require_rendered_prompt
from pm.context_render.correction import (
    AgentReportCorrectionInput, JsonCorrectionInput, render_agent_report_correction, render_json_correction,
)
from pm.memory.media_alt import AltTextRenderInput
from pm.memory.reconsolidation.rendering import (
    SceneSummaryInput, TemporalSummaryInput, RecordedStateInput, FactExtractionInput,
    FactClassificationInput, NarrativeRenderInput, replay_memory_capture,
)
from simulation.rendering import SimulationPromptInput, render_simulation_prompt
from workflows.dialogue_story.standalone import StoryEvaluationInput, render_story_evaluation
from workflows.dialogue_story.context import StoryRenderInput, render_story
from workflows.dialogue_assistant.rendering import (
    AssistantRenderInput, AssistantContinuationInput, render_assistant_prompt, render_assistant_continuation,
)
from workflows.public_speech.speech_review import SpeechReviewRenderInput, render_speech_review
from workflows.main_cognition.rendering import CognitionStoryInput, render_cognition_story
from workflows.main_cognition.experience_rendering import ExperienceRenderInput, render_experience
from workflows.main_cognition.attention_rendering import AttentionRenderInput as MainAttentionRenderInput, render_attention
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.phases.triage import TriageRenderInput, render_triage
from workflows.main_cognition.codelet_agent import CodeletAgentRenderInput, render_codelet_agent
from workflows.main_cognition.retrieval_query import QueryRenderInput, render_retrieval_query
from workflows.cognitive_control.rendering import FocusRenderInput, render_focus
from workflows.inner_thought.rendering import replay_private_capture
from workflows.inner_thought.stages.nominate.steps import NominationRenderInput
from workflows.inner_thought.stages.trigger.steps import TriggerRenderInput, AttentionRenderInput
from workflows.inner_thought.stages.thought.steps import ThoughtRenderInput, GraphRenderInput
from workflows.inner_thought.stages.integrate.steps import IntegrationRenderInput
from workflows.inner_thought.stages.reflection.steps import ReflectionRenderInput
from workflows.tool_call.rendering import ToolSessionRenderInput, render_tool_session
from workflows.note_task.rendering import NoteRenderInput, render_note_selection
from workflows.minecraft_body.planning_prompt import MinecraftActionRenderInput, render_minecraft_action
from workflows.minecraft_project.planning.rendering import NativePlannerRenderInput, render_native_planner, ProjectDecisionRenderInput, render_project_decision
from workflows.minecraft.rendering import (LeafRenderInput, SpecialistRenderInput, ReconciliationEvidence, render_leaf_specification, render_specialist, render_reconciliation, render_gameplay_archive)
from workflows.minecraft_project.planning.context_contracts import GameplayHistory


type CapturedRendererInput = StoryEvaluationInput | StoryRenderInput | SpeechReviewRenderInput | AssistantRenderInput | AssistantContinuationInput | SimulationPromptInput | CognitionStoryInput | MainAttentionRenderInput | ExperienceRenderInput | TriageRenderInput | FocusRenderInput | CodeletAgentRenderInput | QueryRenderInput | NominationRenderInput | TriggerRenderInput | AttentionRenderInput | ThoughtRenderInput | GraphRenderInput | IntegrationRenderInput | ReflectionRenderInput | SceneSummaryInput | TemporalSummaryInput | RecordedStateInput | FactExtractionInput | FactClassificationInput | NarrativeRenderInput | AltTextRenderInput | JsonCorrectionInput | AgentReportCorrectionInput | ToolSessionRenderInput | NoteRenderInput | MinecraftActionRenderInput | NativePlannerRenderInput | ProjectDecisionRenderInput | LeafRenderInput | SpecialistRenderInput | ReconciliationEvidence | GameplayHistory


def replay_renderer(packet: RendererInput[CapturedRendererInput]) -> RenderedPrompt:
    settings = packet.settings
    match packet.input:
        case StoryEvaluationInput() as input:
            return render_story_evaluation(input, settings)
        case StoryRenderInput() as input:
            return render_story(input, settings)
        case SpeechReviewRenderInput() as input:
            return render_speech_review(input, settings)
        case AssistantRenderInput() as input:
            return render_assistant_prompt(input, settings)
        case AssistantContinuationInput() as input:
            return render_assistant_continuation(input, settings)
        case SimulationPromptInput() as input:
            return render_simulation_prompt(input, settings)
        case ExperienceRenderInput() as input:
            return render_experience(input, settings)
        case MainAttentionRenderInput() as input:
            return render_attention(input, settings).prompt
        case CognitionStoryInput() as input:
            return render_cognition_story(input, settings)
        case TriageRenderInput() as input:
            # A pickle can predate optional CycleWork fields. Revalidate its
            # persisted model data at this boundary just as runtime intake does.
            work = CycleWork.model_validate(input.work.model_dump())  # pm-lint: allow PM007 -- read historical renderer capture
            triage = render_triage(replace(input, work=work), settings)
            return require_rendered_prompt("triage", tuple(triage.turns), settings)
        case FocusRenderInput() as input:
            return render_focus(input, settings)
        case CodeletAgentRenderInput() as input:
            return render_codelet_agent(input, settings)
        case QueryRenderInput() as input:
            return render_retrieval_query(input, settings)
        case (NominationRenderInput() | TriggerRenderInput() | AttentionRenderInput() |
              ThoughtRenderInput() | GraphRenderInput() | IntegrationRenderInput() | ReflectionRenderInput()) as input:
            return replay_private_capture(input, settings)
        case (SceneSummaryInput() | TemporalSummaryInput() | RecordedStateInput() |
              FactExtractionInput() | FactClassificationInput() | NarrativeRenderInput() | AltTextRenderInput()) as input:
            return replay_memory_capture(input, settings)
        case JsonCorrectionInput() as input:
            return render_json_correction(input, settings)
        case AgentReportCorrectionInput() as input:
            return render_agent_report_correction(input, settings)
        case ToolSessionRenderInput() as input:
            return render_tool_session(input, settings)
        case NoteRenderInput() as input:
            return render_note_selection(input, settings)
        case MinecraftActionRenderInput() as input:
            return render_minecraft_action(input, settings)
        case LeafRenderInput() as input:
            return render_leaf_specification(input, settings)
        case SpecialistRenderInput() as input:
            return render_specialist(input, settings)
        case ReconciliationEvidence() as input:
            return render_reconciliation(input, settings)
        case GameplayHistory() as input:
            return render_gameplay_archive(input, settings)
        case NativePlannerRenderInput() as input:
            return render_native_planner(input, settings)
        case ProjectDecisionRenderInput() as input:
            return render_project_decision(input, settings)
        case _:
            raise TypeError(f"Unknown renderer input type: {type(packet.input).__qualname__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    with arguments.input.open("rb") as stream:
        packet = cast(RendererInput[CapturedRendererInput], pickle.load(stream))
    output = replay_renderer(packet)
    second = replay_renderer(packet)
    if output != second:
        raise AssertionError("The renderer returned different results for identical inputs.")
    expected_path = arguments.input.with_name("output.pickle")
    if expected_path.exists():
        with expected_path.open("rb") as stream:
            expected = cast(RenderedPrompt, pickle.load(stream))
        # Slotted dataclass pickles contain only the fields present at capture time.
        try:
            expected.visible_source_ids
        except AttributeError:
            expected = RenderedPrompt(expected.operation, expected.turns, expected.raw_text,
                expected.estimated_tokens, expected.input_ceiling, expected.evidence, expected.selected_groups,
                expected.temporal_history, expected.temporal_history_measurement)
            # Those fields had no recorded baseline. All captured fields must still match.
            comparison = replace(output, visible_source_ids=(), required_source_ids=())
        else:
            comparison = output
        if comparison != expected:
            raise AssertionError("Current output differs from the captured output.")
    if arguments.output is not None:
        with arguments.output.open("xb") as stream:
            pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{output.operation}: {output.estimated_tokens}/{output.input_ceiling} estimated tokens; identical replay")


if __name__ == "__main__":
    main()

from enum import StrEnum

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


class NarrativeTypes(StrEnum):
    SelfImage = "SelfImage"
    PsychologicalAnalysis = "PsychologicalAnalysis"
    Relations = "Relations"
    ConflictResolution = "ConflictResolution"
    EmotionalTriggers = "EmotionalTriggers"
    GoalsIntentions = "GoalsIntentions"

def get_narrative_analysis(type: NarrativeTypes, target: str, recent_narrative: str, narration: str, prompt: str):
    sysprompt = "You are a helpful assistant. You help the user analyze the characters in stories."

    if recent_narrative == "":
        prompt = [
            ("system", sysprompt),
            ("user", f"### BEGIN STORY\n{narration}\n### END STORY\n{prompt}"),
        ]
    else:
        prompt = [
            ("system", sysprompt),
            ("user", f"### BEGIN PREVIOUS ANALYSIS\n{recent_narrative}\n### END PREVIOUS ANALYSIS\n### BEGIN STORY\n{narration}\n### END STORY\n{prompt}"),
        ]

    content = controller.completion_text(LlmPreset.Auxiliary, prompt, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=4096))
    return content

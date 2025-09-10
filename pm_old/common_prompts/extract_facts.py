from typing import List

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


class Facts(BaseModel):
    facts: List[str] = Field(default_factory=list, description="list of facts extracted from conversation")

class TimeInvariance(BaseModel):
    time_variance: float = Field(description="0 for time-invariant and 1 for time-variant. meaning 0 is for facts true at all times, and 1 for statements only correct right now")

def extract_facts(block: str):
    sysprompt = "You are a helpful assistant. You extract facts from a conversation."
    prompt = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n"),
    ]

    _, calls = controller.completion_tool(LlmPreset.ExtractFacts, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[Facts])
    facts = []
    for fact in calls[0].facts:
        sysprompt = "You are a helpful assistant. You extract determine if a fact is time-variant or time-invariant. Use 0 for facts that are true at all times, and 1 for facts only true right now."
        prompt = [
            ("system", sysprompt),
            ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n### BEGIN FACT\n{fact}\n### END FACT\nIs the extracted fact from the messages time-independent (0) or time-dependent (1)?"),
        ]

        _, calls2 = controller.completion_tool(LlmPreset.ExtractFacts, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[TimeInvariance])
        time_variance = calls2[0].time_variance
        facts.append((fact, time_variance))

    return facts

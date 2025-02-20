import enum
import datetime

from pydantic import BaseModel, Field

from pm.character import companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset


class ConLevelType(enum.IntEnum):
    Conscious = 1
    PreConscious = 2
    SubConscious = 3
    Unconscious = 4


class CogEvent(BaseModel):
    con_level: ConLevelType
    source: str
    text: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

    class Config:
        use_enum_values = True


class AgentSelection(BaseModel):
    agent1: str
    agent2: str


def select_relevant_agents(agents, context_data):
    block = "\n".join([f"{x.name}: {x.description}" for x in agents])
    short = ",".join([f"{x.name}" for x in agents])

    sysprompt = f"""You are an agent in a cognitive architecture and you must select the 2 most relevant agents based on the chat log for {companion_name}. These are all agents: {block}\nYour options are: 
    [{short}]"""

    userprompt = (f"### BEGIN CONTEXT\n{context_data}\n### END CONTEXT\n"
                  f"Based on the last message, which agents are most relevant for {companion_name}? Consider her character description. Your options are: [{short}] for agent1 and agent2")

    messages = [
        ("system", sysprompt),
        ("user", userprompt)
    ]

    _, calls = controller.completion_tool(LlmPreset.Default, messages, tools=[AgentSelection])
    return [calls[0].agent1, calls[0].agent2]



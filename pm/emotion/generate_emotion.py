import copy
import datetime
import enum
import os.path
from typing import List

from pydantic import BaseModel, Field

from pm.character import char_card_3rd_person_emotional, char_card_3rd_person_neutral, companion_name
from pm.controller import controller

from pm.agents.agent import Agent
from pm.agents_gwt.agent_manager_pregwt import execute_boss_worker_chat_pregwt
from pm.agents_gwt.gwt_layer import pregwt_to_internal_thought
from pm.agents_gwt.schema_main import get_initial_dynamic_schema
from pm.agents_gwt.schema_meta import get_meta_layer
from pm.agents_gwt.summarize_pre_gwt import summarize_pre_gwt
from pm.llm.base_llm import LlmPreset, CommonCompSettings

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

    _, calls = controller.completion_tool(LlmPreset.Emotion, messages, tools=[AgentSelection])
    return [calls[0].agent1, calls[0].agent2]


def generate_first_person_emotion(ctx_msgs: str, last_message: str):
    context = f"""### BEGIN CHARACTER CARD
{char_card_3rd_person_emotional}
### END CHARACTER CARD

### BEGIN CONVERSATION HISTORY
{ctx_msgs}
### END CONVERSATION HISTORY

### BEGIN LAST MESSAGE
{last_message}
### END LAST MESSAGE

You will focus on emotions that are evoked by the last message! Previous messages are only here for context, they have already been evaluated.
FOCUS ON THE LAST MESSAGE!
"""

    context_neutral = f"""### BEGIN CHARACTER CARD
{char_card_3rd_person_neutral}
### END CHARACTER CARD

### BEGIN CONVERSATION HISTORY
{ctx_msgs}
### END CONVERSATION HISTORY

### BEGIN LAST MESSAGE
{last_message}
### END LAST MESSAGE

You will focus on emotions that are evoked by the last message! Previous messages are only here for context, they have already been evaluated.
FOCUS ON THE LAST MESSAGE!
"""
    layer = get_initial_dynamic_schema()
    gwt_output: List[CogEvent] = []
    for subsystem in layer.get_subsystems():
        agents = []
        reso = select_relevant_agents(subsystem.agents, context)
        selected_agents = [a for a in subsystem.agents if a.name in reso]

        for agent in selected_agents:
            prompt = layer.common_prefix + "\n" + subsystem.prompt_prefix + "\n" + agent.prompt
            agents.append(Agent(
                id=agent.name,
                name=agent.name,
                description=agent.description,
                goal=agent.goal,
                system_prompt=prompt
            ))

        state = {}
        result = execute_boss_worker_chat_pregwt(state, context, subsystem.goal, agents, min_confidence=0.8, max_rounds=8, round_robin=True, llmpreset=LlmPreset.Emotion)

        # add subagent messages
        subagent_messages = []
        for msg in result.agent_messages:
            subagent_msg = CogEvent(con_level=ConLevelType.Unconscious, source=msg.name, text=msg.text)
            subagent_messages.append(f"{msg.name}: {subagent_msg}")

        # get task focused summary
        recommendation = summarize_pre_gwt(context, "\n".join(subagent_messages), subsystem.goal)
        subsystem_recommendation = CogEvent(con_level=ConLevelType.SubConscious, source=subsystem.name, text=recommendation)
        gwt_output.append(subsystem_recommendation)

    gwt_as_thought = "\n".join([f"{x.source}: {x.text}" for x in gwt_output])
    gwt_layer_output = pregwt_to_internal_thought(context_neutral, gwt_as_thought)
    return gwt_layer_output

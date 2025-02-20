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
from pm.agents_gwt.schema_main import get_initial_dynamic_schema, GwtLayer
from pm.agents_gwt.schema_meta import get_meta_layer
from pm.agents_gwt.summarize_pre_gwt import summarize_pre_gwt
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.subsystems.subsystem_base import CogEvent, ConLevelType, select_relevant_agents


def execute_subsystem_thought(ctx_msgs: str, last_message: str) -> str:
    context_neutral = f"""### BEGIN CHARACTER CARD
{char_card_3rd_person_emotional}
### END CHARACTER CARD

### BEGIN CONVERSATION HISTORY
{ctx_msgs}
### END CONVERSATION HISTORY

You will focus on possible thoughts that could arise based on the messages {companion_name} received.
"""

    layer = get_initial_dynamic_schema()
    gwt_output: List[CogEvent] = []
    subsystem = layer.get_subsystems(GwtLayer.thought_generation)
    agents = []
    while True:
        reso = select_relevant_agents(subsystem.agents, context_neutral)
        selected_agents = [a for a in subsystem.agents if a.name in reso]
        if len(selected_agents) == 2:
            break

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
    recommendation = summarize_pre_gwt(context_neutral, "\n".join(subagent_messages), subsystem.goal)
    subsystem_recommendation = CogEvent(con_level=ConLevelType.SubConscious, source=subsystem.name, text=recommendation)
    gwt_output.append(subsystem_recommendation)
    return subsystem_recommendation.text


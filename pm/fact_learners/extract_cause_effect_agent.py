import json
import time
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Optional, Any, List, Tuple, Union

from sqlmodel import Field, Session, SQLModel, create_engine, select, col

from pm.agents.agent import Agent
from pm.agents_historized.agent_historized_simple import AgentHistorizedSimpel
from pm.common_prompts.rate_agent_output import rate_agent_output
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from rich import print as pprint

from pydantic import BaseModel, Field

from pm.database.tables import EventCluster, Event, Cluster

sysprompt = "You are a helpful assistant for analyzing text."

prompt = []

class CauseEffect(BaseModel):
    cause: str = Field(default="", description="Triggering action or event.")
    effect: str = Field(default="", description="Resulting outcome.")
    #temporality: float = Field(default=0.5, description="0 = short-term, 1 = long-term.")
    importance: str = Field(default="mid", description="low = minor, medium = moderate, high = major")
    #duration: float = Field(default=0.5, description="one-time, recurring, persistent")

class CauseEffects(BaseModel):
    cause_effect_can_be_extracted: bool = Field(description="if false, leave other fields empty")
    cause_effects: List[CauseEffect] = Field(default_factory=list, description="list of cause effect objects (if there are some)")

ces = CauseEffects(
    cause_effect_can_be_extracted=True
)

sysprompt_convo_to_text = """fYou are analyzing a conversation to extract cause-effect relationships.

Situation:
- Rick (user) and Emmy (AI chatbot) are communicating via text
- Rick is a person
- Emmy is a highly advanced chatbot that mimics human cognition and emotion perfectly

Your task:
- Identify **causes** (an action, event, or decision).
- Identify **effects** (the direct consequences of the cause).
- Ensure the effects explicitly mention how they impact:
  - The AI (Emmy)
  - The user (Rick)
  - The world (if applicable)
- **Prioritize important cause-effect pairs** and omit trivial ones.

### EXAMPLES
Rick: "Hey Emmy, do you know what the weather is like?"
Emmy: "It's sunny outside!"
CauseEffect:
- Cause: "Rick asks about the weather."
- Effect: "Emmy provides weather information."
- Importance: Low

Rick: "I feel really unmotivated today..."
Emmy: "That’s okay! Want me to help you plan something small?"
CauseEffect:
- Cause: "Rick expresses low motivation."
- Effect: "Emmy offers encouragement and suggests an action."
- Importance: High

Rick: "Oh no, I think I broke something!"
Emmy: "Don’t worry, let’s check what happened!"
CauseEffect:
- Cause: "Rick reports an issue."
- Effect: "Emmy reassures him and offers help."
- Importance: Medium
"""



sysprompt_json_converter = """You are a structured data converter.

Your task:
- Convert the extracted cause-effect relationships into **valid JSON** format.
- Use the following structure:
  {
    "cause_effect_can_be_extracted": true,
    "cause_effects": [
      {
        "cause": "...",
        "effect": "...",
        "importance": "low|medium|high"
      },
      ...
    ]
  }
- **Ensure correctness**: If no cause-effect pairs were found, set `"cause_effect_can_be_extracted": false` and return an empty list.

### EXAMPLES
#### Input:
CauseEffect:
- Cause: "Rick asks about the weather."
- Effect: "Emmy provides weather information."
- Importance: Low

#### Output:
{
  "cause_effect_can_be_extracted": true,
  "cause_effects": [
    {
      "cause": "Rick asks about the weather.",
      "effect": "Emmy provides weather information.",
      "importance": "low"
    }
  ]
}
"""

def extract_cas(block: str) -> List[CauseEffect]:
    mt = 4096
    convo_to_text_agent = AgentHistorizedSimpel(
        id="convo_to_text_agent",
        sysprompt=sysprompt_convo_to_text
    )

    query = f"""### BEGIN CONVERSATION
{block}
### END CONVERSATION

Extract cause-effect relationships in a structured list."""

    prompt = convo_to_text_agent.get_prompt(query, mt)
    content = controller.completion_text(LlmPreset.CauseEffect, prompt,  comp_settings=CommonCompSettings(temperature=0.2, max_tokens=1024))
    rating = rate_agent_output(sysprompt_convo_to_text, query, content)
    convo_to_text_agent.add_messages(query, content, rating)


    query = f"""### BEGIN TEXT
{content}
### END TEXT

Convert this to the specified JSON format.
"""

    text_to_cause_effect_json_agent = AgentHistorizedSimpel(
        id="text_to_cause_effect_json_agent",
        sysprompt=sysprompt_json_converter
    )
    res = []
    prompt = text_to_cause_effect_json_agent.get_prompt(query, mt)
    _, calls = controller.completion_tool(LlmPreset.CauseEffect, prompt,  comp_settings=CommonCompSettings(temperature=0.2, max_tokens=1024), tools=[CauseEffects])
    content = json.dumps(calls[0].model_dump(), indent=2)
    if calls[0].cause_effect_can_be_extracted:
        for ce in calls[0].cause_effects:
            if ce.cause != "" and ce.effect != "":
                ces.cause_effects.append(ce)
                res.append(ce)

    rating = rate_agent_output(sysprompt_json_converter, query, content)
    text_to_cause_effect_json_agent.add_messages(query, content, rating)
    return res

if __name__ == '__main__':
    controller.init_db()
    session = controller.get_session()

    clusters = session.exec(select(Cluster).order_by(Cluster.id)).fetchall()

    for cluster in clusters:
        print(cluster.id)
        events = session.exec(select(Event).where(col(Event.id).in_(select(EventCluster.cog_event_id).where(EventCluster.con_cluster_id == cluster.id))).order_by(
            Event.id)).fetchall()
        block = []
        for event in events:
            block.append(f"{event.source}: {event.to_prompt_item()[0].to_tuple()[1]}")
        strblock = "\n".join(block)
        extract_cas(strblock)
        controller.commit_db()
        #exit(1)













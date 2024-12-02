import json
import logging
from typing import List

from pydantic import BaseModel, Field

from pm.agents.generate_thought_aspects import PersonalityAspects
from pm.controller import controller
from pm.database.db_helper import get_facts_str
from pm.agents_dynamic.agent_manager import Agent, execute_boss_worker_chat, BossWorkerChatResult
from pm.llm.base_llm import CommonCompSettings, LlmPreset

logger = logging.getLogger(__name__)


class MemoryQuery(BaseModel):
    """
    Hybrid vector search for the knowledge database.
    """
    query: str = Field(description="query phrase")

    def execute(self, state):
        logger.info(json.dumps(self.model_dump(), indent=2))
        return get_facts_str("", self.query)


def thought_contemplation_dialog(context: str, goal: str, aspects: PersonalityAspects) -> BossWorkerChatResult:
    agents = []

    for i in range(2):
        prompt = (f"{controller.config.character_card_assistant}\n"
                   f"You represent a special aspect of {controller.config.companion_name}:"
                   f"{aspects.aspects[i].perspective}"
                   f"These are your core qualities:"
                   f"{aspects.aspects[i].core_quality}"
                   f"Discuss with these aspect of yourself another aspect of yourself this thought: {goal}"
                   f"Please keep your responses short and concise.")

        # Pessimistic Agent
        agents.append(Agent(
            name=f"{controller.config.companion_name} - {aspects.aspects[i].title}",
            description=f"{controller.config.companion_name} - {aspects.aspects[i].title}",
            system_prompt=prompt,
            tools=[],
            functions=[],
            comp_settings=CommonCompSettings(
                temperature=0.7,
                max_tokens=128
            ),
        ))

    # Execute the boss-worker chat with the agents
    result = execute_boss_worker_chat(context, goal, agents, min_confidence=0.85, convert_to_memory=False)
    return result


import random
from typing import List

from pm.agent_schemas.schema_base import DynamicAgent
from pm.agent_schemas.schema_main import get_agent_schema_main
from pm.agents_manager.agent import Agent
from pm.agents_manager.agent_group_conversation import agent_group_conversation
from pm.character import char_card_3rd_person_emotional
from pm.common_prompts.select_relevant_agents import select_relevant_agents
from pm.common_prompts.summarize_agent_conversation import summarize_agent_conversation
from pm.ghost.mental_state import EmotionalAxesModel
from pm.llm.base_llm import LlmPreset

from pm.subsystem.sensation_evaluation.emotion.emotion_agent_group import (agent_joy, agent_hate, agent_love, agent_pride, agent_shame, agent_trust, agent_dysphoria, agent_mistrust, agent_anxiety,
                                                                           agent_disgust, emotion_agent_group_prefix, emotion_agent_group_goal)


def select_n_relevant_agents(emotional_state: EmotionalAxesModel, n: int = 2) -> List[DynamicAgent]:
    # Score each agent based on the current emotional state.
    # For bipolar axes, if the state is positive, the positive agent gets the score;
    # if negative, the negative agent gets the absolute value.
    scores = {}
    scores[agent_joy.name]      = emotional_state.valence if emotional_state.valence > 0 else 0
    scores[agent_dysphoria.name]= abs(emotional_state.valence) if emotional_state.valence < 0 else 0
    scores[agent_love.name]     = emotional_state.affection if emotional_state.affection > 0 else 0
    scores[agent_hate.name]     = abs(emotional_state.affection) if emotional_state.affection < 0 else 0
    scores[agent_pride.name]    = emotional_state.pride if emotional_state.pride > 0 else 0
    scores[agent_shame.name]    = abs(emotional_state.pride) if emotional_state.pride < 0 else 0
    scores[agent_trust.name]    = emotional_state.trust if emotional_state.trust > 0 else 0
    scores[agent_mistrust.name] = abs(emotional_state.trust) if emotional_state.trust < 0 else 0
    scores[agent_disgust.name]  = emotional_state.disgust  # unipolar, 0 to 1
    scores[agent_anxiety.name]  = emotional_state.anxiety  # unipolar, 0 to 1

    # Add a small random noise to break any ties
    for key in scores:
        scores[key] += random.uniform(0, 0.01)

    # Sort agent names by score in descending order and select the top two.
    selected_names = sorted(scores, key=scores.get, reverse=True)[:n]

    # List of all available agent definitions
    available_agents = [
        agent_joy, agent_dysphoria,
        agent_love, agent_hate,
        agent_pride, agent_shame,
        agent_trust, agent_mistrust,
        agent_disgust, agent_anxiety
    ]
    # Select agent definitions whose names are in selected_names.
    selected_agents = [agent for agent in available_agents if agent.name in selected_names]
    return selected_agents


def execute_agent_group_emotion(ctx_msgs: str, last_message: str, emotional_state: EmotionalAxesModel) -> str:
    """### BEGIN CHARACTER CARD
{char_card_3rd_person_emotional}
### END CHARACTER CARD
"""
    context = f"""Alright, this is the current story and the character card.
### BEGIN CONVERSATION HISTORY
{ctx_msgs}
### END CONVERSATION HISTORY

### BEGIN LAST MESSAGE
{last_message}
### END LAST MESSAGE

How should Emmy react to the new message to make the story realistic? Please don't just give me example dialog, talk with me about WHY she should say / do something. I want this to be the best 
story ever!
"""

    agents = []
    selected_agents = select_n_relevant_agents(emotional_state, 2)
    for agent in selected_agents:
        prompt = emotion_agent_group_prefix + "\n" + agent.prompt
        agents.append(Agent(
            id=agent.name,
            name=agent.name,
            description=agent.description,
            goal=agent.goal,
            system_prompt=prompt
        ))

    state = {}
    result = agent_group_conversation(state, context, emotion_agent_group_goal, agents, min_confidence=0.8, max_rounds=8, round_robin=True, llmpreset=LlmPreset.Emotion)

    # add subagent messages
    subagent_messages = []
    for msg in result.agent_messages:
        subagent_messages.append(f"{msg.name}: {msg.text}")

    # get task focused summary
    recommendation = summarize_agent_conversation("", "\n".join(subagent_messages), emotion_agent_group_goal)
    return recommendation


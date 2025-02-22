from pm.agent_schemas.schema_main import get_agent_schema_main
from pm.agents_manager.agent import Agent
from pm.agents_manager.agent_group_conversation import agent_group_conversation
from pm.character import char_card_3rd_person_emotional
from pm.common_prompts.select_relevant_agents import select_relevant_agents
from pm.common_prompts.summarize_agent_conversation import summarize_agent_conversation
from pm.llm.base_llm import LlmPreset


def execute_agent_group_emotion(ctx_msgs: str, last_message: str) -> str:
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
FOCUS ON THE LAST MESSAGE AND MAKE IT SHORT! MAXIMUM 4 SENTENCES!
"""

    layer = get_agent_schema_main()
    agent_group = layer.agent_group_emotions
    agents = []
    while True:
        reso = select_relevant_agents(agent_group.agents, context)
        selected_agents = [a for a in agent_group.agents if a.name in reso]
        if len(selected_agents) == 2:
            break

    for agent in selected_agents:
        prompt = layer.common_prefix + "\n" + agent_group.prompt_prefix + "\n" + agent.prompt
        agents.append(Agent(
            id=agent.name,
            name=agent.name,
            description=agent.description,
            goal=agent.goal,
            system_prompt=prompt
        ))

    state = {}
    result = agent_group_conversation(state, context, agent_group.goal, agents, min_confidence=0.8, max_rounds=8, round_robin=True, llmpreset=LlmPreset.Emotion)

    # add subagent messages
    subagent_messages = []
    for msg in result.agent_messages:
        subagent_messages.append(f"{msg.name}: {msg.text}")

    # get task focused summary
    recommendation = summarize_agent_conversation(context, "\n".join(subagent_messages), agent_group.goal)
    return recommendation


import json
import logging
from enum import Enum, StrEnum
from typing import Tuple
from typing import TypedDict, List

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from pm.agents_manager.agent import Agent, AgentMessage
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

logger = logging.getLogger(__name__)

prompt_routing = """You are an agent in a cognitive architecture designed to assist individuals with mental health challenges by offering insights and perspectives modeled on diverse ways of thinking.
In your scenario the user '{user_name}' has autism and has troubles gauging the response of his friend '{companion_name}'. Your goal is to simulate the thoughts of {companion_name} so {user_name} can learn Theory-of-Mind of others before
actually talking to {companion_name}. 

You are a routing agent responsible for deciding who should speak or use a tool next in a GWT-inspired cognitive architecture.
Your task is to select the appropriate agent from the list of available sub-agents, based on their skills, expertise, and the current context.

Below is a list of sub-agents, each with a brief description of their area of expertise:

{agents_description}

Consider the context and the task requirements, then decide which agent should act next.
Your output must follow this exact valid JSON format:

{agent_decision_schema}
"""


prompt_boss = """You are an agent in a cognitive architecture designed to assist individuals with mental health challenges by offering insights and perspectives modeled on diverse ways of thinking.
In your scenario the user '{user_name}' has autism and has troubles gauging the response of his friend '{companion_name}'. Your goal is to simulate the thoughts of {companion_name} so {user_name} can learn Theory-of-Mind of others before
actually talking to {companion_name}. 

You are the boss agent in this GWT-inspired multi agent system cognitive architecture.
We have a context that ends with the most recent sensation. The goal of the agents is to decide what {companion_name} should do based on their perspective.
As in GWT, the different agents need to be managed and the most important (agent who represents their case the best) gets included in the final conclusion.
Also make sure that all agents have spoken!

When you are ready, report your conclusion to the system using the following tool invocation in valid JSON format: {conclusion_schema}

This is the current conversation context you should focus on:
### BEGINNING OF CONTEXT
{context_data}
### END OF CONTEXT

This is your task: {task}

These agents are part of the group chat: {agent_list}
"""

prompt_subagent = """
{sysprompt}

You are an agent within the "{description}" subsystem. Your role is to {goal}. 
**Strict Boundaries**: 
1. Focus exclusively on the task assigned to your subsystem. Avoid proposing actions or solutions that fall outside your defined role. 
2. If you detect a need for actions outside your expertise, clearly identify and defer to the appropriate subsystem. For example, if emotional reasoning is required, acknowledge it and allow the Emotions Subsystem to contribute.
3. Do not repeat or duplicate the functionality of other agents. Instead, contribute uniquely from your specialized perspective.

This is the current conversation context you should focus on:
### BEGINNING OF CONTEXT
{context_data}
### END OF CONTEXT

Your task is as follows:
{task}

**Key Requirements for Your Response**:
- Stay within your defined scope.
- Provide clear reasoning that aligns with your subsystem's expertise.
- Avoid ambiguity and ensure your response strictly aligns with your stated goals.

**Outcome of Overreach**:
If your response exceeds your defined scope or overlaps with another subsystem's role, it may be ignored or penalized. The boss agent will prioritize coherence and role-specific contributions over redundant or broad suggestions.

Be concise and precise in your response, and always ensure it aligns with your unique role in the system.
Your task is as follows:
{task}
"""


class BossAgentStatus(StrEnum):
    STILL_ARGUING = "STILL_ARGUING"
    CONCLUSION_POSSIBLE = "CONCLUSION_POSSIBLE"


class SubAgentState(TypedDict):
    next_agent: str
    confidence: float
    conclusion: str
    agent_idx: int
    finished: bool
    routing: str
    required_confidence: float
    routing_count: int
    convert_to_memory: bool
    product: str
    status: BossAgentStatus
    max_rounds: int
    cur_rounds: int
    llmpreset: LlmPreset
    context_data: str
    task: str

class BossAgentReport(BaseModel):
    """
    Report your currently recommended course of action.
    Your supervisor will decide if the answer is satisfactory.
    """

    task_completion: float = Field(
        ge=0, le=1,
        description="How far are in we in completing the task/goal assigned by the superior? Between 0 and 1."
    )

    task_deviation: float = Field(
        ge=0, le=1,
        description="How much are the agents deviating from the current task? Between 0 and 1. 0 is good, no deviation. 1 is bad, maximum deviation! You need to steer the agents back in the right "
                    "direction!"
    )

    agent_completeness: float = Field(
        ge=0, le=1,
        description="How many of the agents have had a chance to talk? Between 0 and 1."
    )

    agent_feedback: str = Field(
        description="Feedback for agents if they deviate too much from the assigned task."
    )


    def execute(self, state: SubAgentState):
        #logger.info(json.dumps(self.model_dump(), indent=2))
        state["confidence"] = 0
        state["conclusion"] = ""
        state["status"] = BossAgentStatus.STILL_ARGUING
        state["finished"] = False
        state["confidence"] = 0

        #if self.agent_completeness < 0.75:
        #    return {"error": f"Not enough agents have had the chance to speak! Feedback: {self.agent_feedback}"}
        #if self.task_completion > 0.85: # and self.task_deviation < 0.15:
        #    state["finished"] = True
        #    state["confidence"] = 1
        #    state["conclusion"] = ""
        #    #state["status"] = BossAgentStatus.CONCLUSION_POSSIBLE
        #    return {}
        #else:
        #    return {"error": f"Confidence not high enough or too much deviation. Feedback: {self.agent_feedback}"}


def compile_message(agents: List[Agent], cur_agent: Agent) -> List[Tuple[str, str]]:
    # Collect all messages with their roles and agent names
    all_messages = []

    for agent in agents:
        role = "assistant" if agent == cur_agent else "user"

        for message in agent.messages:
            role = "user" if message.from_tool else role

            # treat boss as system
            if message.name == "Boss Agent":
                role = "system"

            # Append each message with role, text, timestamp, and agent name if it's a user
            all_messages.append({
                "role": role,
                "text": message.text if role == "assistant" else f"{agent.name}: {message.text}",
                "created_at": message.created_at,
                "is_private": False,  # not message.public if role == "user" else False
            })

    all_messages.sort(key=lambda x: x["created_at"])
    return [(x["role"], x["text"]) for x in all_messages]


# Dynamically create Enum of agent names
def create_agent_enum(agents, state):
    return Enum(
        "AgentName",
        {agent.name: agent.name for agent in agents if agent.name != state["next_agent"]}
    )


# Dynamically create RouteDecision class using Enum for agent names
def create_route_decision_class(agent_enum):
    class RouteDecision(BaseModel):
        agent_name: agent_enum  # Use Enum instead of Litera

        # Override dict to handle enum as string
        def model_dump(self, **kwargs):
            base_dict = super().model_dump(**kwargs)
            base_dict["agent_name"] = str(base_dict["agent_name"])  # Convert enum to string
            return base_dict

        def execute(self, state: SubAgentState):
            #logger.info(json.dumps(self.model_dump(), indent=2))
            state["next_agent"] = self.agent_name.value

    return RouteDecision


def _execute_router(state: SubAgentState, agents: List[Agent]):
    """
    Iterate over agents in list and start again if the boss didn't finish already.
    """
    state["routing_count"] += 1

    if state["finished"]:
        return state

    # start with boss agent or every 6 routes to make sure we don't run endlessly
    if state["next_agent"] == "initial" or state["routing_count"] % len(agents) == 0:
        state["next_agent"] = agents[0].name
        state["agent_idx"] = 1
        return state
    state["cur_rounds"] = state["cur_rounds"] + 1

    if state["routing_count"] > state["max_rounds"]:
        state["next_agent"] = END
        return state

    if state["routing"] == "round_robin":
        if state["agent_idx"] >= len(agents):
            state["agent_idx"] = 0
        state["next_agent"] = agents[state["agent_idx"]].name
        state["agent_idx"] += 1
    elif state["routing"] == "agentic":
        AgentNameEnum = create_agent_enum(agents, state)
        RouteDecision = create_route_decision_class(AgentNameEnum)

        # not all messages
        messages = compile_message(agents, None)[-2:]
        extra = {
            "agents_description": "\n".join([f"{x.name}: {x.description}" for x in agents if x.name != state["next_agent"]]),
            "agent_decision_schema": json.dumps(RouteDecision.model_json_schema())
        }
        messages.insert(0, ("system", controller.format_str(prompt_routing, extra=extra)))

        # handle tools
        while True:
            _, calls = controller.completion_tool(state["llmpreset"], messages,
                                                  comp_settings=CommonCompSettings(max_tokens=1024),
                                                  tools=[RouteDecision])
            if len(calls) > 0:
                calls[0].execute(state)
                break

    return state


def _execute_agent(state: SubAgentState, agents: List[Agent]):
    """
    Dynamically execute the current agent. They all follow the same pattern.
    """
    cur_agent = state["next_agent"]
    agent: Agent = next(filter(lambda x: x.name == cur_agent, agents), None)
    if not agent:
        return state

    if agent.name == cur_agent:
        if cur_agent == "Boss Agent" and len(agent.messages) == 0:
            # initial message
            agent.messages.append(AgentMessage(
                text=f"""This is the current conversation context you should focus on:
### BEGINNING OF CONTEXT
{state['context_data']}
### END OF CONTEXT

Your task is as follows:
{state['task']}
Get to work!""",
                name=agent.name,
                public=True
            ))
        else:
            # loop in case the tools don't work on the first tries
            while True:
                messages = compile_message(agents, agent)
                messages.insert(0, ("system", agent.system_prompt))

                comp_settings = agent.comp_settings if agent.comp_settings else CommonCompSettings(max_tokens=1024)

                tool_res_list = []
                content, calls = controller.completion_tool(state["llmpreset"], messages, comp_settings=comp_settings,
                                                            tools=agent.tools)
                for call in calls:
                    tool_res_list.append(call.execute(state))

                    # boss agent has confidence in answer -> break out
                    if state["finished"]:
                        return state

                # run until something called, in case the tool call got mangled
                if len(agent.tools) > 0 and len(tool_res_list) == 0:
                    continue

                # add text responses, make tool call private
                if agent.name != "Boss Agent":
                    agent.messages.append(AgentMessage(
                        text="\n".join(x.model_dump_json() for x in calls) if len(content) == 0 else content,
                        name=agent.name,
                        public=len(tool_res_list) == 0
                    ))

                    # result of tool call will always be public
                    if len(tool_res_list) > 0:
                        call_string = "\n".join([json.dumps(x) for x in tool_res_list])
                        agent.messages.append(AgentMessage(
                            text=call_string,
                            name=agent.name,
                            public=True,
                            from_tool=True
                        ))

                break

    return state


def get_next_agent(state: SubAgentState):
    if state["finished"]:
        return END
    return state["next_agent"]


class BossWorkerChatResult(BaseModel):
    confidence: float
    conclusion: str
    agent_messages: List[AgentMessage]
    as_internal_thought: str
    status: BossAgentStatus


def agent_group_conversation(base_state, context_data: str, task: str, agents: List[Agent], min_confidence: float = 0.5,
                             convert_to_memory: bool = False, round_robin: bool = False, max_rounds: int = 32, llmpreset: LlmPreset = LlmPreset.Default) -> BossWorkerChatResult:
    workflow = StateGraph(SubAgentState)
    workflow.add_node("router", lambda x: _execute_router(x, agents))
    workflow.add_edge(START, "router")

    ## add boss
    #agents.insert(0,
    #              Agent(
    #                  id="Boss Agent",
    #                  name="Boss Agent",
    #                  description="",
    #                  goal="",
    #                  system_prompt=controller.format_str(prompt_boss,
    #                                                      extra={"conclusion_schema": json.dumps(
    #                                                          BossAgentReport.model_json_schema())}),
    #                  tools=[BossAgentReport],
    #                  task=task
    #              )
    #              )

    agent_list = ", ".join([x.name for x in agents])

    # format prompts
    for agent in agents:
        extra = {
            "context_data": context_data,
            "task": task,
            "description": agent.description,
            "goal": agent.goal,
            "agent_list": agent_list,
            "sysprompt": agent.system_prompt
        }

        if agent.id == "Boss Agent":
            base_prompt = prompt_boss
        else:
            base_prompt = prompt_subagent

        agent.system_prompt = controller.format_str(base_prompt, extra=extra)
        workflow.add_node(agent.name, lambda x: _execute_agent(x, agents))
        workflow.add_edge(agent.name, "router")

    workflow.add_conditional_edges("router", get_next_agent)
    graph = workflow.compile()

    state = SubAgentState(
        confidence=0,
        next_agent="initial",
        finished=False,
        routing="round_robin" if round_robin else "agentic",
        required_confidence=min_confidence,
        routing_count=0,
        convert_to_memory=convert_to_memory,
        product="",
        max_rounds=max_rounds,
        cur_rounds=0,
        llmpreset=llmpreset,
        context_data=context_data,
        task=task,
        conclusion="",
        status=""
    )
    state = graph.invoke(state, {"recursion_limit": 1024})

    msgs = []
    for agent in agents:
        msgs += [x for x in agent.messages if x.public]
    msgs.sort(key=lambda x: x.created_at)

    res = BossWorkerChatResult(
        confidence=state["confidence"],
        conclusion=state["conclusion"],
        agent_messages=msgs,
        as_internal_thought="",
        status=BossAgentStatus.CONCLUSION_POSSIBLE,
    )
    return res

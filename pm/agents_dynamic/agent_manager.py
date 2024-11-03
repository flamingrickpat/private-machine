import json
import logging
from enum import Enum
from typing import Literal, TypedDict, List
from typing import Tuple

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from pm.agents_dynamic.agent import Agent, AgentMessage
from pm.agents_dynamic.boss_agent import prompt_boss
from pm.agents_dynamic.dynamic_subagent import prompt_subagent
from pm.agents_dynamic.routing_agent import prompt_routing
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.tools.generate_documentation import create_header_functions, create_documentation_functions, \
    create_documentation_json

logger = logging.getLogger(__name__)

class SubAgentState(TypedDict):
    next_agent: str
    confidence: float
    conclusion: str
    agent_idx: int
    finished: bool
    routing: str
    required_confidence: float
    routing_count: int

class BossAgentChoice(BaseModel):
    """
    Report your currently recommended course of action.
    Your supervisor will decide if the answer is satisfactory.
    """
    conclusion: str = Field(description="your recommended course of action")
    confidence: float = Field(ge=0, le=1, description="how sure you are with your conclusion (from 0 to 1)")

    def execute(self, state: SubAgentState):
        logger.info(json.dumps(self.model_dump(), indent=2))
        state["confidence"] = self.confidence
        state["conclusion"] = self.conclusion

        if self.confidence > state["required_confidence"]:
            state["finished"] = True
            return {}
        else:
            return {"error": "Confidence not high enough, research more!"}

def compile_message(agents: List[Agent], cur_agent: Agent) -> List[Tuple[str, str]]:
    # Collect all messages with their roles and agent names
    all_messages = []

    for agent in agents:
        role = "assistant" if agent == cur_agent else "user"
        for message in agent.messages:
            # Append each message with role, text, timestamp, and agent name if it's a user
            all_messages.append({
                "role": "user" if message.from_tool else role,
                "text": message.text if role == "assistant" else f"{agent.name}: {message.text}",
                "created_at": message.created_at,
                "is_private": False,  # not message.public if role == "user" else False
            })

    # Sort messages by timestamp
    all_messages.sort(key=lambda x: x["created_at"])

    # Compile messages by detecting role changes
    compiled_messages = []
    role_message_buffer = []
    current_role = None

    for msg in all_messages:
        msg_role = msg["role"]

        # Detect role change or buffer flush condition
        if msg_role != current_role:
            if role_message_buffer:
                # Append the current buffer to compiled messages
                compiled_messages.append((current_role, "\n".join(role_message_buffer)))
                role_message_buffer = []

            current_role = msg_role  # Update the current role

        # Only add public messages to the buffer
        if not msg["is_private"]:
            role_message_buffer.append(msg["text"])

    # Flush any remaining messages in the buffer
    if role_message_buffer:
        compiled_messages.append((current_role, "\n".join(role_message_buffer)))

    return compiled_messages

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
            logger.info(json.dumps(self.model_dump(), indent=2))
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
    if state["next_agent"] == "initial" or state["routing_count"] % 6 == 0:
        state["next_agent"] = agents[0].name
        state["agent_idx"] = 1
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
        messages = compile_message(agents, None)[-6:]
        extra = {
            "agents_description": "\n".join([f"{x.name}: {x.description}" for x in agents]),
            "agent_decision_schema": json.dumps(RouteDecision.model_json_schema())
        }
        messages.insert(0, ("system", controller.format_str(prompt_routing, extra=extra)))

        # handle tools
        while True:
            _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024), tools=[RouteDecision])
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
                text=f"Alright, our task is '{agent.task}'. Get to work!",
                public=True
            ))
        else:
            # loop in case the tools don't work on the first tries
            while True:
                messages = compile_message(agents, agent)
                messages.insert(0, ("system", agent.system_prompt))

                comp_settings = agent.comp_settings if agent.comp_settings else CommonCompSettings(max_tokens=1024)

                tool_res_list = []
                content, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=comp_settings, tools=agent.tools)
                for call in calls:
                    tool_res_list.append(call.execute(state))

                    # boss agent has confidence in answer -> break out
                    if state["finished"]:
                        return state

                # run until something called, in case the tool call got mangled
                if len(agent.tools) > 0 and len(tool_res_list) == 0:
                    continue

                # add text responses, make tool call private
                agent.messages.append(AgentMessage(
                    text=content,
                    public=len(tool_res_list) == 0
                ))

                # result of tool call will always be public
                if len(tool_res_list) > 0:
                    call_string = "\n".join([json.dumps(x) for x in tool_res_list])
                    agent.messages.append(AgentMessage(
                        text=call_string,
                        public=True,
                        from_tool=True
                    ))

                break

    return state


def get_next_agent(state: SubAgentState):
    if state["finished"]:
        return END
    return state["next_agent"]


def execute_boss_worker_chat(context_data: str, task: str, agents: List[Agent], min_confidence: float = 0.5) -> (str, float):
    workflow = StateGraph(SubAgentState)
    workflow.add_node("router", lambda x: _execute_router(x, agents))
    workflow.add_edge(START, "router")

    # add boss
    agents.insert(0,
        Agent(
            name="Boss Agent",
            description="",
            goal="",
            system_prompt=controller.format_str(prompt_boss,
                                                extra={"conclusion_schema": json.dumps(BossAgentChoice.model_json_schema())}),
            tools=[BossAgentChoice],
            task=task
        )
    )

    agent_list = ", ".join([x.name for x in agents])

    # format prompts
    for agent in agents:
        if agent.system_prompt is None:
            agent.system_prompt = prompt_subagent

        extra = {
            "context_data": context_data,
            "task": task,
            "description": agent.description,
            "goal": agent.goal,
            "agent_list": agent_list
        }
        extra.update(create_header_functions(agent.functions))
        extra.update(create_documentation_functions(agent.functions))
        extra.update(create_documentation_json(agent.tools))

        agent.system_prompt = controller.format_str(agent.system_prompt, extra=extra)
        workflow.add_node(agent.name, lambda x: _execute_agent(x, agents))
        workflow.add_edge(agent.name, "router")

    workflow.add_conditional_edges("router", get_next_agent)
    graph = workflow.compile()

    state = SubAgentState(
        confidence=0,
        next_agent="initial",
        finished=False,
        routing="agentic",
        required_confidence=min_confidence,
        routing_count=0
    )
    state = graph.invoke(state, {"recursion_limit": 100})
    return state

import random
from datetime import datetime
from typing import List, Type, Tuple, Callable
import json
import logging
from typing import Literal, TypedDict, Dict, Any, List

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from pm.agents_dynamic.boss_agent import prompt_boss
from pm.agents_dynamic.routing_agent import prompt_routing
from pm.controller import controller
from pm.llm.tools_parser_local import PydanticToolsParserLocal
from pm.agents_dynamic.agent import Agent, AgentMessage
from pm.agents_dynamic.dynamic_subagent import prompt_subagent
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

def _execute_router(state: SubAgentState, agents: List[Agent]):
    """
    Iterate over agents in list and start again if the boss didn't finish already.
    """
    if state["finished"]:
        return state

    if state["next_agent"] == "initial":
        state["next_agent"] = agents[0].name
        state["agent_idx"] = 1
        return state

    if state["routing"] == "round_robin":
        if state["agent_idx"] >= len(agents):
            state["agent_idx"] = 0
        state["next_agent"] = agents[state["agent_idx"]].name
        state["agent_idx"] += 1
    elif state["routing"] == "agentic":
        # create class dynamically
        class RouteDecision(BaseModel):
            agent_name: Literal[tuple(agent.name for agent in agents if agent.name != state["next_agent"])]

            def execute(self, state: SubAgentState):
                logger.info(json.dumps(self.model_dump(), indent=2))
                state["next_agent"] = self.agent_name

        # not all messages
        messages = compile_message(agents, None)[-6:]
        extra = {
            "agents_description": "\n".join([f"{x.name}: {x.description}" for x in agents]),
            "agent_decision_schema": json.dumps(RouteDecision.model_json_schema())
        }
        messages.insert(0, ("system", controller.format_str(prompt_routing, extra=extra)))

        # bind tools and function
        llm = controller.llm
        llm_lc = llm.get_langchain_model().bind_tools([RouteDecision])
        ai_msg = llm_lc.invoke(messages)

        # handle tools
        calls = PydanticToolsParserLocal(tools=[RouteDecision]).invoke(ai_msg)
        calls[0].execute(state)

    return state


def _execute_agent(state: SubAgentState, agents: List[Agent]):
    """
    Dynamically execute the current agent. They all follow the same pattern.
    """
    cur_agent = state["next_agent"]
    agent = next(filter(lambda x: x.name == cur_agent, agents), None)
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

                # bind tools and function
                llm = controller.llm
                llm_lc = llm.get_langchain_model().bind_tools(agent.tools)
                ai_msg = llm_lc.invoke(messages)

                # handle tools
                tool_res_list = []  # contains the result dicts or lists
                calls = PydanticToolsParserLocal(tools=agent.tools).invoke(ai_msg)
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
                    text=ai_msg.content,
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
        if agent.system_prompt == "":
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
        required_confidence=min_confidence
    )
    state = graph.invoke(state, {"recursion_limit": 100})
    return state

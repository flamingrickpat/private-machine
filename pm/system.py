import json
import uuid
from typing import Literal, TypedDict, Dict, Any, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from jinja2 import Template
from sympy.physics.units import temperature

from database.db_model import Message

from pm.llm.llamacpp import LlamaCppLlm, LlmModel, CommonCompSettings
from pm.config.config import read_config_file, MainConfig

llm = LlamaCppLlm()
cfg = read_config_file("config.json")
llm.set_model(cfg.model)

class AgentState(TypedDict):
    next_agent: str
    output: str
    title: str
    conversation_id: str
    messages: List[Dict[str, str]]

SYSTEM_PROMPT = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>"""

def render_chat(messages):
    template_str = """{{ system_prompt }}{% for message in messages -%}
<{{ '<|im_start|>user' if message['role'] == 'user' else '<|im_start|>assistant' }}>{{ message['content'] }}</{{ '<|im_end|>' }}>
{%- endfor %}"""

    template = Template(template_str)
    return template.render(messages=messages, system_prompt=SYSTEM_PROMPT)

def main_agent(state: AgentState):
    prompt = render_chat(state["messages"])
    print(prompt)
    setting = CommonCompSettings(
        max_tokens=512,
        temperature=0.9
    )
    res = llm.completion(prompt=prompt, comp_settings=setting)
    state['output'] = res.output_sanitized
    return state

workflow = StateGraph(AgentState)
workflow.add_node("main_agent", main_agent)

def has_title_conditional_edge(state):
    if state["title"] and state["title"] != "":
        return "pre_check_agent"
    else:
        return "title_agent"

workflow.add_edge(START, "main_agent")
workflow.add_edge("main_agent", END)

graph = workflow.compile()

def make_completion(state: AgentState = None) -> AgentState:
    state = graph.invoke(state)
    return state

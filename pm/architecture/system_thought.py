import json
import logging
import uuid
import random
from enum import Enum
from typing import TypedDict, Dict, List

from IPython.core.interactiveshell import is_integer_string
from accelerate.commands.config.config import description
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from pm.agents.generate_thought_aspects import generate_personality_aspects
from pm.agents.generate_thoughts import generate_thoughts
from pm.agents.validate_thought import validate_thought
from pm.agents_dynamic.schema_thought_contemplation import thought_contemplation_dialog
from pm.architecture.state import AgentState
from pm.consts import THOUGHT_VALIDNESS_MIN, LLM_PRESET_FINAL_OUTPUT
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, rank_table, fetch_relations, fetch_messages_as_string, get_facts_str, insert_object, get_facts
from pm.database.db_model import Message, MessageSummary, MessageInterlocus
from pm.database.load_messages import build_prompt
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.prompts.prompt_main_agent import build_sys_prompt_conscious_assistant
from pm.utils.string_utils import get_last_n_messages_or_words_from_string
from pm.utils.token_utils import quick_estimate_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)


class MetaToolset(str, Enum):
    Default = "default"


class MetaModeDefaultModeNetwork(BaseModel):
    """
    Use this tool when you want to consciously zone-out and daydream.
    """
    thinking_goal: str = Field(description="What you want to zone-out for. Could be any fantasy scenario or such.")


class MetaModeDoingMode(BaseModel):
    """
    Use this tool to get access to advanced tools for interacting with the world such as webbrowser or clock.
    """
    toolset: MetaToolset = Field(description="What toolset you want to use.")


class MetaModeBeingMode(BaseModel):
    """
    Use this tool to disable thinking and instead wait for sensations such as system notifications or user input.
    """
    timelimit_minutes: int = Field(description="How long you want to wait for sensation before being able to select metacognitive state again in minutes.")


class MetaModeManualThinking(BaseModel):
    """
    Use this mode to ponder over a thought with your best backend models.
    """
    contemplation: str = Field(description="Overall concept you want to contemplate.")


class MetaModeInitConversation(BaseModel):
    """
    Use this to initialize a conversation with the user.
    Warning: This will disable further thinking, and you'll have to wait until the user replies.
    """
    message: str = Field(description="Message to send to the user.")

def agent_init_auto_thought(state: AgentState):
    sysmsg = "The user hasn't responded yet, entering autonomous thinking mode..."
    msg_init = Message(
        conversation_id=state.conversation_id,
        role='system',
        text=sysmsg,
        public=True,
        embedding=controller.embedder.get_embedding_scalar_float_list(sysmsg),
        tokens=quick_estimate_tokens(sysmsg),
        interlocus=MessageInterlocus.MessageSystemInst
    )
    insert_object(msg_init)
    return state

def agent_auto_thought_prepare_knowledge(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    conv_context = get_last_n_messages_or_words_from_string(full_text, n_messages=8)

    data = set()
    facts = get_facts(state.conversation_id, conv_context, 4)
    for f in facts:
        data.add(f.text)

    random.shuffle(list(data))
    state.knowledge_implicit_facts = truncate_to_tokens("\n".join(data), 1024)
    return state


def agent_metacognitive_selection(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id, n_thought=8)
    query = get_last_n_messages_or_words_from_string(full_text, n_messages=4)
    msgs = rank_table(state.conversation_id, query, Message)
    sums = rank_table(state.conversation_id, query, MessageSummary)
    rels = fetch_relations("main")

    tools = [MetaModeDefaultModeNetwork, MetaModeInitConversation]  # MetaModeDoingMode, MetaModeBeingMode, MetaModeManualThinking]

    sysprompt = build_sys_prompt_conscious_assistant(False, tools, state.knowledge_implicit_facts)
    messages = build_prompt(False, msgs, sums, rels, full_token_allowance=controller.config.get_model(LLM_PRESET_FINAL_OUTPUT).context_size - (quick_estimate_tokens(sysprompt) + 256))

    # add system prompt
    messages.insert(0, (
        "system",
        sysprompt
    ))

    _, calls = controller.completion_tool(LLM_PRESET_FINAL_OUTPUT, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.6), tools=tools)
    call = calls[0]
    if isinstance(call, MetaModeDefaultModeNetwork):
        state.next_agent = "agent_generate_int_thoughts"
        state.goal_thought = call.thinking_goal
    elif isinstance(call, MetaModeInitConversation):
        state.next_agent = "agent_init_message"
        state.init_message = call.message

    return state

def agent_generate_int_thoughts(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    conv_context = get_last_n_messages_or_words_from_string(full_text, n_messages=8)

    data = set()
    facts = get_facts(state.conversation_id, conv_context, 4)
    for f in facts:
        data.add(f.text)

    ai_memory = "\n".join(get_facts_str(state.conversation_id, conv_context))
    random.shuffle(list(data))
    user_memory = truncate_to_tokens("\n".join(data), 1024)

    while True:
        thought = generate_thoughts(f"Generate a thought for {controller.config.companion_name}! Right now they want to think about {state.goal_thought}.",
                                    controller.config.character_card_story,
                                    conv_context,
                                    ai_memory,
                                    user_memory)
        validness = validate_thought(thought)
        if validness > THOUGHT_VALIDNESS_MIN:
            state.thought = thought
            break

    msg_thought = Message(
        conversation_id=state.conversation_id,
        role='assistant',
        text=thought,
        public=False,
        embedding=controller.embedder.get_embedding_scalar_float_list(thought),
        tokens=quick_estimate_tokens(thought),
        interlocus=MessageInterlocus.MessageThought
    )
    insert_object(msg_thought)

    logger.info(f"Thought: {thought}")
    return state


def agent_generate_aspects(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    conv_context = get_last_n_messages_or_words_from_string(full_text)

    aspects = generate_personality_aspects(controller.config.character_card_story, state.thought)
    res = thought_contemplation_dialog(conv_context, f"Reach a conclusion on this thought: {state.thought}", aspects)

    thought = res.as_internal_thought
    msg_thought = Message(
        conversation_id=state.conversation_id,
        role='assistant',
        text=thought,
        public=False,
        embedding=controller.embedder.get_embedding_scalar_float_list(thought),
        tokens=quick_estimate_tokens(thought),
        interlocus=MessageInterlocus.MessageThought
    )
    insert_object(msg_thought)
    return state


def agent_init_message(state: AgentState):
    output = state.init_message
    msg_ai = Message(
        conversation_id=state.conversation_id,
        role='assistant',
        text=output,
        public=True,
        embedding=controller.embedder.get_embedding_scalar_float_list(output),
        tokens=quick_estimate_tokens(output),
        interlocus=MessageInterlocus.MessageResponse
    )
    insert_object(msg_ai)
    return state

def system_internal_thought():
    pass
    # get random thought from context
    # let conscious ai with conscious history decide if its a thought worth thinking
    # if yes, enter story mode
    # instruct sotry writer agent with 2 aspects
    # end generate for n steps or until breaking out


def add_thought_system_to_graph(workflow: StateGraph, start_node: str, end_node: str):
    workflow.add_node("agent_init_auto_thought", agent_init_auto_thought)
    workflow.add_node("agent_auto_thought_prepare_knowledge", agent_auto_thought_prepare_knowledge)
    workflow.add_node("agent_metacognitive_selection", agent_metacognitive_selection)
    workflow.add_node("agent_generate_int_thoughts", agent_generate_int_thoughts)
    workflow.add_node("agent_generate_aspects", agent_generate_aspects)
    workflow.add_node("agent_init_message", agent_init_message)

    workflow.add_edge(start_node, "agent_init_auto_thought")
    workflow.add_edge("agent_init_auto_thought", "agent_auto_thought_prepare_knowledge")
    workflow.add_edge("agent_auto_thought_prepare_knowledge", "agent_metacognitive_selection")
    workflow.add_conditional_edges("agent_metacognitive_selection", lambda x: x.next_agent)
    workflow.add_edge("agent_generate_int_thoughts", "agent_generate_aspects")
    workflow.add_edge("agent_generate_aspects", end_node)

    workflow.add_edge("agent_init_message", end_node)
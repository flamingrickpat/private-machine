import json
import logging
import uuid
from typing import TypedDict, Dict, List

from langgraph.graph import StateGraph

from pm.agents.assistant_story_check import assistant_story_check
from pm.agents.completion_mode import determine_completion_mode, CompletionModeResponseMode
from pm.agents.determine_complexity import determine_complexity
from pm.agents.generate_thought_aspects import generate_personality_aspects
from pm.agents.generate_thoughts import generate_thoughts
from pm.agents.rewrite_as_thought import rewrite_as_thought
from pm.agents.tool_selection import determine_tools
from pm.agents.validate_response import validate_response
from pm.agents.validate_thought import validate_thought
from pm.agents_dynamic.schema_subconscious import get_plan_from_subconscious_agents
from pm.agents_dynamic.schema_thought_contemplation import thought_contemplation_dialog
from pm.architecture.state import AgentState
from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.consts import COMPLEX_THRESHOLD, RECALC_SUMMARIES_MESSAGES, THOUGHT_VALIDNESS_MIN, RESPONSE_VALIDNESS_MIN
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, rank_table, fetch_relations, fetch_messages_as_string, get_facts, insert_object
from pm.database.db_model import Message, MessageSummary, MessageInterlocus
from pm.database.load_messages import build_prompt
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.prompts.prompt_main_agent import build_sys_prompt_conscious_assistant
from pm.prompts.prompt_main_agent_story import build_sys_prompt_conscious_story
from pm.utils.string_utils import get_last_n_messages_or_words_from_string
from pm.utils.token_utils import quick_estimate_tokens

logger = logging.getLogger(__name__)

def agent_generate_int_thoughts(state: AgentState):
    sysmsg = "The user hasn't responded yet, entering autonomous thinking mode..."
    msg_init = Message(
        conversation_id=state["conversation_id"],
        role='system',
        text=sysmsg,
        public=True,
        embedding=controller.embedder.get_embedding_scalar_float_list(sysmsg),
        tokens=quick_estimate_tokens(sysmsg),
        interlocus=MessageInterlocus.AutoThoughtSystemInst
    )
    insert_object(msg_init)

    user_memory = "No memory yet!"
    full_text = fetch_messages_as_string(state["conversation_id"])
    conv_context = get_last_n_messages_or_words_from_string(full_text)
    ai_memory = "\n".join(get_facts(state["conversation_id"], conv_context))
    thought = generate_thoughts(f"Generate a thought for {controller.config.companion_name}!", controller.config.character_card_story, conv_context, ai_memory, user_memory)
    state["thought"] = thought

    logger.info(f"Thought: {thought}")
    return state


def agent_generate_aspects(state: AgentState):
    full_text = fetch_messages_as_string(state["conversation_id"])
    conv_context = get_last_n_messages_or_words_from_string(full_text)

    aspects = generate_personality_aspects(controller.config.character_card_story, state["thought"])
    res = thought_contemplation_dialog(conv_context, f"Reach a conclusion on this thought: {state['thought']}", aspects)

    thought = res.as_internal_thought
    msg_thought = Message(
        conversation_id=state["conversation_id"],
        role='assistant',
        text=thought,
        public=False,
        embedding=controller.embedder.get_embedding_scalar_float_list(thought),
        tokens=quick_estimate_tokens(thought),
        interlocus=MessageInterlocus.MessageThought
    )
    insert_object(msg_thought)

    return state


def add_thought_system_to_graph(workflow: StateGraph, start_node: str, end_node: str):
    workflow.add_node("agent_generate_int_thoughts", agent_generate_int_thoughts)
    workflow.add_node("agent_generate_aspects", agent_generate_aspects)

    workflow.add_edge(start_node, "agent_generate_int_thoughts")
    workflow.add_edge("agent_generate_int_thoughts", "agent_generate_aspects")
    workflow.add_edge("agent_generate_aspects", end_node)
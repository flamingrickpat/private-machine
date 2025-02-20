import copy
import sys
import os
import traceback
import uuid
from pprint import pprint

from pydantic import BaseModel

from pm.actions import generate_action_selection, action_tool_call, action_reply, action_init_user_conversation, action_internal_contemplation
from pm.agents_gwt.gwt_layer import pregwt_to_internal_thought
from pm.agents_gwt.schema_main import GwtLayer
from pm.common_prompts.rate_complexity_v2 import rate_complexity_v2
from pm.common_prompts.rewrite_as_thought import rewrite_as_thought
from pm.cosine_clustering.cosine_cluster import get_best_category, cluster_text, TEMPLATE_CATEGORIES
from pm.fact_learners.extract_cause_effect_agent import extract_cas
from pm.sensation_evaluation import generate_sensation_evaluation
from pm.subsystems.subsystem_base import CogEvent
from pm.system_classes import Sensation, SensationType, SystemState
from pm.validation.validate_directness import validate_directness
from pm.validation.validate_query_fulfillment import validate_query_fulfillment
from pm.validation.validate_response_in_context import validate_response_in_context

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import time
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Optional, Any, List, Tuple
import re

from sqlmodel import Field, Session, SQLModel, create_engine, select, col
from sqlalchemy import Column, text, INTEGER, func

from pm.cluster.cluster_temporal import MiniContextCluster, temporally_cluster_context, TemporalCluster, temp_cluster_to_level
from pm.cluster.cluster_utils import get_optimal_clusters
from pm.common_prompts.extract_facts import extract_facts
from pm.common_prompts.rate_complexity import rate_complexity
from pm.common_prompts.summary_perspective import summarize_messages
from pm.controller import controller, log_conversation
from pm.embedding.embedding import get_embedding
from pm.embedding.token import get_token
from pm.database.tables import Event, EventCluster, Cluster, ClusterType, PromptItem, Fact, InterlocusType, AgentMessages, CauseEffectDbEntry
from pm.prompt.get_optimized_prompt import get_optimized_prompt, create_optimized_prompt, get_optimized_prompt_temp_cluster
from pm.thought.generate_tot import generate_tot_v1

from pm.character import sysprompt, database_uri, cluster_split, companion_name, timestamp_format, sysprompt_addendum, char_card_3rd_person_neutral
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.common_prompts.get_emotional_state import get_emotional_state
from pm.meta_learning.integrate_rules_final_output import integrate_rules_final_output
from pm.system_utils import *


def sensation_to_event(sen: Sensation) -> Event:
    content = sen.payload
    event = None
    if sen.sensation_type == SensationType.UserInput:
        event = Event(
            source=f"{user_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Public.value
        )
        add_cognitive_event(event)
    elif sen.sensation_type == SensationType.ToolResult:
        event = Event(
            source=f"system",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.SystemMessage.value
        )
        add_cognitive_event(event)
    return event

def run_tick(inp: str) -> str:
    #system_state = system_state_load()
    #system_state.decay(0.05)

    allowed_actions = [ActionType.Reply, ActionType.ToolCall]
    evaluate_sensation = True

    if inp.strip() == "/think":
        payload = generate_thought()
        sen = Sensation(
            sensation_type=SensationType.UserInput,
            payload=payload
        )
    elif inp.strip() == "/sleep":
        check_tasks()
        return "clustered and extracted facts!"
    elif inp.startswith("<tool>"):
        payload = inp.replace("<tool>", "")
        sen = Sensation(
            sensation_type=SensationType.ToolResult,
            payload=payload
        )
    else:
        payload = inp
        sen = Sensation(
            sensation_type=SensationType.UserInput,
            payload=payload
        )

    sensation_event = sensation_to_event(sen)
    #controller.system_state.cache_sensation = sensation_event
    add_cognitive_event(sensation_event)

    sensation_evaluation = generate_sensation_evaluation()
    #self.cache_sensation_evaluation = sensation_evaluation

    action_selection, action_event = generate_action_selection(allowed_actions)
    #self.cache_action_selection = action_event

    if action_selection == ActionType.Ignore:
        pass
    if action_selection == ActionType.ToolCall:
        tool_result = action_tool_call()
        run_tick(tool_result)
    elif action_selection == ActionType.Reply:
        action_reply()
    elif action_selection == ActionType.InitiateIdleMode:
        pass
    elif action_selection == ActionType.InitiateUserConversation:
        action_init_user_conversation()
    elif action_selection == ActionType.InitiateInternalContemplation:
        action_internal_contemplation()

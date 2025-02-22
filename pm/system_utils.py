import copy
import sys
import os
import traceback
import uuid

from django.contrib.auth import user_logged_in
from pydantic import BaseModel

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
from pm.character import sysprompt, database_uri, cluster_split, companion_name, timestamp_format, sysprompt_addendum, char_card_3rd_person_neutral, user_name, char_card_3rd_person_emotional
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.common_prompts.get_emotional_state import get_emotional_state
from pm.meta_learning.integrate_rules_final_output import integrate_rules_final_output
from pm.agents_gwt.gwt_layer import agent_discussion_to_internal_thought
from pm.agents_gwt.schema_main import GwtLayer
from pm.common_prompts.rewrite_as_thought import rewrite_as_thought
from pm.cosine_clustering.cosine_cluster import get_best_category, cluster_text, TEMPLATE_CATEGORIES
from pm.fact_learners.extract_cause_effect_agent import extract_cas
from pm.utils.enum_utils import make_enum_subset
from pm.subsystems_sneed.subsystem_thought import execute_subsystem_thought
from pm.tools.common import tool_docs
from pm.validation.validate_directness import validate_directness
from pm.validation.validate_query_fulfillment import validate_query_fulfillment
from pm.validation.validate_response_in_context import validate_response_in_context


res_tag = "<<<RESULT>>>"
MAX_GENERATION_TOKENS = 4096
MAX_THOUGHTS_IN_PROMPT = 16

def init():
    pass


def get_sensation(allow_timeout: bool = False, override: str = None):
    return controller.get_user_input(override=override)



class OperationMode(StrEnum):
    UserConversation = "UserConversation"
    InternalContemplation = "InternalContemplation"
    #VirtualWorld = "VirtualWorld"


def get_independent_thought_system_msg():
    return ("system", f"System Message: The user hasn't responded yet, entering autonomous thinking mode. Current timestamp: {datetime.now().strftime(timestamp_format)}\n"
                      f"Please use the provided tools to select an action type.")


def add_cognitive_event(event: Event):
    if event is None:
        return

    session = controller.get_session()
    session.add(event)
    session.flush()
    session.refresh(event)

def get_public_event_count() -> int:
    session = controller.get_session()
    events = list(session.exec(select(Event).where(Event.interlocus == 1).order_by(col(Event.id))).fetchall())
    return len(events)


def generate_tot(thought):
    pass


def evaluate_action_type(thought_chain):
    pass


def get_recent_messages_block(n_msgs: int, internal: bool = False):
    session = controller.get_session()
    events = session.exec(select(Event).order_by(col(Event.id))).fetchall()

    cnt = 0
    latest_events = events[::-1]
    lines = []
    for event in latest_events:
        if (event.interlocus >= 0) or internal:
            content = event.to_prompt_item(False)[0].to_tuple()[1].replace("\n", "")
            lines.append(f"{event.source}: {content}")
            if event.source != "system":
                cnt += 1
            if cnt >= n_msgs:
                break

    lines.reverse()
    all = "\n".join(lines)
    return all


def evalue_prompt_complexity():
    all = get_recent_messages_block(10)
    rating = rate_complexity_v2(all)
    return rating

def evalue_emotional_impact():
    all = get_recent_messages_block(10)
    last = get_recent_messages_block(1)
    rating = rate_emotional_impact(all, last)
    return rating

def get_facts_block(n_facts: int, search_string: str) -> str:
    emb = controller.get_embedding(search_string)
    session = controller.get_session()
    facts = session.exec(select(Fact).order_by(Fact.embedding.cosine_distance(emb)).limit(n_facts)).fetchall()
    return "\n".join([f.content for f in facts])

def get_learned_rules_count() -> int:
    session = controller.get_session()
    facts: List[CauseEffectDbEntry] = session.exec(select(CauseEffectDbEntry)).fetchall()
    return len(facts)

def get_learned_rules_block(n_facts: int, search_string: str) -> str:
    emb = controller.get_embedding(search_string)
    session = controller.get_session()
    facts: List[CauseEffectDbEntry] = session.exec(select(CauseEffectDbEntry).order_by(CauseEffectDbEntry.embedding.cosine_distance(emb)).limit(n_facts)).fetchall()
    return "\n".join([f"Cause: '{f.cause}' Effect: '{f.effect}'" for f in facts])


def generate_context_tot(complexity: float) -> Event:
    if complexity < 0.8:
        max_depth = 2
    else:
        max_depth = 2

    ctx = get_recent_messages_block(24, internal=True)
    k = get_facts_block(64, get_recent_messages_block(4))
    thought = generate_tot_v1(ctx, k, max_depth)
    #thought = rewrite_as_thought(thought)
    controller.cache_tot = thought

    event = Event(
        source=f"{companion_name}",
        content=thought,
        embedding=controller.get_embedding(thought),
        token=get_token(thought),
        timestamp=controller.get_timestamp(),
        interlocus=-1
    )

    return event


def generate_thought():
    ctx = get_recent_messages_block(8)
    lm = get_recent_messages_block(1)

    thought = execute_subsystem_thought(ctx, lm)
    return thought















































































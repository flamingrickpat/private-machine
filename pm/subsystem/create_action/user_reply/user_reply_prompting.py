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
from pm.system_utils import get_recent_messages_block
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

MAX_GENERATION_TOKENS = 4096
MAX_THOUGHTS_IN_PROMPT = 16

def get_prompt(story_mode: bool = False) -> List[PromptItem]:
    session = controller.get_session()
    events =        list(session.exec(select(Event).order_by(col(Event.id))).fetchall())
    clusters =      session.exec(select(Cluster).order_by(col(Cluster.id))).fetchall()
    event_cluster = session.exec(select(EventCluster).order_by(col(EventCluster.id))).fetchall()

    thought_cnt = 0
    for i in range(len(events) - 1, -1, -1):
        if events[i].interlocus < 0:
            thought_cnt += 1
            if thought_cnt > MAX_THOUGHTS_IN_PROMPT:
                del events[i]

    parts = get_optimized_prompt_temp_cluster(events, clusters, event_cluster,
                                              search_string=get_recent_messages_block(6),
                                              n_context_size=(controller.get_conscious_context() - controller.get_context_sys_prompt()),
                                              n_split=cluster_split,
                                              story_mode=story_mode)

    # add datetimes
    for i in range(len(parts) - 1, 1, -1):
        part_prev = parts[i - 1]
        part = parts[i]

        delta = (part.timestamp - part_prev.timestamp).total_seconds() / 60
        if delta > 180:
            pi = PromptItem(
                str(uuid.uuid4()),
                part.timestamp,
                "assistant" if story_mode else "system",
                "",
                f"Current time: {part.timestamp.strftime(timestamp_format)}",
                "",
                1)

            parts.insert(i, pi)

    # insert intermediate system prompt
    if len(parts) > 40 and not story_mode:
        for i in range(len(parts) - 32, len(parts) - 2):
            if parts[i].turn == "user":
                pi = PromptItem(
                    str(uuid.uuid4()),
                    datetime.now(),
                    "system",
                    "",
                    f"{sysprompt_addendum}",
                    "",
                    1)
                parts.insert(i + 1, pi)
                break

    return parts


def completion_story_mode():
    items = get_prompt(story_mode=True)
    msgs = [("system", f"You are a story writer that writes and endearing and fun story about the AI companion {companion_name} and her user {user_name}."
                       f"### BEGIN {companion_name} DESCRIPTION"
                       f"{char_card_3rd_person_neutral}"
                       f"### END {companion_name} DESCRIPTION")]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

    msgs.append(("assistant", f"{companion_name}:"))

    content = controller.completion_text(LlmPreset.Default, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024, stop_words=["Rick:", "Current time:"]),
                                         discard_thinks=False)
    return content


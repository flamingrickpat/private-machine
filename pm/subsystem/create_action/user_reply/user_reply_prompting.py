import uuid
from datetime import datetime
from typing import List

from sqlmodel import select, col

from pm.character import cluster_split, companion_name, timestamp_format, sysprompt_addendum, char_card_3rd_person_neutral, user_name
from pm.controller import controller
from pm.database.tables import Event, EventCluster, Cluster, PromptItem
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.prompt.get_optimized_prompt import get_optimized_prompt_temp_cluster
from pm.system_utils import get_recent_messages_block
from pm.validation.validate_directness import validate_directness
from pm.validation.validate_query_fulfillment import validate_query_fulfillment
from pm.validation.validate_response_in_context import validate_response_in_context

MAX_GENERATION_TOKENS = 4096
MAX_THOUGHTS_IN_PROMPT = 16


def get_prompt(story_mode: bool = False) -> List[PromptItem]:
    session = controller.get_session()
    events = list(session.exec(select(Event).order_by(col(Event.id))).fetchall())
    clusters = session.exec(select(Cluster).order_by(col(Cluster.id))).fetchall()
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
                "user" if story_mode else "system",
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


def completion_story_mode(subsystem_description: str) -> str:
    char_card = controller.format_str(char_card_3rd_person_neutral, extra={
        "subsystem_description": subsystem_description
    })

    items = get_prompt(story_mode=True)
    msgs = [("system", f"You are a story writer that writes and endearing and fun story about the AI companion {companion_name} and her user {user_name}."
                       f"### BEGIN {companion_name} DESCRIPTION"
                       f"{char_card}"
                       f"### END {companion_name} DESCRIPTION")]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

    msgs.append(("assistant", f"{companion_name}:"))
    lst_messages = get_recent_messages_block(6)

    while True:
        content = controller.completion_text(LlmPreset.Default, msgs,
                                             comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024,
                                                                              stop_words=[f"{user_name}:",
                                                                                          f"{companion_name}:",
                                                                                          "Current time:"]),
                                             discard_thinks=False)

        rating = validate_response_in_context(lst_messages, content)
        rating_directness = validate_directness(lst_messages, content)
        rating_fulfilment = validate_query_fulfillment(lst_messages, content)

        if rating >= 0.75 and rating_directness >= 0.75 and rating_fulfilment > 0.75:
            return content.strip().removeprefix('"').removesuffix('"').strip()


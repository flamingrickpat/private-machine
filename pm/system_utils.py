from datetime import datetime
from typing import List

from sqlmodel import select, col

from pm.character import timestamp_format
from pm.common_prompts.rate_complexity import rate_complexity
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.controller import controller
from pm.database.tables import Event, Fact, CauseEffectDbEntry

res_tag = "<<<RESULT>>>"


def init():
    pass

def get_sensation(allow_timeout: bool = False, override: str = None):
    return controller.get_user_input(override=override)


def get_independent_thought_system_msg():
    return ("system", f"System Message: The user hasn't responded yet, entering autonomous thinking mode. Current timestamp: {datetime.now().strftime(timestamp_format)}\n"
                      f"Please use the provided tools to select an action type.")

def get_public_event_count() -> int:
    session = controller.get_session()
    events = list(session.exec(select(Event).where(Event.interlocus == 1).order_by(col(Event.id))).fetchall())
    return len(events)

def get_recent_messages_block(n_msgs: int, internal: bool = False, max_tick: int = -1):
    session = controller.get_session()

    if max_tick <= 0:
        events = session.exec(select(Event).order_by(col(Event.id))).fetchall()
    else:
        events = session.exec(select(Event).where(Event.tick_id <= max_tick).order_by(col(Event.id))).fetchall()

    cnt = 0
    latest_events = events[::-1]
    lines = []
    for event in latest_events:
        if event.interlocus >= 0 or internal:
            content = event.to_prompt_item(False)[0].to_tuple()[1].replace("\n", "")
            lines.append(f"{event.timestamp.strftime(timestamp_format)}, {event.source}: {content}")

            if event.interlocus != 0:
                cnt += 1
            if cnt >= n_msgs:
                break

    lines.reverse()
    all = "\n".join(lines)
    return all


def evalue_prompt_complexity():
    all = get_recent_messages_block(10)
    rating = rate_complexity(all)
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


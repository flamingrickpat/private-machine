from datetime import datetime
from typing import List

from sqlmodel import select, col

from pm.character import timestamp_format, user_name
from pm.common_prompts.rate_complexity import rate_complexity
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.controller import controller
from pm.database.tables import Event, Fact, CauseEffect, FactCategory
from pm.embedding.embedding import get_cos_sim, get_embedding, vector_search
from pm.subsystem.sensation_evaluation.emotion.emotion_agent_group import agent_joy, agent_hate

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

def get_recent_user_messages_block(n_msgs: int) -> str:
    session = controller.get_session()
    events = session.exec(select(Event).order_by(col(Event.id))).fetchall()
    cnt = 0
    latest_events = events[::-1]
    lines = []
    for event in latest_events:
        if event.interlocus == 1 and event.source == user_name:
            content = event.to_prompt_item(False)[0].to_tuple()[1].replace("\n", "")
            lines.append(f"{event.timestamp.strftime(timestamp_format)}, {event.source}: {content}")
            cnt += 1
            if cnt >= n_msgs:
                break

    lines.reverse()
    all = "\n".join(lines)
    return all

def get_recent_messages_block_user(n_msgs: int, internal: bool = False, max_tick: int = -1):
    session = controller.get_session()

    if max_tick <= 0:
        events = session.exec(select(Event).order_by(col(Event.id))).fetchall()
    else:
        events = session.exec(select(Event).where(Event.tick_id <= max_tick).order_by(col(Event.id))).fetchall()

    cnt = 0
    latest_events = events[::-1]
    lines = []
    for event in latest_events:
        if event.source == user_name:
            if event.interlocus >= 0 or internal:
                content = event.to_prompt_item(False)[0].to_tuple()[1].replace("\n", "")
                lines.append(f"{event.source}: {content}")

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
    facts = session.exec(select(Fact)).fetchall()
    best_facts = vector_search(facts, emb, n_facts)
    return "\n".join([f.content for f in best_facts])

def get_learned_rules_count() -> int:
    session = controller.get_session()
    facts: List[CauseEffect] = session.exec(select(CauseEffect)).fetchall()
    return len(facts)

def get_learned_rules_block(n_facts: int, search_string: str) -> str:
    emb = controller.get_embedding(search_string)
    session = controller.get_session()
    facts: List[CauseEffect] = session.exec(select(CauseEffect)).fetchall()
    best_facts = vector_search(facts, emb, n_facts)
    return "\n".join([f"Cause: '{f.cause}' Effect: '{f.effect}'" for f in best_facts])

def return_biased_facts_with_emb(bias_dict: dict, embedding: list[float], bias_strength: float = 0.5, n_facts: int = 10) -> str:
    session = controller.get_session()

    # Retrieve all facts
    facts = session.exec(select(Fact)).all()
    scored_facts = []

    for fact in facts:
        similarity = get_cos_sim(embedding, fact.embedding)
        categories = session.exec(select(FactCategory).where(FactCategory.fact_id == fact.id)).all()
        bias_score = 0.0
        if categories:
            for cat in categories:
                if cat.category_id in bias_dict:
                    norm_weight = (cat.weight / cat.max_weight) if cat.max_weight else 0
                    bias_score += norm_weight * bias_dict[cat.category_id]
            bias_score /= len(categories)

        final_score = (similarity * (1 - bias_strength)) + (bias_score * bias_strength)
        scored_facts.append((fact, final_score))

    # Select the top n_facts based on final_score
    scored_facts.sort(key=lambda x: x[1], reverse=True)
    top_facts = scored_facts[:n_facts]

    # Finally, sort these top facts by their id (chronological order)
    top_facts = sorted(top_facts, key=lambda x: x[0].id)

    return "\n".join([f"Fact: '{fact.content}' (ID: {fact.id}, Score: {score:.2f})" for fact, score in top_facts])


if __name__ == '__main__':
    controller.start()
    controller.init_db()
    res = return_biased_facts_with_emb(agent_hate.fact_bias, embedding=get_embedding("berlin"), n_facts=15)
    print(res)
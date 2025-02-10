import sys
import os
import traceback
import uuid

from pydantic import BaseModel

from pm.cosine_clustering.cosine_cluster import get_best_category, cluster_text, TEMPLATE_CATEGORIES
from pm.fact_learners.extract_cause_effect_agent import extract_cas

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import time
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Optional, Any, List, Tuple

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

from pm.character import sysprompt, database_uri, cluster_split, companion_name, timestamp_format, sysprompt_addendum
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.emotion.generate_emotion import generate_first_person_emotion
from pm.common_prompts.get_emotional_state import get_emotional_state
from pm.meta_learning.integrate_rules_final_output import integrate_rules_final_output

res_tag = "<<<RESULT>>>"

def get_engine():
    return create_engine(database_uri)

def init_db():
    engine = get_engine()
    try:
        with Session(engine) as session:
            session.exec(text("CREATE EXTENSION vector;"))
            session.commit()
    except:
        pass

    reset = False

    try:
        with Session(engine) as session:
            if reset:
                session.exec(text("drop table event;"))
                session.exec(text("drop table cluster;"))
                session.exec(text("drop table eventcluster;"))
                session.exec(text("drop table fact;"))
                session.commit()

        SQLModel.metadata.create_all(engine)
    except:
        pass

def init():
    pass

def check_tasks():
    time.sleep(1)
    session = controller.get_session()
    statement = select(Event).where(col(Event.id).not_in(select(col(EventCluster.cog_event_id)))).where(Event.interlocus > 0).order_by(col(Event.id))
    unclustered_events: List[Event] = session.exec(statement).all()

    sum_token = 0
    for event in unclustered_events:
        sum_token += event.token

    limit = (controller.get_conscious_context() - controller.get_context_sys_prompt()) * cluster_split
    if sum_token > limit:
        clusterize()

    factorize()

def events_to_cluster(session, events: List[Event]) -> Cluster:
    content = ", ".join([str(x.id) for x in events])
    token = sum([x.token for x in events])

    max_token = max([x.token for x in events])
    emb = [e * (events[0].token / max_token) for e in events[0].embedding]
    for event in events[1:]:
        for i in range(len(emb)):
            emb[i] += (event.embedding[i] * (event.token / max_token))
    for i in range(len(emb)):
        emb[i] /= len(events)

    cluster = Cluster(
        level=100,
        type=ClusterType.Topical,
        summary="Topic: " + content,
        included_messages=content,
        token=token,
        embedding=emb,
        timestamp_from=events[0].timestamp - timedelta(seconds=5),
        timestamp_to=events[-1].timestamp + timedelta(seconds=5)
    )

    session.add(cluster)
    session.flush()
    session.refresh(cluster)
    cluster_id = cluster.id

    lines = []
    events.sort(key=lambda x: x.timestamp)
    for event in events:
        cecc = EventCluster(cog_event_id=event.id, con_cluster_id=cluster_id)
        session.add(cecc)
        lines.append(f"{event.source}: {event.content}")

    return cluster


def events_to_facts(session, events: List[Event]):
    lines = []
    events.sort(key=lambda x: x.timestamp)
    for event in events:
        lines.append(f"{event.source}: {event.content}")

    max_fact_event_id = session.exec(func.max(Fact.max_event_id)).scalar() or 0

    # Check if the minimum event ID in the cluster is greater than the maximum event ID in the Fact table
    min_event_id = min([event.id for event in events])
    if min_event_id > max_fact_event_id:
        block = "\n".join(lines)
        facts = extract_facts(block)
        if True:
            for fact in facts:
                fstr = fact[0]
                temp = fact[1]
                max_event_id = events[-1].id

                f = Fact(
                    content=fstr,
                    temporality=temp,
                    max_event_id=max_event_id,
                    embedding=controller.get_embedding(fstr),
                    token=get_token(fstr)
                )
                session.add(f)
                session.flush()
                session.refresh(f)



def events_to_ca(session, events: List[Event]):
    lines = []
    events.sort(key=lambda x: x.timestamp)
    for event in events:
        lines.append(f"{event.source}: {event.content}")

    max_fact_event_id = session.exec(func.max(CauseEffectDbEntry.max_event_id)).scalar() or 0

    # Check if the minimum event ID in the cluster is greater than the maximum event ID in the Fact table
    min_event_id = min([event.id for event in events])
    if min_event_id > max_fact_event_id:
        block = "\n".join(lines)
        cas = extract_cas(block)
        if True:
            for ca in cas:
                fstr = f"cause: {ca.cause} effect: {ca.effect}"
                clusters = cluster_text(fstr, template_dict=TEMPLATE_CATEGORIES)
                max_event_id = events[-1].id

                f = CauseEffectDbEntry(
                    cause=ca.cause,
                    effect=ca.effect,
                    category=get_best_category(clusters),
                    max_event_id=max_event_id,
                    embedding=controller.get_embedding(fstr),
                    token=get_token(fstr),
                    timestamp=events[0].timestamp
                )
                session.add(f)
                session.flush()
                session.refresh(f)


def temporal_cluster_to_cluster(clusters: List[TemporalCluster]):
    session = controller.get_session()
    msgs_prev = []
    for cluster in clusters:
        event_ids = []
        for context_cluster_id in cluster.items:
            statement = select(EventCluster).where(EventCluster.con_cluster_id == context_cluster_id)
            tmp: List[EventCluster] = session.exec(statement).all()
            for ec in tmp:
                event_ids.append(ec.cog_event_id)

        msgs = []
        for event_id in event_ids:
            statement = select(Event).where(Event.id == event_id)
            tmp: List[Event] = session.exec(statement).all()
            if len(tmp) > 0:
                event = tmp[0]
                msgs.append(f"{event.source}: {event.content}")

        same_as_prev = all(x == y for x, y in zip(msgs, msgs_prev))
        msgs_prev = msgs

        if same_as_prev:
            continue

        sum = summarize_messages(msgs)
        tc = Cluster(
            level=temp_cluster_to_level(cluster.raw_type),
            type=ClusterType.Temporal,
            summary=sum,
            included_messages=", ".join([str(e) for e in event_ids]),
            token=get_token(sum),
            embedding=controller.get_embedding(sum),
            timestamp_from=cluster.timestamp_begin - timedelta(seconds=5),
            timestamp_to=cluster.timestamp_end + timedelta(seconds=5)
        )

        session.add(tc)
        session.flush()
        session.refresh(tc)
        cluster_id = tc.id

        for event_id in event_ids:
            cecc = EventCluster(cog_event_id=event_id, con_cluster_id=cluster_id)
            session.add(cecc)
            session.flush()
            session.refresh(cecc)


def clusterize():
    print("Clustering...")
    session = controller.get_session()
    session.exec(text("truncate eventcluster"))
    session.exec(text("truncate cluster"))

    statement = select(Event).where(Event.interlocus > 0).order_by(col(Event.id))
    events: List[Event] = session.exec(statement).all()

    event_map = {x.id: x for x in events}
    cluster_desc = get_optimal_clusters(events)

    clusters = []
    prev_cluster = -1
    event_buffer = []
    for id, cluster in cluster_desc.topic_switch_cluster.items():
        event = event_map[id]

        if cluster != prev_cluster and len(event_buffer) > 0:
            clusters.append(events_to_cluster(session, event_buffer))
            event_buffer = []

        event_buffer.append(event)
        prev_cluster = cluster

    clusters.append(events_to_cluster(session, event_buffer))

    # temporal cluster
    tmp = []
    for topic_cluster in clusters:
        a = topic_cluster.timestamp_from
        b = topic_cluster.timestamp_to
        tmp.append(MiniContextCluster(id=topic_cluster.id, timestamp=a + (b - a)/2))

    temp_clusters = temporally_cluster_context(tmp)
    temporal_cluster_to_cluster(temp_clusters)


def factorize():
    session = controller.get_session()
    statement = select(Event).order_by(col(Event.id))
    events: List[Event] = session.exec(statement).all()

    if len(events) < 16:
        return

    print("Extracting facts...")

    #events = events[:-8]
    event_map = {x.id: x for x in events}
    cluster_desc = get_optimal_clusters(events)

    clusters = []
    prev_cluster = -1
    event_buffer = []
    for id, cluster in cluster_desc.topic_switch_cluster.items():
        event = event_map[id]

        if cluster != prev_cluster and len(event_buffer) > 0:
            events_to_facts(session, event_buffer)
            events_to_ca(session, event_buffer)
            event_buffer = []

        event_buffer.append(event)
        prev_cluster = cluster


def get_sensation(allow_timeout: bool = False, override: str = None):
    return controller.get_user_input(override=override)


def completion_conscious(items, preset: LlmPreset) -> Event:
    msgs = [("system", sysprompt)]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

        content = controller.completion_text(preset, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=4096))

        cache_user_input = controller.cache_user_input
        cache_emotion = controller.cache_emotion
        cache_tot = controller.cache_tot
        facts = get_learned_rules_block(64, cache_user_input + cache_emotion + cache_tot + content)
        feedback = integrate_rules_final_output(cache_user_input, cache_emotion, cache_tot, content, facts)

        msgs.append(("system", "System Warning: Answer does not pass the meta-learning check and was not sent."
                               f"Please use these tips to craft a new, better response: ### BEGIN TIPS\n{feedback}\n### END TIPS"))

        content = controller.completion_text(preset, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=4096))
        event = Event(
            source=f"{companion_name}",
            content=content.strip(),
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=1
        )
        return event


class OperationMode(StrEnum):
    UserConversation = "UserConversation"
    InternalContemplation = "InternalContemplation"
    #VirtualWorld = "VirtualWorld"

class ActionType(StrEnum):
    InitiateUserConversation = "InitiateUserConversation"
    InitiateInternalContemplation = "InitiateInternalContemplation"
    InitiateIdleMode = "InitiateIdleMode"

class ActionSelector(BaseModel):
    thoughts: str = Field(description="pros and cons for different selections")
    action_type: ActionType = Field(description="the action type")
    reason: str = Field(description="why you chose this action type?")

def get_independant_thought_system_msg():
    return ("system", f"System Message: The user hasn't responded yet, entering autonomous thinking mode. Current timestamp: {datetime.now().strftime(timestamp_format)}\n"
                      f"Please use the provided tools to select an action type.")

def determine_action_type(items) -> ActionSelector:
    msgs = [("system", sysprompt)]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

    _, calls = controller.completion_tool(LlmPreset.Conscious, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024, presence_penalty=1, frequency_penalty=1),
                                          tools=[ActionSelector])
    return calls[0]

class MetaSystemOperationModeDecision(BaseModel):
    reason: str = Field(description=f"Why {companion_name} wants to be in the operating mode")
    mode: OperationMode = Field(description="the target operation mode")


def get_action(prompt: List[PromptItem], complexity: float):
    preset = LlmPreset.Conscious
    return completion_conscious(prompt, preset=preset)

def add_cognitive_event(event: Event):
    if event is None:
        return

    session = controller.get_session()
    session.add(event)
    session.flush()
    session.refresh(event)

def get_prompt() -> List[PromptItem]:
    session = controller.get_session()
    events =        list(session.exec(select(Event).order_by(col(Event.id))).fetchall())
    clusters =      session.exec(select(Cluster).order_by(col(Cluster.id))).fetchall()
    event_cluster = session.exec(select(EventCluster).order_by(col(EventCluster.id))).fetchall()

    thought_cnt = 0
    for i in range(len(events) - 1, -1, -1):
        if events[i].interlocus < 0:
            thought_cnt += 1
            if thought_cnt > 6:
                del events[i]

    parts = get_optimized_prompt_temp_cluster(events, clusters, event_cluster,
                                              search_string=get_recent_messages_block(6),
                                              n_context_size=(controller.get_conscious_context() - controller.get_context_sys_prompt()),
                                              n_split=cluster_split)

    # add datetimes
    for i in range(len(parts) - 1, 1, -1):
        part_prev = parts[i - 1]
        part = parts[i]

        delta = (part.timestamp - part_prev.timestamp).total_seconds() / 60
        if delta > 180:
            pi = PromptItem(
                str(uuid.uuid4()),
                part.timestamp,
                "system",
                "",
                f"Current time: {part.timestamp.strftime(timestamp_format)}",
                "",
                1)

            parts.insert(i, pi)

    # insert intermediate system prompt
    if len(parts) > 20:
        for i in range(len(parts) - 16, len(parts) - 2):
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

    return parts


def generate_thought():
    pass


def generate_tot(thought):
    pass


def evaluate_action_type(thought_chain):
    pass


def get_recent_messages_block(n_msgs: int):
    session = controller.get_session()
    events = session.exec(select(Event).order_by(col(Event.id))).fetchall()

    latest_events = events[-n_msgs:]
    lines = []
    for event in latest_events:
        content = event.to_prompt_item()[0].to_tuple()[1]
        lines.append(f"{event.source}: {content}")
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


def get_learned_rules_block(n_facts: int, search_string: str) -> str:
    emb = controller.get_embedding(search_string)
    session = controller.get_session()
    facts: List[CauseEffectDbEntry] = session.exec(select(CauseEffectDbEntry).order_by(CauseEffectDbEntry.embedding.cosine_distance(emb)).limit(n_facts)).fetchall()
    return "\n".join([f"Cause: '{f.cause}' Effect: '{f.effect}'" for f in facts])


def generate_context_tot(complexity: float) -> Event:
    if complexity < 0.8:
        max_depth = 2
    else:
        max_depth = 4

    ctx = get_recent_messages_block(24)
    k = get_facts_block(64, get_recent_messages_block(4))
    thought = generate_tot_v1(ctx, k, max_depth)

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


def generate_emotion_tot():
    ctx = get_recent_messages_block(8)
    lm = get_recent_messages_block(1)
    thought = generate_first_person_emotion(ctx, lm)

    controller.cache_emotion = thought

    event = Event(
        source=f"{companion_name}",
        content=thought,
        embedding=controller.get_embedding(thought),
        token=get_token(thought),
        timestamp=controller.get_timestamp(),
        interlocus=-2
    )
    return event


class InitiateUserItem(BaseModel):
    user_name: str = Field(description="name of recipient")
    thought: str = Field(description="why you want to send the message?")
    message: str = Field(description="the message to send")


def get_message_to_user(items) -> InitiateUserItem:
    msgs = [("system", sysprompt)]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

    _, calls = controller.completion_tool(LlmPreset.Conscious, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024, presence_penalty=1, frequency_penalty=1),
                                          tools=[InitiateUserItem])
    return calls[0]

class InternalContemplationItem(BaseModel):
    contemplation: str = Field(description="your contemplation about anything and stuff")

def get_internal_contemplation(items) -> InternalContemplationItem:
    msgs = [("system", sysprompt)]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

    _, calls = controller.completion_tool(LlmPreset.Conscious, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024, presence_penalty=1, frequency_penalty=1),
                                          tools=[InternalContemplationItem])
    return calls[0]


def run_auto_thinking_procedure() -> str:
    content = get_independant_thought_system_msg()[1]
    event = Event(
        source=f"system",
        content=content,
        embedding=controller.get_embedding(content),
        token=get_token(content),
        timestamp=datetime.now(),
        interlocus=0
    )
    controller.get_session().add(event)

    prompt = get_prompt()
    action_type = determine_action_type(prompt)
    #print(action_type.reason)
    content = None
    if action_type.action_type == ActionType.InitiateUserConversation:
        content = f"I will message my user because of the following reasons: {action_type.reason}"
    elif action_type.action_type == ActionType.InitiateIdleMode:
        content = f"I will stay idle until the next heartbeat because of the following reasons: {action_type.reason}"
    elif action_type.action_type == ActionType.InitiateInternalContemplation:
        content = f"I will contemplate for the time being because of the following reasons: {action_type.reason}"

    event = Event(
        source=f"{companion_name}",
        content=content,
        embedding=controller.get_embedding(content),
        token=get_token(content),
        timestamp=datetime.now(),
        interlocus=InterlocusType.ActionDecision.value
    )
    controller.get_session().add(event)

    prompt = get_prompt()
    if action_type.action_type == ActionType.InitiateUserConversation:
        item = get_message_to_user(prompt)

        content = item.thought
        event = Event(
            source=f"{companion_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Thought.value
        )
        controller.get_session().add(event)


        content = item.message
        event = Event(
            source=f"{companion_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Public.value
        )
        controller.get_session().add(event)
        return content
    elif action_type.action_type == ActionType.InitiateIdleMode:
        pass
    elif action_type.action_type == ActionType.InitiateInternalContemplation:
        item = get_internal_contemplation(prompt)

        content = item.contemplation
        event = Event(
            source=f"{companion_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Thought.value
        )
        controller.get_session().add(event)
        return ""

def run_cli() -> str:
    inp = sys.argv[1]
    if inp.strip() == "":
        print("no input")
        exit(1)

    res = run_mp(inp)
    print(res_tag + res)

def run_mp(inp: str) -> str:
    if inp.strip() == "/think":
        res = run_auto_thinking_procedure()
        return res
    if inp.strip() == "/sleep":
        check_tasks()
        return "clustered and extracted facts!"
    else:
        controller.cache_user_input = inp.strip()
        check_tasks()
        try:
            sensation = get_sensation(override=inp)
            add_cognitive_event(sensation)

            emotional_impact = evalue_emotional_impact()
            if emotional_impact > 0.5:
                emotion_chain = generate_emotion_tot()
                add_cognitive_event(emotion_chain)

            complexity = evalue_prompt_complexity()
            if complexity > 0.5:
                thought_chain = generate_context_tot(complexity)
                add_cognitive_event(thought_chain)

            prompt = get_prompt()
            action = get_action(prompt, complexity)
            res = action.content
            add_cognitive_event(action)
            return res
        except Exception as e:
            return str(e)

#def run():
#    auto_thinking_mode = False
#    while True:
#        check_tasks()
#        try:
#            sensation = get_sensation(allow_timeout=not auto_thinking_mode)
#            add_cognitive_event(sensation)
#
#            emotional_impact = evalue_emotional_impact()
#            if emotional_impact > 0.5:
#                emotion_chain = generate_emotion_tot()
#                add_cognitive_event(emotion_chain)
#
#            complexity = evalue_prompt_complexity()
#            if complexity > 0.5:
#                thought_chain = generate_context_tot(complexity)
#                add_cognitive_event(thought_chain)
#
#            prompt = get_prompt()
#            action = get_action(prompt)
#            add_cognitive_event(action)
#        except TimeoutError:
#            auto_thinking_mode = True
#            thought = generate_thought()
#            thought_chain = generate_tot(thought)
#            add_cognitive_event(thought_chain)
#            action_type = evaluate_action_type(thought_chain)
#            if action_type == ActionType.MessageUser:
#                pass
#            elif action_type == ActionType.ContinueInternalMonolouge:
#                pass
#            elif action_type == ActionType.Sleep:
#                pass


def get_action_only():
    prompt = get_prompt()
    #action = get_action(prompt, 0)

    all = get_recent_messages_block(10)
    last = get_recent_messages_block(1)
    rating = rate_emotional_impact(all, last)

    all = get_recent_messages_block(10)
    last = get_recent_messages_block(1)
    state = get_emotional_state(all, last)

    return rating, state

TEST_MODE = True

def run_system_cli():
    init_db()
    init()

    controller.init_db()
    try:
        run_cli()
        if TEST_MODE:
            controller.rollback_db()
        else:
            controller.commit_db()
    except Exception as e:
        print(res_tag)
        print(e)
        print(traceback.format_exc())
        controller.rollback_db()


def run_system_mp(inp: str) -> str:
    init_db()
    init()
    res = ""

    controller.init_db()
    try:
        res = run_mp(inp)
        if TEST_MODE:
            controller.rollback_db()
        else:
            controller.commit_db()
    except Exception as e:
        print(res_tag)
        print(e)
        print(traceback.format_exc())
        controller.rollback_db()
    return res


if __name__ == '__main__':
    run_system_cli()

import sys
import os
import uuid

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
from pm.database.tables import Event, CogEventType, EventCluster, Cluster, ClusterType, PromptItem, Fact
from pm.prompt.get_optimized_prompt import get_optimized_prompt, create_optimized_prompt, get_optimized_prompt_temp_cluster
from pm.thought.generate_tot import generate_tot_v1

from pm.character import sysprompt, database_uri, CONTEXT_SIZE, CONTEXT_SYS_PROMPT, CLUSTER_SPLIT, companion_name, TIMESTAMP_FORMAT
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.common_prompts.rate_emotional_impact import rate_emotional_impact
from pm.emotion.generate_emotion import generate_first_person_emotion

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
    with Session(get_engine()) as session:
        statement = select(Event).where(col(Event.id).not_in(select(col(EventCluster.cog_event_id)))).where(Event.interlocus > 0).order_by(col(Event.id))
        unclustered_events: List[Event] = session.exec(statement).all()

    sum_token = 0
    for event in unclustered_events:
        sum_token += event.token

    limit = (CONTEXT_SIZE - CONTEXT_SYS_PROMPT) * CLUSTER_SPLIT
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
    session.commit()
    cluster_id = cluster.id

    lines = []
    events.sort(key=lambda x: x.timestamp)
    for event in events:
        cecc = EventCluster(cog_event_id=event.id, con_cluster_id=cluster_id)
        session.add(cecc)
        lines.append(f"{event.source}: {event.content}")

    session.commit()
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
                session.commit()

def temporal_cluster_to_cluster(clusters: List[TemporalCluster]):
    with Session(get_engine(), expire_on_commit=False) as session:
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
            session.commit()
            cluster_id = tc.id

            for event_id in event_ids:
                cecc = EventCluster(cog_event_id=event_id, con_cluster_id=cluster_id)
                session.add(cecc)
            session.commit()


def clusterize():
    print("Clustering...")
    with Session(get_engine(), expire_on_commit=False) as session:
        session.exec(text("truncate eventcluster"))
        session.exec(text("truncate cluster"))
        session.commit()

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
    with Session(get_engine(), expire_on_commit=False) as session:
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
            event_buffer = []

        event_buffer.append(event)
        prev_cluster = cluster

    # skip current one
    if False:
        events_to_facts(session, event_buffer)

def get_sensation(allow_timeout: bool = False, override: str = None):
    return controller.get_user_input(override=override)


def completion_conscious(items, preset: LlmPreset) -> Event:
    if controller.backtest_messages:
        if items is not None:
            #logger.info("######### BEGIN PROMPT ###########")
            for item in items:
                #logger.info(item)
                print(item)
            #logger.info("######### END PROMPT ###########")

        if len(controller.messages) == 0:
            exit(1)
        msg = controller.messages[0]
        del controller.messages[0]
        print(msg.__repr__())
        return msg
    else:
        msgs = [("system", sysprompt)]
        if items is not None:
            for item in items:
                msgs.append(item.to_tuple())

        while True:
            content = controller.completion_text(preset, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1, max_tokens=1024))
            if "<think>" in content:
                msgs.append(("assistant", content))
            else:
                #print(f"{companion_name}: {content.strip()}")
                event = Event(
                    source=f"{companion_name}",
                    content=content.strip(),
                    embedding=controller.get_embedding(content),
                    token=get_token(content),
                    timestamp=datetime.now(),
                    interlocus=1
                )
                return event
                
def get_action(prompt: List[PromptItem], complexity: float):
    if complexity > 0.75:
        preset = LlmPreset.Best
    else:
        preset = LlmPreset.Good

    return completion_conscious(prompt, preset=preset)

def add_cognitive_event(event: Event):
    if event is None:
        return

    with Session(get_engine()) as session:
        session.add(event)
        session.commit()

def get_prompt() -> List[PromptItem]:
    with Session(get_engine()) as session:
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
                                              n_context_size=(CONTEXT_SIZE - CONTEXT_SYS_PROMPT),
                                              n_split=CLUSTER_SPLIT)

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
                f"Current time: {part.timestamp.strftime(TIMESTAMP_FORMAT)}",
                "",
                1)

            parts.insert(i, pi)

    return parts


def generate_thought():
    pass


def generate_tot(thought):
    pass


def evaluate_action_type(thought_chain):
    pass


class ActionType(StrEnum):
    Sleep = "Sleep"
    ContinueInternalMonolouge = "ContinueInternalMonolouge"
    MessageUser = "MessageUser"


def get_recent_messages_block(n_msgs: int):
    with Session(get_engine()) as session:
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
    with Session(get_engine()) as session:
        facts = session.exec(select(Fact).order_by(Fact.embedding.cosine_distance(emb)).limit(n_facts)).fetchall()
        return "\n".join([f.content for f in facts])


def generate_context_tot(complexity: float) -> Event:
    if complexity < 0.8:
        max_depth = 2
    else:
        max_depth = 4

    ctx = get_recent_messages_block(24)
    k = get_facts_block(64, get_recent_messages_block(4))
    thought = generate_tot_v1(ctx, k, max_depth)
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
    event = Event(
        source=f"{companion_name}",
        content=thought,
        embedding=controller.get_embedding(thought),
        token=get_token(thought),
        timestamp=controller.get_timestamp(),
        interlocus=-2
    )
    return event


def run_telegram():
    inp = sys.argv[1]
    if inp.strip() == "":
        print("no input")
        exit(1)

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
        print("<<<RESULT>>>" + res)
    except Exception as e:
        print("<<<RESULT>>>" + str(e))

def output_prompt_block():
    all = get_recent_messages_block(10)
    print(all)

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

def run_system():
    init_db()
    init()
    #run()
    run_telegram()
    #output_prompt_block()

if __name__ == '__main__':
    run_system()
import time
from datetime import timedelta
from typing import List

from sqlalchemy import text, func
from sqlmodel import select, col

from pm.character import cluster_split
from pm.cluster.cluster_temporal import MiniContextCluster, temporally_cluster_context, TemporalCluster, temp_cluster_to_level
from pm.cluster.cluster_utils import get_optimal_clusters
from pm.common_prompts.extract_facts import extract_facts
from pm.common_prompts.summary_perspective import summarize_messages
from pm.controller import controller
from pm.cosine_clustering.cosine_cluster import get_best_category, cluster_text, TEMPLATE_CATEGORIES
from pm.database.tables import Event, EventCluster, Cluster, ClusterType, Fact, CauseEffectDbEntry
from pm.embedding.token import get_token
from pm.fact_learners.extract_cause_effect_agent import extract_cas


def optimize_memory():
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

        summary = summarize_messages(msgs)
        tc = Cluster(
            level=temp_cluster_to_level(cluster.raw_type),
            type=ClusterType.Temporal,
            summary=summary,
            included_messages=", ".join([str(e) for e in event_ids]),
            token=get_token(summary),
            embedding=controller.get_embedding(summary),
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
    for _id, cluster in cluster_desc.topic_switch_cluster.items():
        event = event_map[_id]

        if cluster != prev_cluster and len(event_buffer) > 0:
            events_to_facts(session, event_buffer)
            events_to_ca(session, event_buffer)
            event_buffer = []

        event_buffer.append(event)
        prev_cluster = cluster

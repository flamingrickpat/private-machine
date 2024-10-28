import datetime
import uuid

import duckdb

from pm.agents.extract_facts import extract_facts_from_messages
from pm.agents.summary import summarize_messages_for_l0_summary
from pm.agents.summary_summary import summarize_summary_for_ln_summary
from pm.clustering.agg_clustering import get_agglomerative_clusters_as_dict, get_sliding_window_embedded_messages
from pm.database.db_helper import fetch_messages_no_summary, fetch_messages, get_padded_subset, insert_object, \
    fetch_summaries, fetch_summaries_no_summary
from pm.database.db_model import User, Message, Conversation, ConceptualCluster, MessageSummary, Relation, Fact
from pm.controller import controller
from pm.utils.token_utils import quick_estimate_tokens

controller.start()

def cluster_and_summarize(conversation_id: str):
    no_summary = fetch_messages_no_summary(conversation_id)
    all = fetch_messages(conversation_id)

    subset_plain = all[-256:]
    subset = get_sliding_window_embedded_messages(subset_plain)
    clusters = get_agglomerative_clusters_as_dict(subset)

    last_cluster = None
    cur_cluster_list = []
    cur_cluster_list_padded = []
    for i, msg in enumerate(subset):
        if not last_cluster:
            last_cluster = clusters[msg.id]

        cur_cluster = clusters[msg.id]
        if cur_cluster != last_cluster:
            if i < len(subset) - 1:
                cur_cluster_list_padded.append(subset[i + 1])

            do_summarize = True
            for cluster_msg in cur_cluster_list:
                if cluster_msg not in no_summary:
                    do_summarize = False
                    break

            if do_summarize:
                summary = summarize_messages_for_l0_summary(cur_cluster_list)
                summary_id = str(uuid.uuid4())
                cluster = MessageSummary(
                    id=summary_id,
                    conversation_id=conversation_id,
                    level=0,
                    text=summary,
                    embedding=controller.embedder.get_embedding_scalar_float_list(summary),
                    tokens=quick_estimate_tokens(summary),
                    world_time_begin=cur_cluster_list[0].world_time - datetime.timedelta(seconds=1),
                    world_time_end=cur_cluster_list[-1].world_time
                )
                insert_object(cluster)

                facts = extract_facts_from_messages(cur_cluster_list)
                for fact in facts.facts:
                    f = Fact(
                        id=str(uuid.uuid4()),
                        conversation_id=conversation_id,
                        text=fact.fact,
                        importance=fact.importance,
                        embedding=controller.embedder.get_embedding_scalar_float_list(fact.fact),
                        category=fact.category,
                        tokens=quick_estimate_tokens(fact.fact),
                    )
                    insert_object(f)

                for cluster_msg in cur_cluster_list:
                    rel = Relation(
                        a=cluster_msg.id,
                        b=summary_id,
                        rel_ab="summarized_by"
                    )
                    insert_object(rel)

            cur_cluster_list = []
            cur_cluster_list_padded = []
            if i > 0:
                cur_cluster_list_padded.append(subset[i - 1])

            cur_cluster_list_padded.append(msg)
            cur_cluster_list.append(msg)
        else:
            cur_cluster_list_padded.append(msg)
            cur_cluster_list.append(msg)
        last_cluster = cur_cluster


def high_level_summarize(conversation_id: str):
    cur_level = 1
    while True:
        cur_cnt = 0
        all_summaries_lower_layer = fetch_summaries(conversation_id, level=cur_level-1)
        all_summaries_lower_layer_no_summary = fetch_summaries_no_summary(conversation_id, level=cur_level - 1)

        for i in range(1, len(all_summaries_lower_layer) - 3, 2):
            cur_summaries = all_summaries_lower_layer[i - 1:i + 2]

            do_summarize = True
            for s in cur_summaries:
                if s not in all_summaries_lower_layer_no_summary:
                    do_summarize = False
                    break

            if do_summarize:
                block = "\n".join([x.text for x in cur_summaries])
                summary = summarize_summary_for_ln_summary(block)
                cur_cnt += 1

                summary_id = str(uuid.uuid4())
                cluster = MessageSummary(
                    id=summary_id,
                    conversation_id=conversation_id,
                    level=cur_level,
                    text=summary,
                    embedding=controller.embedder.get_embedding_scalar_float_list(summary),
                    tokens=quick_estimate_tokens(summary),
                    world_time_begin=cur_summaries[0].world_time - datetime.timedelta(seconds=1),
                    world_time_end=cur_summaries[-1].world_time
                )
                insert_object(cluster)

                for cur_sum in cur_summaries:
                    rel = Relation(
                        a=cur_sum.id,
                        b=summary_id,
                        rel_ab="summarized_by"
                    )
                    insert_object(rel)

        cur_level += 1
        if cur_cnt == 0:
            break
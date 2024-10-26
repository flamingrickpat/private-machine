from typing import List, Union, Tuple

from pm.database.db_model import Message, MessageSummary, Relation

from typing import List, Tuple, Set
from datetime import datetime

def build_prompt(all_messages: List[Tuple['Message', float]],
                 all_level_summaries: List[Tuple['MessageSummary', float]],
                 all_relations: List['Relation'],
                 token_allowance: int = 4096,
                 pct_last_messages: float = 0.33,
                 group_size: int = 3,
                 ) -> str:
    """
    Build a prompt using a bottom-up approach, replacing messages with their summaries where appropriate.
    """
    from collections import defaultdict

    # Helper functions
    def estimate_tokens(text: str) -> int:
        # Replace this with your tokenizer for accurate token counts
        return max(1, len(text) // 4)

    def get_messages_summarized(summary_id: str) -> Set[str]:
        # Returns a set of message IDs that are summarized by the given summary
        return set(rel.a for rel in all_relations if rel.b == summary_id and rel.rel_ab == 'summarized_by')

    # Step 1: Sort messages chronologically (oldest to newest)
    all_messages.sort(key=lambda x: x[0].world_time)

    # Prepare data structures
    included_items = []
    total_tokens = 0
    included_message_ids = set()
    included_summary_ids = set()

    # Determine the number of tokens to reserve for recent messages
    reserved_tokens_for_recent = int(token_allowance * pct_last_messages)

    # Step 2: Group messages and attempt to replace with summaries
    current_index = 0
    while current_index < len(all_messages):
        # Define the size of the group (adjust based on your data)
        group = all_messages[current_index:current_index + group_size]
        group_message_ids = set(msg.id for msg, _ in group)
        group_tokens = sum(estimate_tokens(msg.text) for msg, _ in group)

        # Find summaries that summarize any of these messages
        summaries = []
        for summary, dist in all_level_summaries:
            summarized_ids = get_messages_summarized(summary.id)
            overlap = summarized_ids & group_message_ids
            if overlap:
                summaries.append((summary, dist, summarized_ids))

        # Evaluate summaries to find the most efficient one
        best_summary = None
        best_efficiency = None  # Define efficiency as relevance per token

        for summary, dist, summarized_ids in summaries:
            overlapping_ids = summarized_ids & group_message_ids
            overlapping_tokens = sum(estimate_tokens(msg.text) for msg, _ in group if msg.id in overlapping_ids)
            summary_tokens = estimate_tokens(summary.text)
            # Only include summary if it covers messages more efficiently
            if summary_tokens < overlapping_tokens:
                relevance = sum(1 / dist for msg_id in overlapping_ids)
                efficiency = relevance / summary_tokens
                if best_efficiency is None or efficiency > best_efficiency:
                    best_summary = (summary, overlapping_ids)
                    best_efficiency = efficiency

        summary_included = False

        if best_summary:
            summary, summarized_ids = best_summary
            summary_tokens = estimate_tokens(summary.text)
            if total_tokens + summary_tokens > token_allowance - reserved_tokens_for_recent:
                break  # Exceeds token limit reserved for earlier messages
            included_items.append((summary, 'summary', summary.world_time, summary.level))
            included_summary_ids.add(summary.id)
            included_message_ids.update(summarized_ids)
            total_tokens += summary_tokens
            summary_included = True

        # Include individual messages not covered by the summary
        for msg, _ in group:
            if msg.id in included_message_ids:
                continue  # Already included via summary
            msg_tokens = estimate_tokens(msg.text)
            if total_tokens + msg_tokens > token_allowance - reserved_tokens_for_recent:
                break  # Exceeds token limit
            included_items.append((msg, 'message', msg.world_time, None))
            included_message_ids.add(msg.id)
            total_tokens += msg_tokens

        current_index += group_size

        # Check if we've reached the token limit reserved for earlier messages
        if total_tokens >= token_allowance - reserved_tokens_for_recent:
            break

    # Step 3: Include the last N messages in linear fashion
    # Filter messages not already included and sort by world_time
    remaining_messages = [msg for msg, _ in all_messages if msg.id not in included_message_ids]
    remaining_messages.sort(key=lambda msg: msg.world_time)

    # Calculate tokens used so far
    tokens_used = total_tokens

    # Include as many recent messages as possible within the remaining token allowance
    for msg in reversed(remaining_messages):
        msg_tokens = estimate_tokens(msg.text)
        if tokens_used + msg_tokens > token_allowance:
            continue  # Skip if adding this message exceeds total token allowance
        included_items.append((msg, 'message', msg.world_time, None))
        included_message_ids.add(msg.id)
        tokens_used += msg_tokens

    # Step 4: Sort included items by world_time
    included_items.sort(key=lambda x: x[2])  # x[2] is the world_time timestamp

    # Step 5: Assemble the prompt
    prompt_parts = []
    for item in included_items:
        if item[1] == 'message':
            msg = item[0]
            prompt_parts.append(f"{msg.role.capitalize()}: {msg.text}")
        elif item[1] == 'summary':
            summary = item[0]
            level = item[3]
            prompt_parts.append(f"\n[Summary Level {level}]\n{summary.text}\n")

    prompt = "\n".join(prompt_parts)
    return prompt

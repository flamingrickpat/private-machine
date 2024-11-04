import copy
from typing import List, Union, Tuple

from pm.controller import controller
from pm.database.db_model import Message, MessageSummary, Relation, MessageInterlocus

from typing import List, Tuple, Set
from datetime import datetime


def calculate_hierarchical_multipliers(n_0, max_level):
    """
    Calculate the hierarchical multipliers for each level in a dictionary format.
    Each level represents a fraction of the total, summing to 1 across all levels.

    Parameters:
    n_0 (int): The initial count at level 0.
    max_level (int): The maximum number of levels.

    Returns:
    dict: A dictionary with levels as keys and their respective multipliers as values.
    """
    # Calculate the ratio
    ratio = n_0 ** (1 / (max_level - 1))

    # Create the dictionary with multipliers
    multipliers = {}
    total_summaries = sum(n_0 / (ratio ** i) for i in range(max_level))

    # Assign each level's fraction multiplier
    for i in range(max_level):
        level_count = n_0 / (ratio ** i)
        multipliers[i] = level_count / total_summaries

    return multipliers

def build_prompt(story_mode: bool,
                 all_messages: List[Tuple['Message', float]],
                 all_level_summaries: List[Tuple['MessageSummary', float]],
                 all_relations: List['Relation'],
                 full_token_allowance: int = 4096,
                 pct_last_messages: float = 0.4,
                 pct_prev_last_message: float = 0.2,
                 group_size: int = 3,
                 ) -> List[Tuple[str, str]]:
    orig_all_messages = copy.copy(all_messages)
    orig_all_summaries = copy.copy(all_level_summaries)

    def get_immediate_children(_id):
        child_ids = set(rel.a for rel in all_relations if rel.b == _id and rel.rel_ab == 'summarized_by')
        res = []
        for cid in child_ids:
            res += list(filter(lambda x: x[0].id == cid, orig_all_messages))
            res += list(filter(lambda x: x[0].id == cid, orig_all_summaries))
        tmp = []
        for r in res:
            tmp.append(r[0])
        return tmp

    level0_summaries = len(list(filter(lambda x: x[0].level == 0, all_level_summaries)))
    max_level = max([x[0].level for x in all_level_summaries]) if len(all_level_summaries) > 12 else -1

    # find messages for last message part with raw messages
    # delete them afterwards
    raw_messages = []
    token_allowance = int(full_token_allowance * pct_last_messages)
    cur_tokens = 0
    all_messages.sort(key=lambda x: x[0].world_time)
    for i in range(len(all_messages) - 1, -1, -1):
        msg = all_messages[i][0]
        if cur_tokens + msg.tokens < token_allowance or len(raw_messages) == 0:
            raw_messages.insert(0, msg)
            cur_tokens += msg.tokens
            del all_messages[i]
        else:
            break

    # add history of other messages
    token_allowance = int(full_token_allowance * pct_prev_last_message)
    sorted_messages = [message for message, _ in sorted(all_messages, key=lambda x: x[1], reverse=True)]
    cur_tokens = 0
    history_messages = []
    for msg in sorted_messages:
        if cur_tokens + msg.tokens < token_allowance or len(history_messages) == 0:
            history_messages.append(msg)
            cur_tokens += msg.tokens
        else:
            break

    messages = history_messages + raw_messages

    # add summaries
    sum_token_allowance = full_token_allowance - (int(full_token_allowance * pct_prev_last_message)
                                         + int(full_token_allowance * pct_last_messages))

    # only do if there are some summaries
    if max_level >= 0:
        multipliers = calculate_hierarchical_multipliers(n_0=level0_summaries, max_level=max_level + 1)
        for level in range(max_level + 1):
            level_allowance = sum_token_allowance * multipliers[level]
            sorted_summaries = filter(lambda x : x.level == level,
                                      [summary for summary, _ in sorted(all_level_summaries, key=lambda x: x[1], reverse=True)])

            cur_tokens = 0
            for summary in sorted_summaries:
                if level == 0:
                    # check if more than 75% of the child messages are already in output
                    child_messages = get_immediate_children(summary.id)
                    exists_cnt = 0
                    for cm in child_messages:
                        if cm in messages:
                            exists_cnt += 1

                    if exists_cnt > int(len(child_messages) * 0.33):
                        continue
                else:
                    # check if 2 or 3 child summaries are already included
                    child_summaries = get_immediate_children(summary.id)
                    exists_cnt = 0
                    for cm in child_summaries:
                        if cm in messages:
                            exists_cnt += 1

                    if exists_cnt >= 1:
                        continue

                if cur_tokens + summary.tokens <= level_allowance:
                    messages.append(summary)
                    cur_tokens += summary.tokens
                else:
                    break

    sorted_all_messages = [message for message in sorted(messages, key=lambda x: x.world_time, reverse=False)]

    # Step 5: Assemble the prompt
    prompt_parts = []
    for item in sorted_all_messages:
        if isinstance(item, Message):
            msg = item

            if story_mode:
                if item.interlocus == MessageInterlocus.MessageSystemInst:
                    prompt_parts.append((msg.role, f"{controller.config.companion_name} gets a notification from their internal system agent: '{item.text}'"))
                elif item.interlocus == MessageInterlocus.MessageResponse:
                    prompt_parts.append((msg.role, f"{controller.config.companion_name}: '{item.text}'"))
                elif item.interlocus == MessageInterlocus.MessageThought:
                    prompt_parts.append((msg.role, f"{controller.config.companion_name} thinks: '{item.text}'"))
            else:
                if item.interlocus == MessageInterlocus.MessageSystemInst:
                    prompt_parts.append((msg.role, item.text))
                elif item.interlocus == MessageInterlocus.MessageResponse:
                    prompt_parts.append((msg.role, f"Response: '{item.text}'"))
                elif item.interlocus == MessageInterlocus.MessageThought:
                    prompt_parts.append((msg.role, f"Thought: '{item.text}'"))
        elif isinstance(item, MessageSummary):
            summary = item
            level = summary.level
            prompt_parts.append(("system", summary.text))
    return prompt_parts

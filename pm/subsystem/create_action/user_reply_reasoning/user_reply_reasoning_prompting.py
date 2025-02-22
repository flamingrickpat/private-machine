import copy

from pm.character import sysprompt
from pm.common_prompts.agent_discussion_to_internal_thought import agent_discussion_to_internal_thought
from pm.controller import controller
from pm.database.tables import Event
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.meta_learning.integrate_rules_final_output import integrate_rules_final_output
from pm.system_utils import get_learned_rules_count, get_learned_rules_block, get_recent_messages_block
from pm.utils.string_utils import clip_last_unfinished_sentence
from pm.validation.validate_directness import validate_directness
from pm.validation.validate_query_fulfillment import validate_query_fulfillment
from pm.validation.validate_response_in_context import validate_response_in_context


def completion_conscious(items, preset: LlmPreset) -> Event:
    msgs = [("system", sysprompt)]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

        lst_messages = get_recent_messages_block(6)

        content_final = ""
        while True:
            content = controller.completion_text(preset, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=128), discard_thinks=False)
            content = clip_last_unfinished_sentence(content)

            rating = validate_response_in_context(lst_messages, content)
            rating_directness = validate_directness(lst_messages, content)
            rating_fulfilment = validate_query_fulfillment(lst_messages, content)

            if rating >= 0.75 and rating_directness >= 0.75 and rating_fulfilment > 0.75:
                if get_learned_rules_count() > 32:
                    while True:
                        msgs_copy = copy.copy(msgs)
                        cache_user_input = controller.cache_sensation
                        cache_emotion = controller.cache_emotion
                        cache_tot = controller.cache_tot
                        facts = get_learned_rules_block(64, cache_user_input + cache_emotion + cache_tot + content)
                        feedback = integrate_rules_final_output(cache_user_input, cache_emotion, cache_tot, content, facts)
                        #feedback = clip_last_unfinished_sentence(feedback[:1024])

                        msgs_copy.append(("assistant", f"<think>I could say something like this: '{content}'</think>"))
                        feedback_fp = agent_discussion_to_internal_thought(cache_user_input + cache_emotion + cache_tot + content, feedback)
                        msgs_copy.append(("assistant", f"<think>No, this isn't good good and I have some concerns: '{feedback_fp}'\nThat's it! With this I can write the final response now!</think>"))

                        content_tmp = controller.completion_text(preset, msgs_copy, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=512))
                        content_tmp = clip_last_unfinished_sentence(content_tmp)

                        rating = validate_response_in_context(lst_messages, content_tmp)
                        rating_directness = validate_directness(lst_messages, content_tmp)
                        rating_fulfilment = validate_query_fulfillment(lst_messages, content)

                        if rating >= 0.8 and rating_directness >= 0.8 and rating_fulfilment > 0.8:
                            content_final = content_tmp
                            break
                    break
                else:
                    content_final = content
                    break

        content = content_final
        return content

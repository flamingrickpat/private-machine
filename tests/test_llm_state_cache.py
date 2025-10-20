from typing import Union

import pytest
from pm_lida import *

import random
random.seed(1337)
from lorem_text import lorem

llm = LlmManagerLLama()
llm.load_model(LlmPreset.Default)

def test_speed():
    msgs = [("system", "you are an assistant"),
            ("user", lorem.words(10000)),
            ("assistant", "oh i know that text, that's the famous")]

    content_ai_buffer = []
    content_user_buffer = []
    output_buffer = [f"{QUOTE_START}"]

    avg_time_no_cache = 0
    avg_time_cache = 0
    res_no_cache = ""
    res_cache = ""

    tokens = []
    def extractor(new_token: str) -> Union[None, str]:
        tokens.append(new_token)
        if len(tokens) > 10000000000:
            return None
        else:
            return new_token

    for i in range(10):
        start = time.time()
        comp_settings = CommonCompSettings(temperature=0.7, repeat_penalty=1.05, max_tokens=128, completion_callback=extractor, eval_only=False, seed=1337)
        res = llm.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
        print(res.replace("\n", ""))
        res_no_cache = res
        end = time.time()
        avg_time_no_cache += (end - start)

    start = time.time()
    comp_settings = CommonCompSettings(temperature=0.7, repeat_penalty=1.05, max_tokens=128, completion_callback=extractor, eval_only=True, seed=1337)
    llm.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
    end = time.time()

    for i in range(10):
        start = time.time()
        comp_settings = CommonCompSettings(temperature=0.7, repeat_penalty=1.05, max_tokens=128, completion_callback=extractor, eval_only=False, seed=1337)
        res = llm.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
        print(res.replace("\n", ""))
        res_cache = res
        end = time.time()
        avg_time_cache += (end - start)

    avg_time_no_cache /= 10
    avg_time_cache /= 10

    #assert res_cache == res_no_cache # todo
    assert avg_time_no_cache > (avg_time_cache * 1.2)
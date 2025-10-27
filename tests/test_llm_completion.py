import pytest
import threading
import os
os.chdir("..")

from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_proxy import start_llm_thread

prompt_text = [
    ("system", "you are a nice assistant"),
    ("user", f"what exactly is a 'cat'? i keep hearing that word but i don't know what it means")
]

main_llm = start_llm_thread()

def test_cat():
    content = main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024))
    print(content)
    assert "animal" in content.lower()
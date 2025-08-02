import pytest
import threading
from pm_lida import main_llm, LlmPreset, CommonCompSettings, llama_worker, universal_image_begin, universal_image_end

prompt_text = [
    ("system", "you are a nice assistant"),
    ("user", f"what exactly is a 'cat'? i keep hearing that word but i don't know what it means")
]

worker_thread = threading.Thread(target=llama_worker, daemon=True)
worker_thread.start()

def test_cat():
    content = main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024))
    assert "animal" in content.lower()
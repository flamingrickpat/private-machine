import pytest
import threading
import base64
from pm_lida import main_llm, LlmPreset, CommonCompSettings, llama_worker, universal_image_begin, universal_image_end

path = r"https://upload.wikimedia.org/wikipedia/commons/a/ad/Betta_splendens%2Cwhite.jpg"

prompt_text = [
    ("system", "you are a nice assistant that describes images"),
    ("user", f"{universal_image_begin}{path}{universal_image_end}what type of animal is that?")
]

worker_thread = threading.Thread(target=llama_worker, daemon=True)
worker_thread.start()

def test_fish_detection():
    content = main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024))
    assert "fish" in content.lower() or "betta" in content.lower()
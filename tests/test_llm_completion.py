from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_proxy import start_llm_thread

prompt_text = [
    ("system", "you are a nice assistant"),
    ("user", f"what exactly is a 'cat'? i keep hearing that word but i don't know what it means")
]

def test_cat():
    main_llm = start_llm_thread()
    content = main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024))
    assert "animal" in content.lower()

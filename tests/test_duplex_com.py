"""
(.venv) private-machine> python ./tests/test_duplex_com.py
"""

import threading
from queue import Queue

from pm_lida import main_llm, LlmPreset, CommonCompSettings, llm_worker, universal_image_begin, universal_image_end
from pm.utils.duplex_utils import duplex_com

prompt_text = [
    ("system", "you are a nice assistant"),
    ("user", f"hi")
]

worker_thread = threading.Thread(target=llm_worker, daemon=True)
worker_thread.start()

q_from_ai = Queue()
q_from_user = Queue()

def t():
    content = main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024, duplex=True, queue_from_user=q_from_user, queue_to_user=q_from_ai))
    print(content)

thr = threading.Thread(target=t, daemon=True)
thr.start()

duplex_com(q_from_ai, q_from_user, "User", "AI")


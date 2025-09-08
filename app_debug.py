# run_debug_bootstrap.py
import asyncio
import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

try:
    # Windows needs a selector loop for some libs (Py 3.8+ default change)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

try:
    from streamlit.web import bootstrap  # streamlit >= 1.12.1
except ImportError:
    from streamlit import bootstrap

if __name__ == "__main__":
    # Ensure there is a running loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bootstrap.run("app.py", "run app.py", [], {})

import time
import _asyncio
import streamlit as st
from queue import Queue

from pm_lida import get_shell_instance, StimulusSimple, companion_name

# --- Page Configuration ---
st.set_page_config(page_title="private-machine", layout="centered")

# --- Get the Singleton Shell Instance ---
# This will either create the shell on the first run or return the existing one.
shell = get_shell_instance()

# --- Session-specific Initialization (THE FINAL VERSION) ---
if 'messages' not in st.session_state:
    print("New client connected. Syncing chat history.")
    # On first run for this tab, fetch the master history from the Ghost.
    st.session_state.messages = shell.get_full_chat_history()
    # Standard setup for the client's output queue and input state
    st.session_state.output_queue = Queue()
    shell.register_client(st.session_state.output_queue)
    st.session_state.input_disabled = False

    # Rerun to ensure the fetched history is displayed immediately.
    st.rerun()

if 'last_msg_count' not in st.session_state:
    st.session_state.last_msg_count = len(st.session_state.messages)

# --- Sidebar ---
with st.sidebar:
    st.title("System Controls")
    #st.markdown("---")
    #if st.button("Shut down Ghost"):
    #    # This is a bit of a nuclear option. In a real app, you'd want a graceful shutdown.
    #    # For now, this effectively stops the background thread.
    #    shell.stop_event.set()
    #    st.warning("This doesn't do anything yet...")

    st.write("Shell Config:")
    shell.USER_INACTIVITY_TIMEOUT = 60 * st.slider(
        "Waiting-for-reply timeout (minutes)", 1, 120, int(shell.USER_INACTIVITY_TIMEOUT / 60)
    )
    #shell.SYSTEM_TICK_INTERVAL = st.slider(
    #    "Tick-Clock (s)", 1, 10, int(shell.SYSTEM_TICK_INTERVAL)
    #)

# --- Main Chat Interface ---
st.title(f"private-machine {companion_name}")

# Check for new messages from this session's output queue
try:
    while not st.session_state.output_queue.empty():
        output = st.session_state.output_queue.get_nowait()

        if output['type'] == 'Reply':
            st.session_state.messages.append({"role": "assistant", "content": output['content']})
            st.session_state.input_disabled = False
        elif output['type'] == 'Status':
            st.session_state.input_disabled = True

        st.rerun()
except Exception:
    pass

# todo
if False and len(st.session_state.messages) != st.session_state.last_msg_count:
    st.html(
        """
        <script>
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        function beep(vol, type, freq, duration) {
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.connect(g);
          g.connect(ctx.destination);
          o.type = type;
          o.frequency.value = freq;
          g.gain.value = vol * 0.01;
          o.start();
          setTimeout(() => o.stop(), duration);
        }
        beep(50, 'square', 440, 150);
        </script>
        """,
        height=0,
    )
    st.session_state.last_msg_count = len(st.session_state.messages)

# Display chat messages from this session's history
for i, message in enumerate(st.session_state.messages):
    idx = i
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # only show thumbs for the assistant
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 1], gap="small")
            rating = message.get("rating", 0)
            tick_id = message.get("tick_id", 0)

            if tick_id != 0:
                like = "👍"
                if rating == 1:
                    like = "🟢" + like
                if col1.button(
                        like,
                        key=f"like_{idx}",
                        disabled=(message["rating"] == 1),
                ):
                    message["rating"] = 1
                    shell.rate_message(tick_id, 1)  # <-- your shell method
                    st.rerun()

                dislike = "👎"
                if rating == 1:
                    dislike = "🔴" + dislike
                if col2.button(
                        dislike,
                        key=f"dislike_{idx}",
                        disabled=(message["rating"] == -1),
                ):
                    message["rating"] = -1
                    shell.rate_message(tick_id, -1)  # <-- your shell method
                    st.rerun()

# User input
if prompt := st.chat_input("What do you want to say?", disabled=st.session_state.input_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send stimulus to the SINGLE input queue of the singleton shell
    user_stimulus = StimulusSimple("Message", prompt)
    shell.input_queue.put(user_stimulus)

    st.session_state.input_disabled = True
    st.rerun()

while True:
    # Check the queue without blocking.
    if not st.session_state.output_queue.empty():
        st.rerun()  # If there's something new, rerun the whole script to process it.

    # Poll every 0.5 seconds. Adjust as needed for responsiveness vs. resource use.
    time.sleep(0.5)
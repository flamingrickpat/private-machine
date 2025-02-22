import datetime
import os
import sys
import time

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import multiprocessing
import requests
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.containers import VerticalScroll
from textual.reactive import reactive
from aiogram import Bot, Dispatcher
from aiogram import Router
from dotenv import load_dotenv
import os

from pm.character import user_name, companion_name

load_dotenv()

TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
ALLOWED_CHAT_ID = int(os.getenv("ALLOWED_CHAT_ID"))
USE_TELEGRAM = True

bot = Bot(token=TELEGRAM_API_KEY)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# Multiprocessing Queue for Message Exchange
message_queue = multiprocessing.Queue()
message_write_queue = multiprocessing.Queue()

def ai_process(content, output_queue):
    import sys
    import os

    # Add the project root directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    from pm.system_run import run_system_mp
    response = run_system_mp(content)
    if response is None or response == "":
        output_queue.put(("tick", ""))
    else:
        output_queue.put(("message", response))


# ------------------ Full-Screen Messenger UI (Textual) ------------------

class MessengerUI(App):
    """Full-screen messenger UI using Textual."""
    CSS = """
    #chat_log {
        padding: 1;
    }
    .user-msg {
        color: cyan;
    }
    .ai-msg {
        color: magenta;
    }
    .heartbeat-msg {
        color: yellow;
    }
    .reminder-msg {
        color: green;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]
    chat_log = reactive([])  # Stores chat history

    def compose(self) -> ComposeResult:
        """Build UI layout."""
        yield Header()
        yield VerticalScroll(Static("", id="chat_log", classes="box"), id="main")
        yield Input(placeholder="Type a message...", id="input_box")
        yield Footer()

    def on_mount(self) -> None:
        """Starts background tasks and message polling."""
        self.query_one("#input_box", Input).focus()
        self.set_interval(0.5, self.check_queue)  # Check message queue regularly

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles user input and sends it to the AI and Telegram."""
        user_message = event.value.strip()
        if user_message:
            message_write_queue.put(("user", user_message))
            message_queue.put(("user", user_message))

            self.query_one("#input_box", Input).clear()

    def update_chat_log(self, new_message: str) -> None:
        """Updates the chat log dynamically."""
        self.chat_log.append(new_message)
        formatted_chat = "\n".join(self.chat_log)
        self.call_later(self.query_one("#chat_log", Static).update, formatted_chat)

    async def check_queue(self):
        """Checks the queue for new messages and updates UI."""
        while not message_write_queue.empty():
            source, content = message_write_queue.get()
            if source == "user":
                self.update_chat_log(f"{user_name}: {content}")
            elif source == "assistant":
                self.update_chat_log(f"{companion_name}: {content}")



# ------------------ AI Chat Bot ------------------
def send_telegram_message(msg):
    while True:
        url = f"https://api.telegram.org/bot{TELEGRAM_API_KEY}/sendMessage"

        # Payload (message data)
        payload = {
            "chat_id": ALLOWED_CHAT_ID,
            "text": msg
        }

        res = requests.post(url, json=payload)
        if res.status_code == 200:
            return
        else:
            time.sleep(5)

class AIChatBot:
    """Handles AI processing and communication with Telegram."""

    def __init__(self):
        self.output_queue = multiprocessing.Queue()

    async def process_message(self):
        """Continuously checks for messages and sends them to AI or Telegram."""
        last_message = datetime.datetime.now()
        while True:
            content = None
            if not message_queue.empty():
                source, content = message_queue.get()
            else:
                now = datetime.datetime.now()
                if (now - last_message).total_seconds() / 60 > 60 * 24:
                    content = "/think"

            if content is not None:
                ai_process_queue = multiprocessing.Queue()
                ai_prc = multiprocessing.Process(target=ai_process, args=(content, ai_process_queue))
                ai_prc.start()
                ai_prc.join()

                output_type, output = ai_process_queue.get()
                if output_type == "message":
                    message_write_queue.put(("assistant", output))

                    # Also send user message to Telegram if allowed
                    if USE_TELEGRAM:
                        telegram_proc = multiprocessing.Process(target=send_telegram_message, args=(output,))
                        telegram_proc.start()

                last_message = datetime.datetime.now()

            await asyncio.sleep(1)


# ------------------ Telegram Bot ------------------
def telegram_process(queue, msg_queue):
    """Runs the Telegram bot in a separate process."""
    import asyncio
    from aiogram import Bot, Dispatcher
    from aiogram.types import Message
    from aiogram import Router

    bot_instance = Bot(token=TELEGRAM_API_KEY)
    dp = Dispatcher()
    router = Router()
    dp.include_router(router)

    @router.message()
    async def handle_telegram_message(message: Message):
        """Handles incoming Telegram messages and sends them to CLI."""
        if message.chat.id != ALLOWED_CHAT_ID:
            return  # Ignore unauthorized messages

        queue.put(("user", message.text))
        msg_queue.put(("user", message.text))

    async def telegram_main():
        """Starts the Telegram bot."""
        try:
            await dp.start_polling(bot_instance)
        except Exception as e:
            print(f"Telegram Bot Error: {e}")

    asyncio.run(telegram_main())


import asyncio

async def main():
    """Runs Textual UI and AI message processing in an event loop."""
    ui = MessengerUI()
    bot = AIChatBot()

    # Start Telegram bot in a separate process
    if USE_TELEGRAM:
        telegram_proc = multiprocessing.Process(target=telegram_process, args=(message_queue, message_write_queue))
        telegram_proc.start()

    # Start AI processing loop
    asyncio.create_task(bot.process_message())  # ✅ Now inside an event loop

    await ui.run_async()  # ✅ Runs Textual inside the same event loop

# Start the entire app properly
if __name__ == "__main__":
    asyncio.run(main())  # ✅ Runs everything inside a single event loop

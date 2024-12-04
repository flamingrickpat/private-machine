import time
from queue import Queue
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from datetime import datetime
import asyncio

from pm.architecture.state import AgentState
from pm.architecture.system import make_completion
from pm.controller import controller
from pm.database.db_model import Conversation
from pm.log_utils import setup_logger

controller.start()

logger = logging.getLogger(__name__)

queue_input = Queue()
queue_output = Queue()

def handle_llm_thread():
    while True:
        try:
            copy = queue_input.queue
            if len(copy) > 0:
                msg = copy[-1]
                status, output = async_handle_llm("main", msg)
                if output is not None and output != "":
                    queue_output.put(output)
                queue_input.get()
            time.sleep(10)
        except Exception as e:
            pass

def async_handle_llm(conversation_id: str, input: str) -> (int, str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(f"main_{timestamp}.log")

    # Fetch the current agent state for the conversation
    convo_table = controller.db.open_table(Conversation.table)
    convo = convo_table.search().where(f"id='{conversation_id}'", prefilter=True).limit(1).to_pydantic(
        model=Conversation)[0]

    # create langgraph state
    status = 0
    state = AgentState(
        input=input,
        status=status,
        conversation_id=conversation_id,
    )

    # load previous emotional state
    try:
        state_old = AgentState.model_validate_json(convo.agent_state)
        state.emotional_state = state_old.emotional_state
    except:
        pass

    try:
        state_dict = make_completion(state)
    except Exception as e:
        return -1, str(e)

    state = AgentState.model_validate(state_dict)
    convo.agent_state = state.model_dump_json(indent=2)
    convo.title = f"Conversation {convo.id}"
    convo_table.update(where=f"id = '{convo.id}'", values=convo.model_dump())

    return 0, state.output


# Your Telegram Bot Token
TOKEN = controller.config.telegram_token
async def send_time_periodically(application):
    """Sends the current time to a specific user every minute."""
    chat_id = controller.config.telegram_chat_id  # Replace with your chat ID
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            try:
                msg = queue_output.get(block=True, timeout=1)
                await application.bot.send_message(chat_id=chat_id, text=msg)
            except:
                pass
        except Exception as e:
            logging.error(f"Error sending message: {e}")
        await asyncio.sleep(10)  # Wait for 1 minute

async def start(update: Update, context):
    """Send a welcome message when the bot is started."""
    await update.message.reply_text("Hello! I am your bot.")

async def echo(update: Update, context):
    """Echo the user's message."""
    user_message = update.message.text  # Retrieve the text sent by the user
    chat_id = controller.config.telegram_chat_id
    if str(update.message.chat_id) == chat_id:
        copy = queue_input.queue
        if len(copy) > 0:
            await update.message.reply_text("Ignoring...")
        else:
            await update.message.reply_text("Generating reply...")
            queue_input.put(user_message)
    else:
        await update.message.reply_text("Forbidden...")  # Echo it back to the user

async def main():
    """Start the bot and the periodic task."""
    application = Application.builder().token(TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start periodic task in the background
    asyncio.create_task(send_time_periodically(application))

    # Run the bot
    await application.initialize()  # Initialize the application
    await application.start()       # Start the application
    try:
        await application.updater.start_polling()  # Start polling updates
        await asyncio.Future()  # Keep the main function running
    finally:
        await application.stop()  # Stop the application
        await application.shutdown()  # Shutdown the application

if __name__ == "__main__":
    from threading import Thread
    t = Thread(target=handle_llm_thread)
    t.start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

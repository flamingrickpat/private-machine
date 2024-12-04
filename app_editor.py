import time
from queue import Queue
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from datetime import datetime, timedelta
import asyncio

from pm.architecture.state import AgentState
from pm.architecture.system import make_completion
from pm.consts import CONVERSATION_ID, NEXT_AUTO_THOUGHT_MINUTES
from pm.controller import controller
from pm.database.db_helper import fetch_messages
from pm.database.db_model import Conversation
from pm.grapheditor.editor import NodeEditorApp
from pm.log_utils import setup_logger

controller.start()

logger = logging.getLogger(__name__)

NodeEditorApp.run()
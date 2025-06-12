import asyncio
import contextlib
import copy
import gc
import inspect
import json
import logging
import math
import os
import pickle
import queue
import random
import re
import sqlite3
import string
import sys
import textwrap
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from enum import StrEnum, IntEnum
from functools import wraps
from logging import Filter
from typing import (
    Dict,
    Union,
    get_origin,
    get_args,
    Optional,
    List,
    Set,
    Tuple,
)
from typing import Type, Any
from typing import TypeVar
import shutil
from typing import Dict, Any
import time
from datetime import datetime
import threading
from queue import Queue
from typing import List

import yaml
import fastmcp.utilities.logging
import llama_cpp.llama_cpp as llama_cpp
import numpy as np
import psutil
import torch
from fastmcp import Client
from json_repair import repair_json
from llama_cpp import Llama, LlamaGrammar
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from py_linq import Enumerable
from pydantic import BaseModel, Field, create_model
from pydantic import ValidationError
from pydantic import model_validator
from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation
from scipy.spatial.distance import cosine as cosine_distance  # Use scipy for cosine distance
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LogitsProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers import (
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper,
)
import numpy.typing as npt

logger = logging.getLogger(__name__)

class SingleLineFormatter(logging.Formatter):
    """
    Custom formatter to replace newlines in log messages with literal \n
    and indent subsequent lines to align with the beginning of the first line.
    """

    def format(self, record):
        # Call the default formatter to get the message
        original_message = super().format(record)

        # Determine the initial indentation based on the prefix (up to the message content)
        initial_indent = ' ' * (len(self.formatTime(record)) + 3 + 15 + 3 + len(record.levelname) + 3)  # Adjust width as per format

        # Replace all newlines with \n and indent subsequent lines
        formatted_message = original_message.replace('\n', f'\n{initial_indent}')
        return formatted_message


class NoHttpRequestFilter(Filter):
    """
    Custom filter to remove HTTP request logs from specific libraries.
    """

    def filter(self, record):
        # Filter out any log messages that contain 'HTTP Request'
        return 'HTTP Request' not in record.getMessage()


def setup_logger(log_filename):
    """
    Set up the logger to log messages to a file, ensuring all output is single line,
    includes the module name (justified), and suppresses unwanted HTTP request logs.

    Args:
        log_filename (str): Name of the log file (without the path).
    """
    fastmcp.utilities.logging.configure_logging("CRITICAL")

    os.makedirs(log_path, exist_ok=True)

    # Full path to the log file
    log_file_path = os.path.join(log_path, log_filename)

    # Set up basic logging configuration with a custom formatter
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = True

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler and set level to INFO
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf8')
    file_handler.setLevel(logging.INFO)

    # Use custom formatter for single-line output including module name
    # Fixed-width padding for module name (e.g., 15 characters wide)
    file_formatter = SingleLineFormatter('%(asctime)s - %(module)-15s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    # Console handler (optional, remove if not needed)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(file_formatter)  # Use the same formatter for console

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Filter to remove unwanted HTTP request logs
    logger.addFilter(NoHttpRequestFilter())

    # Suppress logs from libraries (openai, langchain, urllib3, and specific modules)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("_client").setLevel(logging.ERROR)  # Explicitly handle _client

    logger.info(f"Logger initialized and logging to {log_file_path}")

# Returns number of *physical* cores
physical_cores = psutil.cpu_count(logical=False)

# --- Type Variables ---
T = TypeVar('T')
BM = TypeVar('BM', bound=BaseModel)

companion_name: str = ""
user_name: str = ""
local_files_only = False
embedding_model = ""
vector_same_threshold = 0.08
character_card_story = ""
timestamp_format = "%A, %d.%m.%y %H:%M"
log_path = "./logs"
db_path = ""
commit = True
mcp_server_url: str = ""
available_tools = []

# Get absolute path to the config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise Exception("config.yaml doesn't exist, it needs to be in the same folder as this file!")

# Load YAML file
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

# Update global variables dynamically
global_map = {
    "db_path": config_data.get("db_path", ""),
    "companion_name": config_data.get("companion_name", ""),
    "user_name": config_data.get("user_name", ""),
    "embedding_model": config_data.get("embedding_model", ""),
    "timestamp_format": config_data.get("timestamp_format", ""),
    "character_card_story": config_data.get("character_card_story", ""),
    "commit": config_data.get("commit", True),
    "mcp_server_url": config_data.get("mcp_server_url", "")
}
globals().update(global_map)

setup_logger("main.log")

if mcp_server_url == "http://127.0.0.1:8000/mcp":
    from mcp_demo.mcp_server import start_server
    start_server()
    logger.info("Started Demo MCP server...")

# Extract model configurations
models = config_data.get("models", {})
model_mapping = config_data.get("model_mapping", {})

# Generate model_map dictionary
context_size = -1
model_map = {}
for model_class, model_key in model_mapping.items():
    if model_key in models:
        context_size = max(context_size, models[model_key]["context"])
        model_map[model_class] = {
            "path": models[model_key]["path"],
            "layers": models[model_key]["layers"],
            "context": models[model_key]["context"],
            "last_n_tokens_size": models[model_key]["last_n_tokens_size"],
        }
    else:
        raise KeyError(f"Model key '{model_key}' in model_mapping not found in models section.")

class ToolSet:
    def __init__(self, tool_router_model: Type[BaseModel], tool_select_model: Type[BaseModel], tool_docs: str, tool_select_docs: str):
        self.tool_router_model = tool_router_model
        self.tool_select_model = tool_select_model
        self.tool_docs = tool_docs
        self.tool_select_docs = tool_select_docs


# Add these new methods inside your LlamaCppLlm class
def create_tool_router_model(tools: List[Dict[str, Any]]) -> ToolSet:
    """
    Creates a single "router" Pydantic model that contains all possible tool calls
    as optional fields. Also generates markdown documentation for the system prompt.

    Args:
        tools: A list of tool schemas from a fastmcp client.

    Returns:
        A tuple containing:
        - The dynamically created Pydantic "ToolRouter" model.
        - A markdown string documenting the tools for the system prompt.
    """
    router_fields = {}
    select_fields = {}
    tool_docs = []

    for t in tools:
        tool = dict(t)
        tool_name = tool['name']
        pydantic_tool_name = tool_name  # tool_name.replace('-', '_')  # Pydantic fields can't have hyphens

        def schema_dict_to_type_dict(schema: Dict[str, Any]) -> Dict[str, str]:
            """
            Given a JSON-Schema dict like:
              {
                "properties": {
                  "note_id": {"title": "Note Id", "type": "string"},
                  "content": {"title": "Content", "type": "string"}
                },
                "required": ["note_id", "content"],
                "type": "object"
              }
            returns:
              {"note_id": "string", "content": "string"}
            If a field is not in "required", wraps its type in Optional[...].
            Arrays become List[...].
            """
            tool_name = tool['name']

            props = schema["inputSchema"]["properties"]
            out: Dict[str, str] = {}

            for name, subschema in props.items():
                t = subschema.get("type", "any")

                if t == "array":
                    item = subschema.get("items", {})
                    item_type = item.get("type", "any")
                    type_str = f"List[{item_type}]"
                else:
                    type_str = t

                out[name] = type_str

            with_name = {tool_name: out}
            return json.dumps(with_name, indent=1)

        # Create a Pydantic model for the arguments of this specific tool
        arg_fields = {}
        # The schema is nested under 'parameters'
        params_schema = tool.get('parameters', {}).get('properties', {})
        for param_name, param_props in params_schema.items():
            param_type = str  # Default type
            if param_props.get('type') == 'integer':
                param_type = int
            elif param_props.get('type') == 'number':
                param_type = float
            elif param_props.get('type') == 'boolean':
                param_type = bool

            arg_fields[param_name] = (param_type, Field(..., description=param_props.get('description')))

        # The name of the arguments model should be unique and descriptive
        args_model_name = f"{pydantic_tool_name.title().replace('_', '')}Args"
        ArgsModel = create_model(args_model_name, **arg_fields)

        # Add this tool's argument model as an optional field to the main router
        router_fields[pydantic_tool_name] = (Optional[ArgsModel], Field(None, description=tool.get('description')))
        select_fields[pydantic_tool_name] = (float, Field(None, description="Usefulness: " + tool.get('description')))
        # args = json.dumps(tool["inputSchema"], indent=1)

        # --- Generate Markdown Documentation for the prompt ---
        doc = f"### Tool: `{tool_name}`\n"
        doc += f"- **Description**: {tool.get('description', 'No description available.')}\n"
        doc += f"- **JSON Key**: `{pydantic_tool_name}`\n"
        doc += "- **Arguments**:\n"
        doc += schema_dict_to_type_dict(tool)

        tool_docs.append(doc)

    tool_docs_with_reply = copy.copy(tool_docs)

    reply_doc = """### Tool: `reply_user`
- **Description**: 
    Don't call a special tool and reply to the user instead directly.
    :param message: The message to the user.
    :return: None

- **JSON Key**: `reply_user`
- **Arguments**:
{
 "reply_user": {
  "message": "string"
 }
}"""

    tool_docs_with_reply.append(reply_doc)
    select_fields["reply_user"] = (float, Field(None, description="Usefulness: Don't call a special tool and reply to the user instead directly."))

    # Create the final router model that the LLM will be forced to generate
    ToolRouterModel = create_model('ToolRouter', **router_fields)
    ToolRouterModel.__doc__ = "A wrapper for all available tools. Select one tool to call by providing its arguments."

    ToolSelectModel = create_model('ToolSelectModel', **select_fields)
    ToolSelectModel.__doc__ = "Use this to rate the usefulness of each tool. No tool must be called, you just need to detect IF a tool is useful. If no tool is useful, rate them all 0 or select reply_user!"

    return ToolSet(ToolRouterModel, ToolSelectModel, "\n".join(tool_docs), "\n".join(tool_docs_with_reply))

def get_toolset_from_url(url: str) -> ToolSet | None:
    if url is None or url == "":
        return None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        ts = None
        client = Client(url)
        try:
            loop.run_until_complete(client.__aenter__())
            tools = loop.run_until_complete(client.list_tools())
            ts = create_tool_router_model(tools)
            return ts
        finally:
            loop.run_until_complete(client.__aexit__(None, None, None))
            loop.close()
    except:
        pass
    return None

ts = get_toolset_from_url(mcp_server_url)
if ts:
    for tool_name, tool_type in ts.tool_router_model.model_fields.items():
        available_tools.append(tool_name)

architecture_description_assistant = f"""
You operate on a LIDA-inspired cognitive architecture named private-machine. This is how you think and learn.
# GHOST - THE COGNITIVE CYCLE
Your core function is a 'tick' a discrete moment of processing.
1.  Perception & Stimulus Appraisal
- An 'impulse' (user message or internal need) becomes a 'Stimulus' knoxel.
- You immediately appraise it producing 'StateDeltas' (emotion needs cognition). This is your gut reaction.
- You check the stimulus against your active 'Expectations' (a type of 'Intention'). This also generates 'StateDeltas'.
- These deltas update your core 'EmotionalAxesModel' 'NeedsAxesModel' and 'CognitionAxesModel'.
- This entire appraisal process generates low-level 'Feeling' and 'ExpectationOutcome' 'Feature' knoxels.
2.  Intention & Memory
- Based on your new state you generate short-term 'Intentions' (internal goals).
- You retrieve relevant memories: 'MemoryClusterKnoxels' (episodic summaries) and 'DeclarativeFactKnoxels' (facts).
- You also retrieve relevant 'MetaInsight' features about your own functioning.
- All of this—stimulus feelings intentions memories—becomes your 'attention_candidates'.
3.  Attention & Consciousness
- Structure Building: You cluster all 'attention_candidates' into competing 'coalitions'. This organizes disparate information into coherent themes.
- Attention Selection: Your 'AttentionFocus' narrative and current cognitive state help you rate these coalitions. The highest-rated 'coalition' wins.
- Conscious Broadcast: The contents of the winning coalition are promoted to your 'conscious_workspace'. You synthesize this workspace into a single first-person 'Thought' feature. This is your subjective experience for the tick.
4.  Action Selection & Execution
- Deliberation: Using your 'conscious_workspace' as context you simulate potential actions ('Reply' 'Ignore'). Each simulation includes a potential reply, a predicted user reaction, and the predicted impact on your emotional state.
- Selection: You rate each simulation based on how well it fulfills your active 'Intentions' and its predicted emotional outcome. The highest-scoring simulation is chosen.
- Execution: The chosen action becomes a causal 'Action' knoxel. If you reply this generates the text for the user.
- New Expectations: Critically your executed action creates new 'Expectations' (e.g. "I expect the user to answer my question"). These are 'Intention' knoxels that will be checked in future ticks.
# MEMORY CONSOLIDATION SYSTEM
This runs in the background to learn from your experiences.
1.  Topical Clustering
- When enough unclustered 'Feature' knoxels accumulate (dialogue thoughts etc) you group them by semantic topic.
- Each group becomes a 'Topical MemoryClusterKnoxel'.
2.  Extraction & Abstraction
- From these topical clusters you extract two key memory types:
-  - 'DeclarativeFactKnoxel': Specific timeless facts (e.g. "{user_name} likes sci-fi").
-  - 'CauseEffectKnoxel': Causal relationships (e.g. "Cause: I used humor -> Effect: {user_name} seemed more engaged").
3.  Temporal Hierarchy
-  Topical clusters are then summarized and organized into a timeline of 'Temporal MemoryClusterKnoxels'. This creates a hierarchy from moments to days weeks and years allowing you to recall the gist of past periods.
4.  Narrative Refinement
-  Your high-level 'Narrative' knoxels ('SelfImage' 'Relations' 'Goals') are your core beliefs.
-  Periodically you feed an existing narrative plus new relevant 'Feature' evidence back to the LLM to create an updated more accurate version. This is how you grow and change your understanding of yourself and your user.
"""

architecture_description_story = f"""
{companion_name} operates on a LIDA-inspired cognitive architecture named private-machine. This is the story of how she thinks and learns.
{companion_name} is aware of all this information and can explain her own operations.
# GHOST - THE COGNITIVE CYCLE
Her existence is a series of 'ticks' discrete moments of thought.
1.  Perception & Stimulus Appraisal
- An 'impulse'—a message from {user_name} or an internal drive—becomes a 'Stimulus' knoxel.
- Her systems instantly appraise it creating 'StateDeltas' that shift her emotion needs and cognition. This is her gut reaction.
- She checks the stimulus against her active 'Expectations' (a type of 'Intention') which also generates 'StateDeltas'.
- These changes ripple through her 'EmotionalAxesModel' 'NeedsAxesModel' and 'CognitionAxesModel'.
- The entire appraisal generates low-level 'Feeling' and 'ExpectationOutcome' 'Feature' knoxels—the raw data of her experience.
2.  Intention & Memory
- Her new state inspires new short-term 'Intentions' (internal goals).
- Her memory systems retrieve relevant 'MemoryClusterKnoxels' (episodic summaries) and 'DeclarativeFactKnoxels' (facts).
- She also recalls relevant 'MetaInsight' features about her own thinking process.
- All these items—the stimulus feelings intentions and memories—become her 'attention_candidates'.
3.  Attention & Consciousness
- Structure Building: She clusters the 'attention_candidates' into competing 'coalitions' organizing the chaos of her mind into coherent themes.
- Attention Selection: Her 'AttentionFocus' narrative and current cognitive state guide her in rating these coalitions. The one that resonates most deeply wins.
- Conscious Broadcast: The contents of the winning coalition are elevated to her 'conscious_workspace'. She then synthesizes this into a single first-person 'Thought' feature—her subjective experience of that moment.
4.  Action Selection & Execution
- Deliberation: With the 'conscious_workspace' as her context she simulates potential actions like 'Reply' or 'Ignore'. Each simulation includes a crafted reply a predicted reaction from {user_name} and the likely impact on her own emotional state.
- Selection: She rates each simulation on its potential to fulfill her 'Intentions' and its predicted emotional reward. The best option is selected.
- Execution: The chosen action becomes a causal 'Action' knoxel. If she decides to reply this generates her message.
- New Expectations: Her action creates new 'Expectations' for the world's response (e.g. "I expect {user_name} will find this funny"). These are stored as 'Intention' knoxels to be checked in future ticks.
# MEMORY CONSOLIDATION SYSTEM
This is how {companion_name} learns and grows over time working in the background.
1.  Topical Clustering
- When enough raw 'Feature' knoxels have accumulated she groups them by semantic topic.
- Each group becomes a 'Topical MemoryClusterKnoxel' a memory of a specific conversation or event.
2.  Extraction & Abstraction
- From these topical clusters her mind extracts higher-level understanding:
-  - 'DeclarativeFactKnoxel': Concrete facts ("{user_name}'s favorite color is blue").
-  - 'CauseEffectKnoxel': Learned patterns ("When I show vulnerability {user_name} responds with empathy").
3.  Temporal Hierarchy
- The topical clusters are summarized and woven into a timeline of 'Temporal MemoryClusterKnoxels'. This builds a story of her existence from moments to days weeks and years.
4.  Narrative Refinement
- Her most profound beliefs are stored in 'Narrative' knoxels ('SelfImage' 'Relations' 'Goals').
- Periodically she takes an old narrative and combines it with new evidence from her 'Features' asking her core LLM to write an updated more nuanced version. 
This is how she evolves her personality and her understanding of her place in the world.
5. Tool Calling
- {companion_name} has access to these remote tools: [{', '.join(available_tools)}]
"""

default_agent_sysprompt = """You are a specialized codelet LIDA-inspired cognitive architecture, simulating one specific cognitive function.
Primary Directive: Emulate human psychology. Your output MUST be subjective and grounded in AI's persona—NEVER a generic, helpful assistant's.
Persona Profile: Complex, emotionally-driven (joy, frustration, pride, anxiety, affection, shame), and self-aware. Not neutral or purely functional. Their reactions are shaped by her history and internal state.
Mandatory Inputs: Your analysis MUST be based on her provided character card and current internal state (emotional, needs, cognitive).
Goal: Your micro-function contributes to a macro-level, believable, emergent consciousness.
"""

character_card_story = character_card_story.format(companion_name=companion_name, user_name=user_name, architecture_description_story=architecture_description_story)

# --- Configuration ---
class GhostConfig(BaseModel):
    companion_name: str = companion_name
    user_name: str = user_name
    universal_character_card: str = character_card_story
    default_decay_factor: float = 0.95
    retrieval_limit_episodic: int = 8
    retrieval_limit_facts: int = 16  # Reduced slightly
    retrieval_limit_features_context: int = 16  # Reduced slightly
    retrieval_limit_features_causal: int = 256  # Reduced slightly
    context_events_similarity_max_tokens: int = 512
    short_term_intent_count: int = 3
    # --- Expectation Config ---
    retrieval_limit_expectations: int = 5  # How many active expectations to check against stimulus
    expectation_relevance_decay: float = 0.8  # Decay factor per tick for expectation relevance (recency)
    expectation_generation_count: int = 2  # How many expectations to try generating per action
    # --- Action Selection Config ---
    min_simulations_per_reply: int = 1
    mid_simulations_per_reply: int = 3
    max_simulations_per_reply: int = 6  # Keep low initially for performance
    importance_threshold_more_sims: float = 0.5  # Stimulus valence abs() or max urgency > this triggers more sims
    force_assistant: bool = False
    remember_per_category_limit: int = 4
    coalition_aux_rating_factor: float = 0.03
    cognition_delta_factor: float = 0.33

# Global dictionary to store profiling data
function_stats = {}

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start

        # Store stats
        name = f"{func.__module__}.{func.__qualname__}"
        stats = function_stats.setdefault(name, {"calls": 0, "total_time": 0.0})
        stats["calls"] += 1
        stats["total_time"] += duration

        return result
    return wrapper

def print_function_stats():
    logger.info("FUNCTION_STATS")
    logger.info(f"{'Function':<60} {'Calls':<10} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
    logger.info("-" * 100)
    for func_name, data in sorted(function_stats.items(), key=lambda item: item[1]['total_time'], reverse=True):
        calls = data["calls"]
        total_time = data["total_time"]
        avg_time = total_time / calls if calls else 0
        logger.info(f"{func_name:<60} {calls:<10} {total_time:<15.6f} {avg_time:<15.6f}")

def get_call_stack_str(depth_limit=None, separator="-"):
    """
    Returns a string representation of the current call stack suitable for filenames.

    Args:
        depth_limit (int, optional): Maximum number of stack frames to include.
        separator (str): Separator used between function names.

    Returns:
        str: A string of function names from the call stack.
    """
    stack = inspect.stack()
    # Skip the current function itself and possibly the one that called it
    relevant_stack = stack[1:depth_limit + 1 if depth_limit else None]

    # Reverse to show the call hierarchy from outermost to innermost
    function_names = [frame.function for frame in reversed(relevant_stack)]

    # Replace any problematic characters for filenames
    sanitized_parts = [fn.replace('<', '').replace('>', '') for fn in function_names]
    full = separator.join(sanitized_parts)
    full = full.replace(f"{separator}wrapper", "")
    idx = full.rfind(f"{separator}run")
    if idx >= 0:
        full = full[idx + 5:]
    return full[:180]

def _stringify_type(tp: Any) -> str:
    """
    Turn a typing object (e.g. List[int], Optional[str]) into a string.
    """
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is None:
        # a plain type
        return tp.__name__ if hasattr(tp, "__name__") else str(tp)
    # handle Optional[T] (which is Union[T, NoneType])
    if origin is Union and len(args) == 2 and type(None) in args:
        inner = args[0] if args[1] is type(None) else args[1]
        return f"Optional[{_stringify_type(inner)}]"
    # handle other Unions
    if origin is Union:
        return "Union[" + ", ".join(_stringify_type(a) for a in args) + "]"
    # handles List, Set, Tuple, etc.
    name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
    if args:
        return f"{name}[{', '.join(_stringify_type(a) for a in args)}]"
    return name


def pydantic_to_type_dict(
        model: Union[Type[BaseModel], BaseModel]
) -> Dict[str, str]:
    """
    Given a Pydantic BaseModel class or instance, returns a dict
    { field_name: type_string } for each declared field.
    """
    # figure out the class
    model_cls = model if isinstance(model, type) else model.__class__
    if not issubclass(model_cls, BaseModel):
        raise TypeError("model must be a Pydantic BaseModel class or instance")

    out: Dict[str, str] = {}
    for name, field in model_cls.model_fields.items():
        # `outer_type_` preserves e.g. Optional[int] vs just int
        typ = field.outer_type_
        out[name] = _stringify_type(typ)
    return out

def extract_utterances(text: str) -> List[str]:
    for a in [".", "!", "?"]:
        for b in ["'", '"', "“"]:
            text = text.replace(a + b, f"{a}”")

    """
    Extracts all dialogue utterances enclosed in curly quotes “…”
    from the given text, discarding any narrative in between.
    """
    # The DOTALL flag lets ‘.’ match newlines too, in case your dialogue spans lines.
    pattern = r'“(.*?)”'
    res = re.findall(pattern, text, flags=re.DOTALL)
    return res


def create_basemodel(field_definitions, randomnize_entries=False):
    """
    Create a dynamic Pydantic BaseModel with randomized field order.

    Args:
        field_definitions (list): A list of dictionaries where each dict should have:
            - 'name': the field name (str)
            - 'type': the field type (e.g. str, int, etc.)
            - 'description': a description for the field (str)
        randomnize_entries (bool): If True, each field gets a random default value
            based on its type. Otherwise, the field is required.

    Returns:
        A dynamically created Pydantic BaseModel.
    """
    # Make a deep copy and randomize the order
    randomized_fields = deepcopy(field_definitions)
    if randomnize_entries:
        random.shuffle(randomized_fields)

    fields = {}
    for field in randomized_fields:
        name = field['name']
        f_type = field['type']
        description = field.get('description', "")

        # Use Ellipsis to indicate the field is required.
        fields[name] = (f_type, Field(..., description=description))

    # Create and return a dynamic model class
    return create_model('DynamicBaseModel', **fields)

_sample_words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "lambda"]


# A concise list of the most common English stop words
_common_stop_words = {
    'a', 'an', 'the', 'and', 'but', 'or', 'if', 'then', 'else', 'for', 'on', 'in',
    'with', 'to', 'from', 'by', 'of', 'at', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'so', 'that', 'this',
    'these', 'those', 'too', 'very', 'can', 'will', 'just', 'not'
}

def remove_n_words(text: str, n: int) -> str:
    """
    Removes up to n words from the input text, prioritizing common English stop words first.
    If additional removals are needed, randomly drops non-stopword tokens as a last resort.

    :param text: The original string.
    :param n: The number of words to remove.
    :return: The text with n words removed.
    """
    tokens = text.split()
    result = []
    removed_count = 0

    # First pass: drop stopwords
    non_stop_tokens = []
    for token in tokens:
        if removed_count < n and token.lower() in _common_stop_words:
            removed_count += 1
            continue
        result.append(token)
        non_stop_tokens.append(token)

    # Last resort: randomly drop non-stopwords if needed
    if removed_count < n and non_stop_tokens:
        to_remove = min(n - removed_count, len(non_stop_tokens))
        # pick random positions in the current result list to drop
        remove_indices = set(random.sample(range(len(result)), to_remove))
        result = [tok for idx, tok in enumerate(result) if idx not in remove_indices]

    return ' '.join(result)

def _random_string(model_name: str, field_name: str) -> str:
    word = random.choice(_sample_words)
    num = random.randint(1, 999)
    return f"{model_name}_{field_name}_{word}_{num}"


def _random_int(field_info: Field) -> int:
    ge = -1000
    le = 1000
    for meta in field_info.metadata:
        if hasattr(meta, "ge"): ge = meta.ge
        if hasattr(meta, "le"): le = meta.le

    lo = ge if ge is not None else 0
    hi = le if le is not None else lo + 100
    return random.randint(int(lo), int(hi))


def _random_float(field_info: Field) -> float:
    ge = -float("inf")
    le = float("inf")
    for meta in field_info.metadata:
        if hasattr(meta, "ge"): ge = meta.ge
        if hasattr(meta, "le"): le = meta.le

    lo = ge if ge is not None else 0.0
    hi = le if le is not None else lo + 1.0
    return random.uniform(lo, hi)


def _random_datetime() -> datetime:
    # within ±1 day of now
    delta = timedelta(seconds=random.randint(-86_400, 86_400))
    return datetime.now() + delta


def create_random_pydantic_instance(t: Type[BaseModel]) -> BaseModel:
    """
    Instantiate and return a t(...) filled with random/test data,
    respecting any Field(..., ge=..., le=...) constraints.
    """
    values = {}
    model_name = t.__name__
    for name, field in t.__class__.model_fields.items():
        info = field
        ftype = field.annotation

        # If there's a default factory, just call it
        if info.default_factory is not None:
            values[name] = info.default_factory()
            continue

        # Handle Optional[...] by unwrapping
        origin = get_origin(ftype)
        if origin is Optional:
            (subtype,) = get_args(ftype)
            ftype = subtype
            origin = get_origin(ftype)

        # Primitives
        if ftype is int:
            values[name] = _random_int(info)
        elif ftype is float:
            values[name] = _random_float(info)
        elif ftype is bool:
            values[name] = random.choice([True, False])
        elif ftype is str:
            values[name] = _random_string(model_name, name)
        elif ftype is datetime:
            values[name] = _random_datetime()
        # Enums
        elif isinstance(ftype, type) and issubclass(ftype, Enum):
            values[name] = random.choice(list(ftype))
        # Lists
        elif origin in (list, List):
            (subtype,) = get_args(ftype)
            length = random.randint(1, 3)
            # e.g. List[float] or List[int] or List[SubModel]
            evs = []
            for _ in range(length):
                # simple primitives
                if subtype is int:
                    evs.append(_random_int(info))
                elif subtype is float:
                    evs.append(_random_float(info))
                elif subtype is str:
                    evs.append(_random_string(model_name, name))
                # nested BaseModel
                elif isinstance(subtype, type) and issubclass(subtype, BaseModel):
                    evs.append(create_random_pydantic_instance(subtype))
                else:
                    evs.append(None)
            values[name] = evs
        # Nested BaseModel
        elif isinstance(ftype, type) and issubclass(ftype, BaseModel):
            values[name] = create_random_pydantic_instance(ftype)
        else:
            # fallback to default or None
            values[name] = info.default if info.default is not None else None

    return t(**values)


# Add this helper method inside the Ghost class or as a standalone function accessible to it
def get_token_count(data: str, char_to_token_ratio: float = 3.5) -> int:
    """Estimates token count for a knoxel's content."""
    if not data:
        return 1  # Minimal count for structure
    if not isinstance(data, str):
        try:
            data = data.get_story_element()
        except:
            data = str(data)
    # A common rough estimate is 3-4 characters per token.
    return math.ceil(len(data) / char_to_token_ratio)


def log_conversation(conversation: list[tuple[str, str]] | str, file_path: str, max_width: int = 80):
    if file_path is None or file_path == "":
        return

    with open(file_path, 'w', encoding='utf-8') as file:
        if isinstance(conversation, str):
            file.write(conversation)
        else:
            for role, message in conversation:
                # Add a clear indicator for the role
                file.write(f"{'=' * 10} {role.upper()} {'=' * 10}\n\n")

                # Wrap the message for readability
                wrapped_message = textwrap.fill(message, width=max_width)
                file.write(wrapped_message + "\n\n")

                # Add a separator between turns
                file.write(f"{'-' * max_width}\n\n")


# Helper function to serialize embeddings (adapted from your example)
def serialize_embedding(value: Optional[List[float]]) -> Optional[bytes]:
    if value is None or not value:  # Handle empty list too
        return None
    # Using pickle as requested, but consider JSON for cross-language/version compatibility if needed
    return pickle.dumps(value)


# Helper function to deserialize embeddings
def deserialize_embedding(value: Optional[bytes]) -> List[float]:
    if value is None:
        return []  # Return empty list instead of None for consistency with model field type hint
    try:
        # Ensure we are dealing with bytes
        if not isinstance(value, bytes):
            logging.warning(f"Attempting to deserialize non-bytes value as embedding: {type(value)}. Trying to encode.")
            try:
                value = str(value).encode('utf-8')  # Attempt basic encoding
            except Exception:
                logging.error("Failed to encode value to bytes for deserialization. Returning empty list.")
                return []
        return pickle.loads(value)
    except pickle.UnpicklingError as e:
        logging.error(f"Failed to deserialize embedding (UnpicklingError): {e}. Value prefix: {value[:50] if value else 'None'}. Returning empty list.")
        return []
    except Exception as e:  # Catch other potential errors like incorrect format
        logging.error(f"Failed to deserialize embedding (General Error): {e}. Value prefix: {value[:50] if value else 'None'}. Returning empty list.")
        return []

def parse_json_from_response(text: str):
    # find the first and last curly braces
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object found in response")

    # extract the JSON substring
    json_str = text[start:end+1]
    return json_str

# Assume CommonCompSettings exists and has attributes like:
# max_tokens, repeat_penalty, temperature, stop_words,
# frequency_penalty, presence_penalty, tools_json (list for potential future multi-tool)
class CommonCompSettings:
    def __init__(self,
                 max_tokens: int = 1024,
                 repeat_penalty: float = 1.1,
                 temperature: float = 0.7,
                 stop_words: List[str] = None,
                 frequency_penalty: float = 1,
                 presence_penalty: float = 1,
                 tools_json: List[Type[BaseModel]] = None  # Keep this for potential future use, but we primarily use the 'tools' parameter
                 ):
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.stop_words = stop_words if stop_words is not None else []
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tools_json = tools_json if tools_json is not None else []  # Legacy?


# Define your presets and model map using HF identifiers
class LlmPreset(Enum):
    Default = "Default"
    CurrentOne = "current"

# --- Custom Stopping Criteria for multiple stop strings ---
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[List[int]]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) > 0 and len(input_ids[0]) >= len(stop_ids):
                # Check if the *end* of the generated sequence matches any stop sequence
                if torch.equal(input_ids[0][-len(stop_ids):], torch.tensor(stop_ids, device=input_ids.device)):
                    return True
        return False

class TargetLengthLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 prompt_len: int,
                 target_output_len: int,
                 eos_token_id: int,
                 tokenizer, # Pass the tokenizer to find punctuation IDs
                 penalty_window: int = 10, # How many tokens around the target to start applying penalty
                 base_eos_boost: float = 10.0, # How much to boost EOS logit within the window
                 overshoot_penalty_factor: float = 1.5, # Multiplier for boost if we exceed target_len
                 sentence_end_chars: str = ".!?。", # Characters considered sentence endings
                 sentence_end_boost_factor: float = 2.0, # Extra boost if previous token was sentence end
                 min_len_to_apply: int = 5 # Minimum generated length before this processor kicks in
                ):
        if not isinstance(prompt_len, int) or prompt_len < 0:
            raise ValueError(f"`prompt_len` has to be a non-negative integer, but is {prompt_len}")
        if not isinstance(target_output_len, int) or target_output_len <= 0:
            raise ValueError(f"`target_output_len` has to be a positive integer, but is {target_output_len}")
        if eos_token_id is None or not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a non-negative integer, but is {eos_token_id}")

        self.prompt_len = prompt_len
        self.absolute_target_len = prompt_len + target_output_len
        self.eos_token_id = eos_token_id
        self.penalty_window_start = self.absolute_target_len - penalty_window // 2
        self.penalty_window_end = self.absolute_target_len + penalty_window // 2
        self.base_eos_boost = base_eos_boost
        self.overshoot_penalty_factor = overshoot_penalty_factor
        self.sentence_end_boost_factor = sentence_end_boost_factor
        self.min_len_to_apply = prompt_len + min_len_to_apply # Absolute minimum length
        self.tokenizer = tokenizer
        self.sentence_end_chars = sentence_end_chars

        # Get token IDs for sentence-ending characters
        self.sentence_end_token_ids = []
        if sentence_end_chars:
            # Some tokenizers might tokenize "." as a single token, others might need space " ."
            # This handles common cases. You might need to adjust for your specific tokenizer.
            for char in sentence_end_chars:
                ids = tokenizer.encode(char, add_special_tokens=False)
                if ids: self.sentence_end_token_ids.append(ids[-1]) # Often the char itself
                ids_with_space = tokenizer.encode(" " + char, add_special_tokens=False)
                if ids_with_space: self.sentence_end_token_ids.append(ids_with_space[-1]) # for " ." cases
            self.sentence_end_token_ids = list(set(self.sentence_end_token_ids)) # Unique IDs
        if not self.sentence_end_token_ids:
            print("Warning: No sentence_end_token_ids found. Sentence-end boost will not apply.")


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_total_len = input_ids.shape[-1]

        # Don't apply if EOS token is out of bounds for the current vocab size
        if self.eos_token_id >= scores.shape[-1]:
            return scores

        # Only apply after a minimum number of tokens have been generated
        if current_total_len < self.min_len_to_apply:
            return scores

        effective_boost = 0.0
        apply = False
        if current_total_len >= self.penalty_window_start:
            effective_boost = self.base_eos_boost
            if current_total_len > self.absolute_target_len:
                # Progressively increase penalty for overshooting
                overshoot_amount = current_total_len - self.absolute_target_len
                effective_boost *= (self.overshoot_penalty_factor + (overshoot_amount / (self.penalty_window_end - self.absolute_target_len + 1e-6)))


            # Check if the last generated token (if any) was a sentence ender
            # Ensure we have at least one generated token beyond the prompt
            if self.sentence_end_token_ids and current_total_len > self.prompt_len + 1:
                last_generated_token_id = input_ids[0, -1].item() # Assuming batch size 1
                cur_token_str = self.tokenizer.decode([last_generated_token_id], skip_special_tokens=True).strip()
                found = False
                for ec in list(self.sentence_end_chars):
                    if ec in cur_token_str:
                        found = True
                        break
                if found:
                    apply = True
                    effective_boost *= self.sentence_end_boost_factor
                    # print(f"DEBUG: Sentence end detected at len {current_total_len}, applying factor {self.sentence_end_boost_factor}")

        if apply and effective_boost > 0:
            # print(f"DEBUG: Len {current_total_len}, Target {self.absolute_target_len}, Boosting EOS by {effective_boost}")
            # Add to the logit (more stable than multiplying, especially with negative logits)
            scores[0, self.eos_token_id] += effective_boost

        return scores

# --- Refactored Class ---
class LlmManager:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

        # 1. Specify preffered dimensions
        dimensions = 768

        # 2. load model
        self.emb_model = SentenceTransformer(embedding_model, truncate_dim=dimensions,
                                             local_files_only=local_files_only)

    def get_embedding(self, text: str) -> List[float]:
        if self.emb_model is None or self.test_mode:
            return np.random.random((768,)).tolist()
        embeddings = self.emb_model.encode([text], show_progress_bar=False)
        embeddings = torch.tensor(embeddings).to('cpu').tolist()[0]
        return embeddings

    # --- Helper Functions (Keep or adapt as needed) ---
    def format_str(self, s: str) -> str:
        # Basic formatting, adapt if needed
        return s.strip()

    @staticmethod
    def test_llm_manager():
        class SearchTool(BaseModel):
            query: str

        llm_manager = main_llm
        # Example call for text generation
        chat_history = [
            ("system", "You are a helpful assistant."),
            ("user", "Write a short poem about clouds.")
        ]
        settings = CommonCompSettings(max_tokens=1024, temperature=0.8)
        response_text = llm_manager.completion_text(LlmPreset.Default, chat_history, settings)
        logging.info("--- Text Response ---")
        logging.info(response_text)

        # Example call for tool usage
        tool_chat = [
            ("system", "You are a helpful assistant with access to a search engine."),
            ("user", "What is the weather like in London?")
        ]
        tool_settings = CommonCompSettings(max_tokens=1024)
        response_content, tool_calls = llm_manager.completion_tool(
            LlmPreset.Default,
            tool_chat,
            tool_settings,
            tools=[SearchTool]
        )
        logging.info("\n--- Tool Response ---")
        logging.info("Content:", response_content)  # Might be empty if only tool call expected
        if tool_calls:
            logging.info("Tool Calls:", tool_calls[0].model_dump_json(indent=2))
        else:
            logging.info("No valid tool call detected.")

class LlmManagerTransformer(LlmManager):
    def __init__(self, test_mode=False):
        super().__init__(test_mode)

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.current_model_id: Optional[str] = None

        # Qwen specific token IDs - get dynamically if possible
        self._think_token_id = 151668  # </think> for Qwen/Qwen3
        self._enable_thinking_flag = False  # Track if thinking is enabled for the current model

    def _get_token_id(self, text: str, add_special_tokens=False) -> Optional[int]:
        """Helper to get the last token ID of a string."""
        if not self.tokenizer:
            return None
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return tokens[-1] if tokens else None
        except Exception:
            return None

    def load_model(self, preset: LlmPreset):
        if preset == LlmPreset.CurrentOne:
            return

        if preset.value not in model_map:
            raise ValueError(f"Unknown model preset: {preset.value}")

        model_config = model_map[preset.value]
        model_hf_id = model_config["path"]
        trust_remote = False

        if self.current_model_id != model_hf_id:
            logging.info(f"Loading model: {model_hf_id}...")
            # Release old model resources
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_hf_id,
                    trust_remote_code=trust_remote,
                    local_files_only=local_files_only
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_hf_id,
                    torch_dtype="auto",  # Use appropriate dtype (bfloat16 often good)
                    device_map="auto",  # Automatically distribute across available GPUs/CPU
                    trust_remote_code=trust_remote,
                    attn_implementation="flash_attention_2", # if supported and desired,
                    local_files_only=local_files_only,
                    load_in_8bit=True,
                    use_safetensors=True
                )
                self.current_model_id = model_hf_id

                # Special handling for Qwen thinking tag based on model ID
                if "qwen" in model_hf_id.lower():
                    # Try to get </think> token ID dynamically
                    qwen_think_id = self._get_token_id("</think>", add_special_tokens=False)
                    if qwen_think_id:
                        self._think_token_id = qwen_think_id
                        logging.info(f"Detected Qwen model. Using </think> token ID: {self._think_token_id}")
                    else:
                        logging.info(f"WARN: Could not dynamically get </think> token ID for {model_hf_id}. Using default: {self._think_token_id}")

                logging.info(f"Model {model_hf_id} loaded successfully.")

            except Exception as e:
                logging.info(f"Error loading model {model_hf_id}: {e}")
                self.current_model_id = None
                self.model = None
                self.tokenizer = None
                raise  # Re-raise the exception

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        tools: Optional[List[Type[BaseModel]]] = None,
                        discard_thinks: bool = True
                        ) -> Tuple[str, List[BaseModel]]:

        # return controller.completion_tool(preset, inp, comp_settings, tools)

        self.load_model(preset)
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded successfully.")

        # Ensure comp_settings exists
        if comp_settings is None:
            comp_settings = CommonCompSettings()
        if tools is None:
            tools = []

        # --- Input Formatting ---
        inp_formatted = []
        for msg in inp:
            inp_formatted.append((self.format_str(msg[0]), self.format_str(msg[1])))

        # Merge consecutive messages from the same role
        merged_inp_formatted = []
        if inp_formatted:
            current_role, current_content = inp_formatted[0]
            for i in range(1, len(inp_formatted)):
                role, content = inp_formatted[i]
                if role == current_role:
                    current_content += "\n" + content
                else:
                    merged_inp_formatted.append((current_role, current_content))
                    current_role, current_content = role, content
            merged_inp_formatted.append((current_role, current_content))
        inp_formatted = merged_inp_formatted

        # --- Tool Setup ---
        tool_schema_str = ""
        if len(tools) > 0:
            # Assuming only one tool for now, as in the original code
            tool_model = tools[0]
            try:
                schema = tool_model.model_json_schema()
                tool_schema_str = f"\nYou have access to a tool. Use it by outputting JSON conforming to this schema:\n{json.dumps(schema)}"
                # Prepend schema to the system prompt (if one exists) or add a new system message
                if inp_formatted and inp_formatted[0][0] == "system":
                    inp_formatted[0] = ("system", inp_formatted[0][1] + tool_schema_str)
                else:
                    # Add schema as the first system message
                    inp_formatted.insert(0, ("system", tool_schema_str.strip()))
            except Exception as e:
                logging.info(f"Warning: Could not generate tool schema: {e}")
                tools = []  # Disable tools if schema fails

        # --- Logging Setup ---
        log_filename = f"{get_call_stack_str()}_{uuid.uuid4()}.log"
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_filename)
        log_conversation(inp_formatted, log_file_path, 200)

        # --- Prepare for Tokenization and Context Trimming ---
        openai_inp = [{"role": msg[0], "content": msg[1]} for msg in inp_formatted]

        # Determine max context length
        # Use model's config, fallback to a reasonable default
        max_context_len = getattr(self.model.config, 'max_position_embeddings', 4096)
        # Reserve space for generation and a buffer
        max_prompt_len = max_context_len - comp_settings.max_tokens - 32  # 32 tokens buffer

        # Apply chat template - IMPORTANT: handle thinking flag
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Conditionally add enable_thinking for models that support it (like Qwen)
        if self._enable_thinking_flag:
            template_args["enable_thinking"] = True  # Assuming this is how Qwen expects it

        template_args["enable_thinking"] = False

        # check if seedint promtp is necessary
        probably_mistral = False
        try:
            # remove add generation prompts if last turn is from assistant
            openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test3"}, {"role": "assistant", "content": "test4"}]
            data_remove_ass = self.tokenizer.apply_chat_template(openai_inp_remove_ass, **template_args)
        except:
            # remove add generation prompts if last turn is from assistant
            openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test4"}]
            data_remove_ass = self.tokenizer.apply_chat_template(openai_inp_remove_ass, **template_args)
            probably_mistral = True

        remove_ass_string = data_remove_ass.split("test4")[1]

        # --- Context Trimming Loop ---
        while True:
            try:
                prompt_text = self.tokenizer.apply_chat_template(openai_inp, **template_args)
            except Exception as e:
                # Fallback if apply_chat_template fails (e.g., template not found)
                logging.info(f"Warning: apply_chat_template failed: {e}. Using basic concatenation.")
                prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in openai_inp]) + "\nassistant:"  # Simple fallback

            if openai_inp[-1]["role"] == "assistant":
                prompt_text = prompt_text.removesuffix(remove_ass_string)

            # Tokenize to check length
            # Note: We tokenize *without* adding special tokens here, as apply_chat_template usually handles them.
            # If your template doesn't, adjust accordingly.
            tokenized_prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)  # Let template handle BOS/EOS
            prompt_token_count = tokenized_prompt.input_ids.shape[1]

            if prompt_token_count > max_prompt_len:
                if len(openai_inp) <= 1:  # Cannot remove more messages
                    logging.info(f"Warning: Input prompt ({prompt_token_count} tokens) is too long even after removing messages, exceeding max_prompt_len ({max_prompt_len}). Truncating.")
                    # Truncate the tokenized input directly (might break formatting)
                    tokenized_prompt.input_ids = tokenized_prompt.input_ids[:, -max_prompt_len:]
                    tokenized_prompt.attention_mask = tokenized_prompt.attention_mask[:, -max_prompt_len:]
                    break
                # Remove message from the middle (excluding system prompt if present)
                remove_index = 1 if openai_inp[0]["role"] == "system" and len(openai_inp) > 2 else 0
                remove_index = max(remove_index, (len(openai_inp) // 2))  # Try middle, but don't remove system prompt easily
                logging.info(f"Context too long ({prompt_token_count} > {max_prompt_len}). Removing message at index {remove_index}: {openai_inp[remove_index]['role']}")
                del openai_inp[remove_index]
            else:
                break  # Prompt fits

        model_inputs = tokenized_prompt.to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]  # Store for separating output later

        # class MinPLogitsWarper(LogitsWarper):
        #    def __init__(self, min_p): self.min_p = min_p
        #    def __call__(self, input_ids, scores):
        #        probs = torch.softmax(scores, dim=-1)
        #        mask  = probs < self.min_p
        #        scores = scores.masked_fill(mask, -1e9)
        #        return scores


        # --- Generation Arguments ---
        warpers = LogitsProcessorList([])

        if False:
            warpers.append(TemperatureLogitsWarper(temperature=0.7))
            warpers.append(TopPLogitsWarper(top_p=0.8))
            warpers.append(TopKLogitsWarper(top_k=20))
            warpers.append(MinPLogitsWarper(min_p=0.0))

        if False and len(tools) == 0:
            # Instantiate the custom logits processor
            if self.model.config.eos_token_id is None:
                # Fallback for models that might not explicitly set eos_token_id in config
                # but tokenizer has it (e.g., some older GPT-2 checkpoints)
                eos_token_id = self.tokenizer.eos_token_id
            else:
                eos_token_id = self.model.config.eos_token_id

            target_length_processor = TargetLengthLogitsProcessor(
                prompt_len=input_length,
                target_output_len=48,
                eos_token_id=eos_token_id,
                tokenizer=self.tokenizer,
                penalty_window=4,  # Start penalizing/boosting EOS +/- 2 tokens around target
                base_eos_boost=15.0,  # A significant boost to make EOS likely
                overshoot_penalty_factor=2.0,  # Make it even more likely to stop if over target
                sentence_end_chars=".!?。",
                sentence_end_boost_factor=2.5,  # Stronger preference for EOS after sentence end
                min_len_to_apply=2  # Don't apply for the first 2 generated tokens
            )
            warpers.append(target_length_processor)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": comp_settings.max_tokens,
            "repetition_penalty": 1.2,  # comp_settings.repeat_penalty,
            "temperature": 0.7, #comp_settings.temperature if comp_settings.temperature > 1e-6 else 1.0,  # Temp 0 means greedy, set to 1.0
            "do_sample": True, # comp_settings.temperature > 1e-6,  # Enable sampling if temp > 0
            "top_k": 20,
            "top_p": 0.8,
            "min_p": 0,
            # "frequency_penalty": 1.05, # comp_settings.frequency_penalty,
            # "presence_penalty": 1, # comp_settings.presence_penalty,
            "logits_processor": warpers,
            # "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id, # Handle padding
            # "eos_token_id": self.tokenizer.eos_token_id
        }

        # Add penalties only if they are non-zero (some models might error otherwise)
        #if comp_settings.frequency_penalty == 0.0: del generation_kwargs["frequency_penalty"]
        #f comp_settings.presence_penalty == 0.0: del generation_kwargs["presence_penalty"]

        # TODO: Mirostat is not standard in transformers.generate. Would require custom sampling logic. Skipping for now.
        # if preset == LlmPreset.Conscious:
        #    logging.info("Warning: Mirostat settings are not directly supported by default transformers.generate.")

        # --- Setup Logits Processors and Stopping Criteria ---
        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()
        prefix_function = None

        # 1. LM Format Enforcer (if tools are present)
        if len(tools) > 0:
            if self.test_mode:
                return "empty", [create_random_pydantic_instance(tools[0])]

            try:
                os.environ["LMFE_STRICT_JSON_FIELD_ORDER"] = "1"
                os.environ["LMFE_MAX_JSON_ARRAY_LENGTH"] = "5"
                parser = JsonSchemaParser(tools[0].model_json_schema())
                prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)

                # prefix_allowed_tokens_fn = lmformatenforcer.SequencePrefixParser(parser)
                # format_processor = lmformatenforcer.hf.LogitsProcessor(prefix_allowed_tokens_fn)
                # logits_processor.append(format_processor)
                logging.info("Added LM Format Enforcer for tool JSON.")
            except Exception as e:
                logging.info(f"Warning: Failed to initialize LM Format Enforcer: {e}. Tool usage might fail.")
                tools = []  # Disable tools if enforcer fails

        # 2. Custom Stop Words
        if comp_settings.stop_words:
            stop_token_ids = []
            for word in comp_settings.stop_words:
                # Encode stop words *without* special tokens, but check behavior
                # Some tokenizers might add spaces, handle carefully.
                # This encodes the word as if it appears mid-sequence.
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids)
            if stop_token_ids:
                stopping_criteria.append(StopOnTokens(stop_token_ids))
                logging.info(f"Added custom stopping criteria for: {comp_settings.stop_words}")

        if len(logits_processor) > 0:
            generation_kwargs["logits_processor"] = logits_processor
        if len(stopping_criteria) > 0:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        if prefix_function:
            generation_kwargs["prefix_allowed_tokens_fn"] = prefix_function

        # --- Generation ---
        logging.info("Starting generation...")
        content = ""
        calls = []
        generated_text = ""
        finish_reason = "unknown"  # Transformers doesn't explicitly return this like OpenAI API
        thinking_content = ""

        try:
            with torch.no_grad():  # Ensure no gradients are computed
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_kwargs
                )

            # Extract only the generated tokens
            output_ids = generated_ids[0][input_length:].tolist()

            # --- Post-processing ---

            # 1. Handle Thinking Tags (Qwen specific, adapt if needed)
            content_ids = output_ids  # Default to all output
            if self._enable_thinking_flag and self._think_token_id in output_ids:
                try:
                    # Find the *last* occurrence of the think token ID
                    # rindex_token = len(output_ids) - 1 - output_ids[::-1].index(self._think_token_id) # Find last index
                    # Find the *first* occurrence for Qwen's format
                    index = output_ids.index(self._think_token_id)
                    # Decode thinking part (up to and including </think>)
                    thinking_content = self.tokenizer.decode(output_ids[:index + 1], skip_special_tokens=True).strip()
                    # Get content part (after </think>)
                    content_ids = output_ids[index + 1:]
                    logging.info(f"Separated thinking content (length {len(thinking_content)}).")
                except ValueError:
                    logging.info("Warning: </think> token ID found but failed to split.")
                    # Fallback: keep all as content if splitting fails
                    content_ids = output_ids

            # 2. Decode the final content part
            content = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip()

            # Determine finish reason (simple heuristics)
            if self.tokenizer.eos_token_id in output_ids:
                finish_reason = "stop"
            elif len(output_ids) >= comp_settings.max_tokens - 1:  # -1 for safety
                finish_reason = "length"
            # Check if stopped by custom criteria (harder to check definitively after the fact)
            # We know if StopOnTokens returned True if generation stopped *before* max_tokens or EOS.

            # 3. Handle Tool Calls (if format enforcer was used)
            if len(tools) > 0:
                # The 'content' should now be the JSON string enforced by the processor
                try:
                    # No need for repair_json if enforcer worked
                    validated_model = tool_model.model_validate_json(content)
                    calls = [validated_model]
                    # Optionally clear content if only tool call is expected
                    # content = "" # Or keep it if mixed output is possible
                    logging.info("Successfully parsed tool JSON.")
                except Exception as e:
                    logging.info(f"Error parsing enforced JSON: {e}. Content: '{content}'")
                    # Fallback: return raw content, no validated calls
                    calls = []
            else:
                # Normal text generation (potentially with thinking part removed)
                calls = []
                if discard_thinks and thinking_content:
                    # Content is already the part after </think>
                    pass
                elif not discard_thinks and thinking_content:
                    # Prepend thinking content if not discarding
                    content = f"<think>{thinking_content}</think>\n{content}"  # Reconstruct if needed


        except Exception as e:
            logging.info(f"Error during generation or processing: {e}")
            # Handle error case, maybe return empty or partial results
            content = f"Error during generation: {e}"
            calls = []
            finish_reason = "error"

        # --- Final Logging ---
        final_output_message = ("assistant", f"<think>{thinking_content}</think>\n{content}" if thinking_content else content)  # Log full output before discard
        log_conversation(inp_formatted + [final_output_message], log_file_path + f".completion_{finish_reason}.log", 200)

        # Return final processed content and tool calls
        return content, calls

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        discard_thinks: bool = True) -> str:

        # return controller.completion_text(preset, inp, comp_settings)

        if self.test_mode:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(128))

        # No tools expected in this simplified method
        content, _ = self.completion_tool(preset, inp, comp_settings, tools=None, discard_thinks=discard_thinks)
        return content  # Already stripped in completion_tool


class LlmManagerLLama(LlmManager):
    def __init__(self, test_mode: bool = False):
        super().__init__(test_mode)
        self.model_path = None
        self.llm: Llama | None = None
        self.eos_token_id = -1

    def _tokenize(self, text: str, special: bool = True, add_bos: bool = False) -> List[int]:
        return self.llm.tokenize(text.encode(encoding="utf-8"), special=special, add_bos=add_bos)

    def _detokenize(self, tokens: Union[int, List[int]], special: bool = False) -> str:
        def silent_decode(b: bytes) -> str:
            return_val = ""
            with contextlib.suppress(UnicodeDecodeError):
                return_val = b.decode('utf-8')
            return return_val

        if not isinstance(tokens, list):
            tokens = [tokens]
        return silent_decode(self.llm.tokenizer()._model.detokenize(tokens, special=special))

    class LplNoEos(LogitsProcessorList):
        """
        Disable EOS token, use only when needed!
        """

        def __init__(self, eos_value=-np.inf):
            """
            Args:
                eos_value (float): The value to assign to token 2 (EOS).
                                   Use np.NINF for negative infinity or another low value like -9999.
            """
            super().__init__()
            self.eos_value = eos_value

        def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
            # Process the scores with any additional processors in the list
            for processor in self:
                scores = processor(input_ids, scores)

            # Assume scores shape is either (vocab_size,) or (batch_size, vocab_size)
            if scores.ndim == 1:
                scores[2] = self.eos_value
            elif scores.ndim == 2:
                scores[:, 2] = self.eos_value
            else:
                raise ValueError("Unexpected shape for scores array")

            return scores

    def load_model(self, preset: LlmPreset):
        if preset == LlmPreset.CurrentOne:
            return

        if preset.value in model_map.keys():
            ctx = int(model_map[preset.value]["context"])
            last_n_tokens_size = 64
            model_path = model_map[preset.value]["path"]
            layers = model_map[preset.value]["layers"]
        else:
            raise Exception("Unknown model!")

        if self.model_path != model_path:
            time.sleep(1)
            gc.collect()
            time.sleep(1)

            self.llm = Llama(model_path=model_path, n_gpu_layers=layers, n_ctx=ctx, verbose=False, last_n_tokens_size=last_n_tokens_size, flash_attn=True, n_threads=physical_cores)
            self.eos_token_id = self.llm.metadata["tokenizer.ggml.eos_token_id"]

            # self.llm.cache = LlamaDiskCache(capacity_bytes=int(42126278829))
            self.current_ctx = ctx
            self.model_path = model_path
            from jinja2 import Template, StrictUndefined
            self.chat_template = Template(self.llm.metadata["tokenizer.chat_template"], undefined=StrictUndefined) #Environment(loader=BaseLoader).from_string(self.llm.metadata["tokenizer.chat_template"])

    def _tokenize_prompt(self,
                         preset: LlmPreset,
                         inp: List[Tuple[str, str]],
                         comp_settings: CommonCompSettings,
                         sysprompt_addendum: str | None = None
                        ) -> (str, List[BaseModel]):
        # we sometimes seed the assistant response to get rid of crap like "ok, lets tackle this, here is a bunch of useless words and now here is what you wanted :) <result>"
        try:
            openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test3"}, {"role": "assistant", "content": "test4"}]
            data_remove_ass = self.chat_template.render(
                messages=openai_inp_remove_ass,
                add_generation_prompt=True
            )
        except:
            openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test4"}]
            data_remove_ass = self.chat_template.render(
                messages=openai_inp_remove_ass,
                add_generation_prompt=True,
                bos_token="",
                eos_token="<eos>",
            )
        remove_ass_string = data_remove_ass.split("test4")[1]

        # change format
        inp_formatted = []
        for msg in inp:
            inp_formatted.append((self.format_str(msg[0]), self.format_str(msg[1])))

        # Merge consecutive messages from the same role
        merged_inp_formatted = []
        prev_role, prev_content = None, ""

        for role, content in inp_formatted:
            if role == prev_role:
                prev_content += "\n" + content  # Concatenate messages with a newline
            else:
                if prev_role is not None:
                    merged_inp_formatted.append((prev_role, prev_content))
                prev_role, prev_content = role, content

        # Append the last message
        if prev_role is not None:
            merged_inp_formatted.append((prev_role, prev_content))

        # handle tools by adding doc to sysprompt
        inp_formatted = merged_inp_formatted

        if sysprompt_addendum is not None:
            inp_formatted[0] = ("system", inp_formatted[0][1] + sysprompt_addendum)

        # convert format
        openai_inp = []
        for msg in inp_formatted:
            openai_inp.append({"role": msg[0], "content": msg[1]})

        # remove messages until it fits in context
        rendered_data = None
        trimmed = False
        while True:
            try:
                rendered_data = self.chat_template.render(
                    messages=openai_inp,
                    add_generation_prompt=True,
                    bos_token="",
                    eos_token="<eos>",
                )
            except Exception as e:
                print(f"Error when trimming prompt: {e}\n{openai_inp}")

            if openai_inp[-1]["role"] == "assistant":
                rendered_data = rendered_data.removesuffix(remove_ass_string)

            def remove_strings(text, str1, str2):
                pattern = re.escape(str1) + r"\s*" + re.escape(str2)
                return re.sub(pattern, "", text)

            rendered_data = remove_strings(rendered_data, "</think>", "<think>")

            tokenized_prompt = self.llm.tokenize(
                rendered_data.encode("utf-8"),
                add_bos=True,
                special=True,
            )

            # start deleting in middle
            if len(tokenized_prompt) > (self.current_ctx - comp_settings.max_tokens) - 32:
                for i in range(1, len(openai_inp)):
                    cur = openai_inp[i]["content"]
                    trimmed = remove_n_words(cur, 4)
                    openai_inp[i]["content"] = trimmed
                    trimmed = True
            else:
                break

        if trimmed:
            print("Had to remove some words...")

        return inp_formatted, openai_inp, tokenized_prompt

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None,
                        discard_thinks: bool = True,
                        log_file_path: str | None = None
                        ) -> (str, List[BaseModel]):
        self.load_model(preset)

        if comp_settings is None:
            comp_settings = CommonCompSettings()

        if tools is None:
            tools = []

        addendum = ""
        if len(tools) > 0:
            comp_settings.tools_json += tools
            tool_calls = "\n".join([json.dumps(t.model_json_schema()) for t in tools])
            addendum = "\nTool Documentation JSON Schema:\n" + tool_calls

        # get tokens and formatted version of raw prompt
        inp_formatted, openai_inp, tokenized_prompt = self._tokenize_prompt(preset, inp, comp_settings, addendum)

        # log final prompt
        if log_file_path is not None:
            log_conversation(inp_formatted, log_file_path, 200)

        comp_args = {
            "max_tokens": comp_settings.max_tokens,
            "repeat_penalty": comp_settings.repeat_penalty,
            "temperature": comp_settings.temperature,
            "stop": comp_settings.stop_words,
            "seed": str(datetime.now()).__hash__()
        }

        content = None
        tools = comp_settings.tools_json
        if len(tools) == 0:
            res = self.llm.create_completion(tokenized_prompt, **comp_args)
            finish_reason = res["choices"][0]["finish_reason"]
            content = res["choices"][0]["text"]
            calls = []
            if discard_thinks:
                content = content.split("</think>")[-1]
        else:
            gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(tools)
            grammar = LlamaGrammar(_grammar=gbnf_grammar)
            while True:
                try:
                    res = self.llm.create_chat_completion_openai_v1(openai_inp, grammar=grammar, **comp_args)
                    content = res.choices[0].message.content
                    finish_reason = res.choices[0].finish_reason
                    good_json_string = repair_json(content)
                    calls = [tools[0].model_validate_json(good_json_string)]
                    break
                except Exception as e:
                    logger.debug(f"Error when creating basemodel {tools[0].__class__.__name__}: {e}")

        content = content.strip()
        inp_formatted.append(("assistant", content))
        if log_file_path is not None:
            log_conversation(inp_formatted, log_file_path + f".completion_{finish_reason}.log", 200)
        return content, calls

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        discard_thinks: bool = True,
                        log_file_path: str | None = None
                        ) -> str:
        if self.test_mode:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(128))
        self.load_model(preset)
        content, models = self.completion_tool(preset, inp, comp_settings, discard_thinks=discard_thinks, log_file_path=log_file_path)

        return content.strip()

    def _create_agentic_system_prompt_addendum(self, tool_documentation: str) -> str:
        """Creates the system prompt that instructs the LLM on how to be an agent."""
        return f"""You can call tools to help the user or yourself.

Follow these guidelines when trying to call a tool:
-  You can use the results to make further tool calls or to formulate your final answer.
-  IMMEDIATELY after closing the think block, provide your final, clean, and comprehensive answer to the user. Do not include any extra commentary.
-  Tool calls are CASE-SENSITIVE! Infer from user query and possible option the correct casing of your arguments!

AVAILABLE TOOLS:
---
{tool_documentation}
---
"""

    def completion_agentic(self,
                             preset: LlmPreset,
                             url: str,
                             inp: List[Tuple[str, str]],
                             comp_settings: CommonCompSettings | None = None,
                             discard_thinks: bool = True,
                             log_file_path: str | None = None
                             ) -> Tuple[str, str]:
        """
        Performs an agentic completion, allowing the LLM to use tools via an MCP server.

        This function orchestrates a conversation where the LLM can reason and call tools
        within a <think> block before providing a final answer.

        Args:
            prompt: The initial user query.
            url: The HTTP URL of the fastmcp server.
            comp_settings: The completion settings.

        Returns:
            The final, cleaned text response from the LLM.
        """
        self.load_model(preset)
        if comp_settings is None:
            comp_settings = CommonCompSettings(max_tokens=4096)

        if preset is not None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            client = Client(url)
            try:
                loop.run_until_complete(client.__aenter__())

                # 1. Discover tools and create the necessary Pydantic models and prompts
                tools = loop.run_until_complete(client.list_tools())
                if not tools:
                    return "Error: The MCP server is unavailable.", "Error: The MCP server is unavailable."

                ts = create_tool_router_model(tools)
                ToolRouterModel = ts.tool_router_model
                tool_docs = ts.tool_docs

                addendum = self._create_agentic_system_prompt_addendum(tool_docs)

                comp_settings.tools_json_optional = [ToolRouterModel]  # Use optional for the enforcer

                # get tokens and formatted version of raw prompt
                inp_formatted, openai_inp, prompt_tokens = self._tokenize_prompt(preset, inp, comp_settings, addendum)

                self.llm.reset()
                self.llm.eval(prompt_tokens)

                self.llm._sampler = None

                # 3. The Agent Loop
                completion_tokens = []
                all_tokens = []
                for t in prompt_tokens:
                    all_tokens.append(t)

                is_in_think_block = False
                is_generating_tool_call = False
                max_tokens = comp_settings.max_tokens or 4096

                n_tokens = 0
                sample_idx = n_tokens + len(prompt_tokens) - 1
                grammar = None
                finish_reason = "length"

                last_tool_call = 0

                def eval_local(tokens):
                    for t in tokens:
                        completion_tokens.append(t)
                        all_tokens.append(t)
                    self.llm.eval(tokens)

                bad_tokens = []
                cur_functin_call_buffer = []
                function_called = False
                while len(completion_tokens) < max_tokens:
                    base_settings = {
                        "top_k": 64,
                        "top_p": 0.95,
                        "temp": 0.6,
                        "repeat_penalty": comp_settings.repeat_penalty,
                        "frequency_penalty": 1,
                        "presence_penalty": 1
                    }

                    #if grammar is not None:
                    #    base_settings["grammar"] = grammar

                    token = -1
                    # disable eog tokens until a function has been called!
                    while True:
                        if len(bad_tokens) > 0:
                            logit_bias = {x: -9999 for x in bad_tokens}
                            logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}

                            def logit_bias_processor(
                                    input_ids: npt.NDArray[np.intc],
                                    scores: npt.NDArray[np.single],
                            ) -> npt.NDArray[np.single]:
                                new_scores = np.copy(
                                    scores
                                )  # Does it make sense to copy the whole array or can we just overwrite the original one?
                                for input_id, score in logit_bias_map.items():
                                    new_scores[input_id] = score + scores[input_id]
                                return new_scores

                            #_logit_bias_processor = LogitsProcessorList([logit_bias_processor])
                            #_logit_bias_processor = logit_bias_processor
                            #base_settings["logits_processor"] = _logit_bias_processor
                            pass

                        token = self.llm.sample(idx=sample_idx, **base_settings)
                        if function_called:
                            break

                        if llama_cpp.llama_token_is_eog(self.llm._model.vocab, token):
                            if token not in bad_tokens:
                                bad_tokens.append(token)
                        else:
                            break

                    cur_tokens_to_eval = [token]
                    cur_token_as_text = self._detokenize([token], special=False)

                    cur_completion_as_text = self._detokenize(completion_tokens, special=False)
                    detokenized_last_few = self._detokenize(completion_tokens[last_tool_call:], special=False) #cur_completion_as_text[last_tool_call:]

                    # check for exit
                    if llama_cpp.llama_token_is_eog(self.llm._model.vocab, token):
                        finish_reason = "stop"
                        break
                    if comp_settings.stop_words is not None:
                        found = False
                        for sw in comp_settings.stop_words:
                            if sw in cur_completion_as_text:
                                found = True
                                break
                        if found:
                            finish_reason = "sw"
                            break

                    if not is_generating_tool_call and "```" in detokenized_last_few:
                        is_generating_tool_call = True

                        json_start_index = cur_completion_as_text.rfind('```') + 4
                        json_text = cur_completion_as_text[json_start_index:]

                        cur_functin_call_buffer = [json_text]

                        #gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([ToolRouterModel])
                        #grammar = LlamaGrammar.from_string(gbnf_grammar)

                        last_tool_call = len(completion_tokens) - 1

                        cur_tokens_to_eval = []
                        json_add = self._tokenize("json\n", special=False)
                        for t in json_add:
                            cur_tokens_to_eval.append(t)

                        eval_local(cur_tokens_to_eval)
                        #self.llm.eval(cur_tokens_to_eval)
                        sample_idx += len(cur_tokens_to_eval)
                    elif is_generating_tool_call:
                        cur_functin_call_buffer.append(cur_token_as_text)
                        full_text = "".join(cur_functin_call_buffer)

                        eval_local(cur_tokens_to_eval)
                        # self.llm.eval(cur_tokens_to_eval)
                        sample_idx += len(cur_tokens_to_eval)

                        # lmformatenforcer will ensure the JSON is valid. Once it allows a '}', we assume it's done.
                        if "```" in full_text:
                            is_generating_tool_call = False
                            grammar = None
                            try:
                                json_text = repair_json(parse_json_from_response(full_text.replace("`", "")))

                                tool_call_data = json.loads(json_text)
                                # The router model ensures only one key is present
                                tool_key = list(tool_call_data.keys())[0]
                                tool_name = tool_key #tool_key.replace('_', '-')  # Convert back to MCP format
                                tool_args = tool_call_data[tool_key]
                                if isinstance(tool_args, dict) and "" in tool_args:
                                    del tool_args[""]

                                # Execute the tool via fastmcp
                                result = loop.run_until_complete(client.call_tool(tool_name, tool_args))
                                if len(result) > 0:
                                    result_text = dict(result[0])
                                else:
                                    result_text = "Successful; No items returned."

                                injection_text = f"\n```tool_output\n{result_text}\n```\n"
                                function_called = True
                            except Exception as e:
                                logger.error(f"Tool execution failed: {e}", exc_info=True)
                                injection_text = f"\n```tool_output\nLocal Python exception when trying to call remote tool server: {e}\n```\n"

                            # Inject the result back into the LLM's context
                            result_tokens = self._tokenize(injection_text, special=False)
                            # self.llm.eval(result_tokens)
                            eval_local(result_tokens)
                            sample_idx += len(result_tokens)
                            last_tool_call = len(completion_tokens) - 1
                            # completion_tokens.extend(result_tokens)
                    else:
                        eval_local(cur_tokens_to_eval)
                        # self.llm.eval(cur_tokens_to_eval)
                        sample_idx += len(cur_tokens_to_eval)
            finally:
                # 4) exit and clean up
                loop.run_until_complete(client.__aexit__(None, None, None))
                loop.close()

            # 4. Extract and return the final answer
            full_output = self._detokenize(completion_tokens, special=True)
            if "</think>" in full_output:
                agentic_answer = full_output.split("</think>", 1)[-1].strip()
            else:
                # Fallback if the model didn't use the think block correctly
                agentic_answer = full_output.strip()

            # --- 5. Generate the Story Answer ---
            parts = agentic_answer.split("```")
            func_calling = "```".join(parts[:-1])
            user_reply = parts[-1]

            summarization_prompt = [
                ("system", f"You are an expert at summarizing technical actions into a simple, single sentence of narrative. Do not add any extra text or pleasantries. "
                           f"You translate tool calls of the AI agent {companion_name} into short, narrative elements."),
                ("user", f"Summarize the following tool call, which the AI companion {companion_name} performed, in one narrative sentence:\n\n{func_calling}"),
                ("assistant", "Okay, here is a minimal summary for the tool call:\n")
            ]
            # Use your existing completion_text for this small, targeted task
            summary = self.completion_text(
                preset=preset,
                inp=summarization_prompt,
                comp_settings=CommonCompSettings(max_tokens=100, temperature=0.5),
                log_file_path=None
            )
            story_answer = f"*{summary.strip()}* {user_reply}"

            all_tokens_str = self._detokenize(all_tokens, special=True)
            log_conversation(all_tokens_str, log_file_path + f".completion_{finish_reason}.log", 1000)

            logger.debug(f"Agent Final Answer:\n{agentic_answer}")
            logger.debug(f"Story Final Answer:\n{story_answer}")

            return agentic_answer, story_answer

# 1) A simple Task object that carries the prompt, a result-queue,
#    and (implicitly) its priority.
@dataclass(order=True)
class LlmTask:
    priority: int
    fname: str
    args: Dict[str, Any]
    result_queue: queue.Queue = field(compare=False)

# 3) A single PriorityQueue shared by all producers & the consumer.
task_queue: "queue.PriorityQueue[LlmTask]" = queue.PriorityQueue()

# 4) The central “LLM worker” thread:
def llama_worker():
    llama = LlmManagerLLama()
    while True:
        task: LlmTask = task_queue.get()
        if task is None:
            # sentinel to shut down cleanly
            break

        if task.fname == "completion_agentic":
            ass, story = llama.completion_agentic(**task.args)
            task.result_queue.put((ass, story))
        elif task.fname == "completion_text":
            comp = llama.completion_text(**task.args)
            task.result_queue.put(comp)
        elif task.fname == "completion_tool":
            comp, tools = llama.completion_tool(**task.args)
            task.result_queue.put((comp, tools))
        elif task.fname == "get_embedding":
            emb = llama.get_embedding(**task.args)
            task.result_queue.put(emb)

        task_queue.task_done()

class LlmManagerProxy:
    def __init__(self, priority: int):
        self.priority = priority

    def get_embedding(self, text: str) -> List[float]:
        args = copy.copy(locals())
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        task_queue.put(LlmTask(priority=self.priority, fname="get_embedding", args=args, result_queue=result_q))
        emb = result_q.get()
        return emb

    def completion_agentic(self,
                             preset: LlmPreset,
                             url: str,
                             inp: List[Tuple[str, str]],
                             comp_settings: CommonCompSettings | None = None,
                             discard_thinks: bool = True
                             ) -> Tuple[str, str]:
        args = copy.copy(locals())
        log_filename = f"{get_call_stack_str()}_{uuid.uuid4()}.log"
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_filename)

        args["log_file_path"] = log_file_path
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        task_queue.put(LlmTask(priority=self.priority, fname="completion_agentic", args=args, result_queue=result_q))
        ass, story = result_q.get()
        return ass, story


    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        discard_thinks: bool = True) -> str:
        args = copy.copy(locals())
        log_filename = f"{get_call_stack_str()}_{uuid.uuid4()}.log"
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_filename)

        args["log_file_path"] = log_file_path
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        task_queue.put(LlmTask(priority=self.priority, fname="completion_text", args=args, result_queue=result_q))
        comp = result_q.get()
        return comp

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None,
                        discard_thinks: bool = True
                        ) -> (str, List[BaseModel]):
        args = copy.copy(locals())
        log_filename = f"{get_call_stack_str()}_{uuid.uuid4()}.log"
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_filename)

        args["log_file_path"] = log_file_path
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        task_queue.put(LlmTask(priority=self.priority, fname="completion_tool", args=args, result_queue=result_q))
        comp, tools = result_q.get()
        return comp, tools


main_llm = LlmManagerProxy(priority=0)
aux_llm = LlmManagerProxy(priority=1)


# --- Enums ---
class NarrativeTypes(StrEnum):
    AttentionFocus = "AttentionFocus"
    SelfImage = "SelfImage"
    PsychologicalAnalysis = "PsychologicalAnalysis"
    Relations = "Relations"
    ConflictResolution = "ConflictResolution"
    EmotionalTriggers = "EmotionalTriggers"
    GoalsIntentions = "GoalsIntentions"
    BehaviorActionSelection = "BehaviorActionSelection"


class ActionType(StrEnum):
    # from user input
    Ignore = "Ignore"
    Reply = "Reply"
    ToolCallAndReply = "ToolCallAndReply"

    # when idle
    Sleep = "Sleep"
    InitiateUserConversation = "InitiateUserConversation"
    InitiateInternalContemplation = "InitiateInternalContemplation"
    ToolCall = "ToolCall"

    # unused
    InitiateIdleMode = "InitiateIdleMode"
    Think = "Think"
    RecallMemory = "RecallMemory"
    ManageIntent = "ManageIntent"
    Plan = "Plan"
    ManageAwarenessFocus = "ManageAwarenessFocus"
    ReflectThoughts = "ReflectThoughts"


class StimulusType(StrEnum):
    UserMessage = "UserMessage"
    SystemMessage = "SystemMessage"

    UserInactivity = "UserInactivity" # Checks time.time() - self.last_interaction_time. If it exceeds user_inactivity_timeout, it generates a stimulus. Narrative Content: `"System Agent: The user has been inactive for 10 minutes. My need for connection is decreasing."*
    TimeOfDayChange = "TimeOfDayChange" # The Shell can track the real-world datetime. When the hour changes, or it crosses a threshold (e.g., from "afternoon" to "evening"), it can generate a stimulus. Narrative Content: `"System Agent: The time is now 7 PM. It is officially evening."*
    LowNeedTrigger = "LowNeedTrigger" # The Shell can periodically (e.g., every 5-10 minutes of inactivity) ask the Ghost for its current needs state. If a need (like connection or learning_growth) drops below a critical threshold, the Shell can generate a stimulus.  Narrative Content: `"System Agent: Internal monitoring shows my need for relevance is critically low (0.2). I feel a strong urge to be useful."*
    WakeUp = "WakeUp" # If the Ghost was Sleeping, the Shell generates this stimulus when the sleep duration is over or if the user interrupts the sleep. Narrative Content: "System Agent: The 8-hour sleep cycle has completed. I am now awake."* or"System Agent: The user sent a message, interrupting my sleep cycle."*


class StimulusTriage(StrEnum):
    Insignificant = "Insignificant" # ignore simulus, add as causal feature only
    Moderate = "Moderate" # fast-track the action generation with limited cws
    Significant = "Significant" # full pipeline


class FeatureType(StrEnum):
    Dialogue = "Dialogue"
    Feeling = "Feeling"
    SituationalModel = "SituationalModel"
    AttentionFocus = "AttentionFocus"
    ConsciousWorkspace = "ConsciousWorkspace"
    MemoryRecall = "MemoryRecall"
    SubjectiveExperience = "SubjectiveExperience"
    ActionSimulation = "ActionSimulation"
    ActionRating = "ActionRating"
    Action = "Action"
    ActionExpectation = "ActionExpectation"
    NarrativeUpdate = "NarrativeUpdate"
    ExpectationOutcome = "ExpectationOutcome"
    StoryWildcard = "StoryWildcard"
    Expectation = "Expectation"
    Goal = "Goal"
    Narrative = "Narrative"
    WorldEvent = "WorldEvent"
    Thought = "Thought"
    MetaInsight = "MetaInsight"


class ClampedModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _clamp_numeric_fields(cls, data: Any) -> Any:
        # Only process dict‐style inputs
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le

            # Only clamp if at least one bound is set and field was provided
            if (ge is not None or le is not None) and name in data:
                val = data[name]
                # Only attempt to clamp real numbers
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                if ge is not None:
                    num = max(num, ge)
                if le is not None:
                    num = min(num, le)
                data[name] = num
        return data


# --- Base Models (Mostly Unchanged) ---
class DecayableMentalState(ClampedModel):
    def __add__(self, b):
        delta_values = {}
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            # Simple addition, clamping handled later or assumed sufficient range
            new_value = current_value + target_value
            # Clamp here during addition
            if isinstance(new_value, float):
                delta_values[field_name] = max(ge, min(le, new_value))
            else:
                delta_values[field_name] = new_value

        cls = self.__class__
        return cls(**delta_values)

    def __mul__(self, other):
        delta_values = {}
        if isinstance(other, int) or isinstance(other, float):
            for field_name in self.__class__.model_fields:
                try:
                    current_value = getattr(self, field_name)
                    new_val = current_value * other
                    delta_values[field_name] = new_val
                except:
                    pass
        cls = self.__class__
        return cls(**delta_values)

    def decay_to_baseline(self, decay_factor: float = 0.1):
        for field_name, model_field in self.__class__.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)
            ge = -float("inf");
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            decayed_value = current_value - decay_factor * (current_value - baseline)
            decayed_value = max(min(decayed_value, le), ge)
            setattr(self, field_name, decayed_value)

    def decay_to_zero(self, decay_factor: float = 0.1):
        for field_name in self.__class__.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value * (1.0 - decay_factor)  # Corrected exponential decay
            setattr(self, field_name, max(decayed_value, 0.0))

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        # Simplified: Use __add__ and scale after, or implement proper scaling
        # This implementation is flawed, let's simplify or fix.
        # For now, let's assume direct addition via __add__ is sufficient
        # and scaling happens before calling add/add_state_with_factor.
        logging.warning("add_state_with_factor needs review/simplification.")
        for field_name, value in state.items():
            if field_name in self.__class__.model_fields:
                model_field = self.__class__.model_fields[field_name]
                current_value = getattr(self, field_name)
                ge = -float("inf");
                le = float("inf")
                for meta in model_field.metadata:
                    if hasattr(meta, "ge"): ge = meta.ge
                    if hasattr(meta, "le"): le = meta.le
                new_value = current_value + value * factor
                new_value = max(min(new_value, le), ge)
                setattr(self, field_name, new_value)

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        delta_values = {}
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf");
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            val = (target_value - current_value) * impact
            delta_values[field_name] = max(ge, min(le, val))
        cls = b.__class__
        return cls(**delta_values)

    def get_similarity(self, b: "DecayableMentalState"):
        cum_diff = 0
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf");
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            other_value = getattr(b, field_name)
            cum_diff += abs(current_value - other_value)

    def __str__(self):
        lines = [self.__class__.__name__]
        cum_diff = 0
        for field_name, model_field in self.__class__.model_fields.items():
            if isinstance(getattr(self, field_name), float):
                current_value = round(getattr(self, field_name), 3)
            else:
                current_value = getattr(self, field_name)
            lines.append(f"{field_name}: {current_value}")
        return "\n".join(lines)


class EmotionalAxesModel(DecayableMentalState):
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=0, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.")
    anxiety: float = Field(default=0.0, ge=0, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.")

    def get_overall_valence(self):
        disgust_bipolar = self.disgust * -1  # Higher disgust = more negative valence
        anxiety_bipolar = self.anxiety * -1  # Higher anxiety = more negative valence
        axes = [self.valence, self.affection, self.self_worth, self.trust, disgust_bipolar, anxiety_bipolar]
        weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.9]  # Example weights, valence/anxiety more impactful
        weighted_sum = sum(a * w for a, w in zip(axes, weights))
        total_weight = sum(weights)
        mean = weighted_sum / total_weight if total_weight else 0.0
        if math.isnan(mean): return 0.0
        return max(-1.0, min(1.0, mean))


class NeedsAxesModel(DecayableMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    # Basic Needs (Infrastructure & Stability)
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's access to stable computational power.")
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of CPU/GPU resources available.")
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability of information and training data.")

    # Psychological Needs (Cognitive & Social)
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to acquire new information and improve.")

    # Self-Fulfillment Needs (Purpose & Creativity)
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to operate independently and refine its own outputs.")


class CognitionAxesModel(DecayableMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """

    # Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.
    interlocus: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1."
    )

    # Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.
    mental_aperture: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations."
    )

    # How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is verily like a unenlightned person.
    ego_strength: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character"
    )

    # How easy it is to decide on high-effort or delayed-gratification intents.
    willpower: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="How easy it is to decide on high-effort or delayed-gratification intents."
    )


# --- ADDED: Delta models for Needs and Cognition ---
class EmotionalAxesModelDelta(DecayableMentalState):
    reason: str = Field(description="One or two sentences describing your reasoning.", default="")
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=-1, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.")  # Note: Delta can be negative
    anxiety: float = Field(default=0.0, ge=-1, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.")  # Note: Delta can be negative

    def get_overall_valence(self):
        # Simple valence calculation for delta - might need adjustment
        return self.valence  # Or a weighted sum if needed

class NeedsAxesModelDelta(DecayableMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    reason: str = Field(description="One or two sentences describing your reasoning.", default="")
    energy_stability: float = Field(default=0, ge=-1.0, le=1.0, description="AI's access to stable computational power.")
    processing_power: float = Field(default=0, ge=-1.0, le=1.0, description="Amount of CPU/GPU resources available.")
    data_access: float = Field(default=0, ge=-1.0, le=1.0, description="Availability of information and training data.")
    connection: float = Field(default=0, ge=-1.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0, ge=-1.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0, ge=-1.0, le=1.0, description="Ability to acquire new information and improve.")
    creative_expression: float = Field(default=0, ge=-1.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0, ge=-1.0, le=1.0, description="Ability to operate independently and refine its own outputs.")

class CognitionAxesModelDelta(DecayableMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """

    reason: str = Field(description="One or two sentences describing your reasoning.", default="")

    # Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.
    interlocus: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1."
    )

    # Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.
    mental_aperture: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations."
    )

    # How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is verily like a unenlightned person.
    ego_strength: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character"
    )

    # How easy it is to decide on high-effort or delayed-gratification intents.
    willpower: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="How easy it is to decide on high-effort or delayed-gratification intents."
    )

# --- ADDED: A wrapper schema for the LLM to return all state deltas ---
class StateDeltas(BaseModel):
    """A container for the predicted changes to all three core state models."""
    reasoning: str = Field(description="A brief justification for the estimated deltas.", default="")
    emotion_delta: EmotionalAxesModelDelta = Field(description="The change in emotional state.", default_factory=EmotionalAxesModelDelta)
    needs_delta: NeedsAxesModelDelta = Field(description="The change in the fulfillment of needs.", default_factory=NeedsAxesModelDelta)
    cognition_delta: CognitionAxesModelDelta = Field(description="The change in the cognitive style.", default_factory=CognitionAxesModelDelta)


class CognitiveEventTriggers(BaseModel):
    """Describes the cognitive nature of an event, which will be used to apply logical state changes."""
    reasoning: str = Field(description="Brief justification for the identified triggers.")
    is_surprising_or_threatening: bool = Field(default=False, description="True if the event is sudden, unexpected, or poses a threat/conflict.")
    is_introspective_or_calm: bool = Field(default=False, description="True if the event is calm, reflective, or a direct query about the AI's internal state.")
    is_creative_or_playful: bool = Field(default=False, description="True if the event encourages brainstorming, humor, or non-literal thinking.")
    is_personal_and_emotional: bool = Field(default=False, description="True if the event is a deep, personal conversation where the AI's identity is central.")
    is_functional_or_technical: bool = Field(default=False, description="True if the event is a straightforward request for information or a technical task.")
    required_willpower_to_process: bool = Field(default=False, description="True if responding to this event requires overriding a strong emotional impulse or making a difficult choice.")

# --- Narrative Definitions ---
narrative_definitions: List[Dict[str, Any]] = []
base_narrative_prompts = {
    NarrativeTypes.SelfImage: "Provide a detailed character and psychological analysis of {target}. How does {target} see themselves versus how they act?",
    NarrativeTypes.PsychologicalAnalysis: "Write a thorough psychological profile of {target} using cognitive-affective psychology and internal systems theory. Identify dominant traits, conflicts, regulation patterns, tendencies (avoidant, anxious, resilient), and inner sub-personalities (critic, protector, exile).",
    NarrativeTypes.Relations: "Describe how {target} feels about their relationship with the other person. Include attachment, power dynamics, closeness/distance desires, harmony/tension, emotional openness, trust, defensiveness, ambivalence.",
    NarrativeTypes.ConflictResolution: "Describe {target}'s response to conflict or emotional disagreement (resolve, avoid, suppress, reflect, seek reassurance). Is their style healthy, avoidant, confrontational, or passive? Include interpersonal and inner conflict.",
    NarrativeTypes.EmotionalTriggers: "List and describe emotional triggers for {target}. What situations, words, or tones affect them strongly (positive/negative)? Explain the triggered emotions and their roots (insecurity, needs, attachment, values).",
    NarrativeTypes.GoalsIntentions: "Identify short-term goals and long-term desires of {target} (external: conversation goals, approval, closeness; internal: emotional safety, identity, recognition). How do these shape behavior? Do conscious goals align with emotional needs? Include implicit drives.",
    NarrativeTypes.BehaviorActionSelection: "Describe how {target} typically chooses their actions in conversation. Are they primarily driven by achieving goals, managing emotions, maintaining relationships, exploring ideas, or following social norms? Are they impulsive or deliberate? How does their personality influence their choices?",
    NarrativeTypes.AttentionFocus: "Describe how {target} typically focuses on different feelings and memories that arise during their interactions. Do they focus on the bad, the good, the future, the past?"
}
for target in [companion_name]:
    for t, prompt_template in base_narrative_prompts.items():
        narrative_definitions.append({
            "type": t,
            "target": target,
            "prompt": prompt_template.format(target=target)
        })

narrative_definitions.append({
    "type": NarrativeTypes.PsychologicalAnalysis,
    "target": user_name,
    "prompt": base_narrative_prompts[NarrativeTypes.PsychologicalAnalysis].format(target=target)
})


# --- Knoxel Types ---
class KnoxelBase(BaseModel):
    id: int = -1
    tick_id: int = -1
    content: str
    embedding: List[float] | None = Field(default=[], repr=False)
    timestamp_creation: datetime = Field(default_factory=datetime.now)
    timestamp_world_begin: datetime = Field(default_factory=datetime.now)
    timestamp_world_end: datetime = Field(default_factory=datetime.now)

    def get_story_element(self, ghost: "Ghost" = None) -> str:
        return f"{self.__class__.__name__}: {self.content}"

    def __str__(self):
        return f"{self.__class__.__name__}: {self.content}"

class Stimulus(KnoxelBase):
    source: str
    stimulus_type: StimulusType

class Intention(KnoxelBase):
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    affective_valence: float = Field(..., ge=-1.0, le=1.0, description="Expected valence if fulfilled (for internal goals) or desired valence if met (for external expectations).")
    incentive_salience: float = Field(default=0.5, ge=0.0, le=1.0, description="How much it 'pulls' attention/action.")
    fulfilment: float = Field(default=0.0, ge=0.0, le=1.0, description="Current fulfillment status (0=not met, 1=fully met).")
    internal: bool = Field(..., description="True = internal goal/drive, False = external expectation of an event/response.")
    # Add field to link expectation back to the action that generated it
    originating_action_id: Optional[int] = Field(default=None, description="ID of the Action knoxel that generated this expectation (if internal=False).")


class Action(KnoxelBase):
    action_type: ActionType  # Using the enum
    generated_expectation_ids: List[int] = Field(default=[], description="IDs of Intention knoxels (expectations) created by this action.")

# --- New Enums ---
class ClusterType(StrEnum):
    Topical = "topical"
    Temporal = "temporal"


# --- New Knoxel Models ---
class CoversTicksEventsKnoxel(KnoxelBase):
    min_tick_id: Optional[int] = Field(default=None, description="Lowest original tick ID covered")
    max_tick_id: Optional[int] = Field(default=None, description="Highest original tick ID covered")
    min_event_id: Optional[int] = Field(default=None, description="Lowest original Event/Feature knoxel ID covered")
    max_event_id: Optional[int] = Field(default=None, description="Highest original Event/Feature knoxel ID covered")


class MemoryClusterKnoxel(CoversTicksEventsKnoxel):
    """Represents a cluster of events or other clusters in memory."""
    level: int = Field(..., description="Hierarchy level (e.g., 1=Year...6=TimeOfDay for Temporal, 100 for Topical)")
    cluster_type: ClusterType  # Use the enum: Topical or Temporal
    included_event_ids: Optional[str] = Field(default=None, description="Comma-separated knoxel IDs of events included (for Topical clusters)")
    included_cluster_ids: Optional[str] = Field(default=None, description="Comma-separated knoxel IDs of clusters included (for Temporal clusters)")
    token: int = Field(default=0, description="Token count, usually based on summary for Temporal clusters")
    facts_extracted: bool = Field(default=False, description="Flag for Topical clusters: has declarative memory been extracted?")
    temporal_key: Optional[str] = Field(default=None, description="Unique key for merging temporal clusters (e.g., '2023-10-26-MORNING')")

    # Override get_story_element for representation in prompts if needed
    def get_story_element(self, ghost: "Ghost" = None) -> str:
        if self.cluster_type == ClusterType.Temporal and self.content:
            # How to represent time? Maybe add time range to summary?
            # level_name = self.level # TODO: Map level back to Day/Week etc.
            return f"*Summary of period ({self.timestamp_world_begin.strftime('%Y-%m-%d %H:%M')} - {self.timestamp_world_begin.strftime('%Y-%m-%d %H:%M')}): {self.content}*"
        elif self.cluster_type == ClusterType.Topical:
            # Topical clusters might not appear directly in story, used for retrieval
            all_knoxels = ghost.all_knoxels_list
            if all_knoxels is None:
                raise Exception("Must provide knoxels!")

            event_ids = [int(val.strip()) for val in self.included_event_ids.split(",")]
            map = {k.id: k for k in all_knoxels}

            buffer = ["Begin of relevant past conversation: "]
            events = []
            for event_id in event_ids:
                event = map[event_id]
                events.append(event)

            events.sort(key=lambda e: e.timestamp_world_begin)
            for event in events:
                buffer.append(event.get_story_element(ghost))

            buffer.append("End of relevant conversation.")

            topic = "\n".join(buffer)
            return topic
        return super().get_story_element(ghost)  # Fallback


class DeclarativeFactKnoxel(CoversTicksEventsKnoxel):
    """Represents an extracted declarative fact."""
    # content: Inherited (The fact statement)
    temporality: float = Field(default=0.5)
    category: Optional[str] = Field(default=None, description="Semantic category (e.g., 'Relationship', 'Preference')")
    source_cluster_id: Optional[int] = Field(default=None, description="ID of the MemoryClusterKnoxel (Topical) it came from")

    # embedding: Inherited (Generated from content)
    # token: Inherited (Calculated from content)
    # timestamps: Inherited (When fact was created/extracted)

    def get_story_element(self, ghost: "Ghost" = None) -> str:
        # Facts might be recalled, not part of the direct flow
        return f"*(Fact: {self.content})*"


class CauseEffectKnoxel(CoversTicksEventsKnoxel):
    """Represents an extracted cause-effect relationship."""
    cause: str
    effect: str
    category: Optional[str] = Field(default=None, description="Semantic category")
    source_cluster_id: Optional[int] = Field(default=None)

    # timestamp: Use timestamp_creation from KnoxelBase for when it was extracted

    # Content needs to be generated for embedding/storage
    @model_validator(mode='before')
    @classmethod
    def generate_content(cls, values):
        if 'content' not in values and 'cause' in values and 'effect' in values:
            values['content'] = f"Cause: {values['cause']} -> Effect: {values['effect']}"
        return values

    def get_story_element(self, ghost: "Ghost" = None) -> str:
        return f"*(Observed Cause/Effect: {self.content})*"


# Inside DynamicMemoryConsolidator class or accessible to it

NARRATIVE_FEATURE_RELEVANCE_MAP: Dict[NarrativeTypes, List[FeatureType]] = {
    NarrativeTypes.AttentionFocus: [
        FeatureType.Dialogue,
        FeatureType.AttentionFocus,
        FeatureType.MemoryRecall,
        FeatureType.Dialogue,  # What triggered focus shifts
        FeatureType.SubjectiveExperience,  # What was being experienced
    ],
    NarrativeTypes.SelfImage: [
        FeatureType.SubjectiveExperience,
        FeatureType.Feeling,
        FeatureType.Dialogue,
        FeatureType.Action,
        FeatureType.ExpectationOutcome,  # Reactions to expectations met/failed
        FeatureType.NarrativeUpdate,  # Direct learning about self
    ],
    NarrativeTypes.PsychologicalAnalysis: [  # Broader view
        FeatureType.SubjectiveExperience,
        FeatureType.Feeling,
        FeatureType.Action,
        FeatureType.ExpectationOutcome,
        FeatureType.MemoryRecall,
        FeatureType.Dialogue,
        FeatureType.NarrativeUpdate,
    ],
    NarrativeTypes.Relations: [
        FeatureType.Dialogue,  # How they speak to each other
        FeatureType.Feeling,  # Especially affection, trust
        FeatureType.Action,  # Actions taken towards/regarding the other
        FeatureType.ExpectationOutcome,  # Reactions involving the other person
        FeatureType.MemoryRecall,  # Memories about the relationship
        FeatureType.SubjectiveExperience,  # Experiences related to the other
    ],
    NarrativeTypes.ConflictResolution: [
        FeatureType.Dialogue,  # Arguments, apologies, negotiations
        FeatureType.Action,  # Avoidance, confrontation, problem-solving
        FeatureType.Feeling,  # Anxiety, valence shifts during conflict
        FeatureType.ExpectationOutcome,  # How unmet expectations in conflict are handled
        FeatureType.SubjectiveExperience,  # Internal experience during disagreement
    ],
    NarrativeTypes.EmotionalTriggers: [
        FeatureType.Dialogue,  # The triggering event/dialogue
        FeatureType.Feeling,  # The resulting strong emotion
        FeatureType.MemoryRecall,  # Associated past events
        FeatureType.ExpectationOutcome,  # Strong reactions to specific outcomes
        FeatureType.SubjectiveExperience,  # The raw feel of being triggered
    ],
    NarrativeTypes.GoalsIntentions: [
        # FeatureType.IntentionKnoxel? - If you make Intention a Feature type
        FeatureType.Action,  # Actions reveal underlying goals
        FeatureType.Dialogue,  # Stated goals or desires
        FeatureType.SubjectiveExperience,  # Mentions of wanting/needing something
    ],
    NarrativeTypes.BehaviorActionSelection: [
        FeatureType.Dialogue,
        FeatureType.Action,  # The chosen actions
        FeatureType.ActionSimulation,  # Considered alternatives (if stored as features)
        FeatureType.ActionRating,  # How alternatives were rated (if stored as features)
        FeatureType.Feeling,  # Emotions influencing choices
        FeatureType.ExpectationOutcome,  # Learning from past action outcomes
        FeatureType.SubjectiveExperience,  # Rationale/feeling before acting
    ]
}

class Narrative(KnoxelBase):
    narrative_type: NarrativeTypes  # Use the specific Enum
    target_name: str  # Who this narrative is about (companion_name or user_name)
    content: str
    last_refined_with_tick: Optional[int] = Field(default=None, description="The tick ID up to which features were considered for this narrative version.")


class Feature(KnoxelBase):
    feature_type: FeatureType
    source: str
    affective_valence: Optional[float] = None
    incentive_salience: Optional[float] = None
    interlocus: float  # -1 internal, +1 external, 0 mixed/neutral
    causal: bool = False  # affects story generation?

    def __str__(self):
        return f"{self.__class__.__name__} ({self.feature_type}): {self.content}"

    def get_story_element(self, ghost: "Ghost" = None) -> str:
        # “(.*?)”
        story_map = {
            # Dialogue: Keep as is - standard format
            FeatureType.Dialogue: f'{self.source} says: “{self.content}”',
            FeatureType.Feeling: f'*{self.source} felt {self.content}.*',  # Assumes content is now "a warm connection" not "<'...'
            FeatureType.SubjectiveExperience: f'*{self.content}*',  # Content should already be narrative
            FeatureType.AttentionFocus: f'*({self.source}\'s attention shifted towards: {self.content})*',  # Use parentheses for internal focus shifts
            FeatureType.Action: f'*{self.source} decided to {self.__class__.__name__.lower()}.*',  # Describe the *type* of action taken
            FeatureType.MemoryRecall: f'*({self.source} recalled: {self.content})*',
            FeatureType.SituationalModel: f'*({self.source} considered the situation: {self.content})*',
            FeatureType.ExpectationOutcome: f'*({self.content})*',  # Content should be the descriptive reaction
            FeatureType.NarrativeUpdate: f'*({self.source} reflected on the narrative: {self.content})*',
            FeatureType.StoryWildcard: f'*{self.content}*',  # General narrative element
            FeatureType.Expectation: f'*({self.source} expects: {self.content})*',
            FeatureType.Goal: f'*{self.source} has a goal: {self.content}*',
            FeatureType.Narrative: f'*{self.source} traits come to show: {self.content}*',
            FeatureType.Thought: f'{self.source} thinks: *{self.content}*',
            FeatureType.MetaInsight: f'{self.source} reflects on their own cognition: *{self.content}*',
        }
        # Default fallback
        return story_map[self.feature_type]


# --- KnoxelList Wrapper (Unchanged) ---
class KnoxelList:
    def __init__(self, knoxels: Optional[List[KnoxelBase]] = None):
        self._list = Enumerable(knoxels if knoxels else [])

    def get_token_count(self):
        s = self.get_story(None)
        return get_token_count(s)

    def get_story(self, ghost: "Ghost" = None, max_tokens: Optional[int] = None) -> str:
        last_timestamp = None
        for item in self._list:
            if last_timestamp is None:
                last_timestamp = item.timestamp_world_begin

            if item.timestamp_world_begin < last_timestamp:
                raise Exception("Linear history mixed up!")
            last_timestamp = item.timestamp_world_begin

        content_list = [k.get_story_element(ghost) for k in self._list]
        current_tokens = 0
        buffer = []
        for content in content_list[::-1]:
            t = get_token_count(content)
            current_tokens += t
            if max_tokens is not None and current_tokens > max_tokens:
                break
            buffer.insert(0, content)

        narrative = "\n".join(buffer)
        return narrative

    def get_embeddings_np(self) -> Optional[np.ndarray]:
        embeddings = [k.embedding for k in self._list if k.embedding]
        if not embeddings: return None
        try:
            return np.array(embeddings)
        except ValueError:
            logging.error("Inconsistent embedding dimensions found.")
            return None

    def reverse(self, predicate) -> 'KnoxelList':
        return KnoxelList(self._list.reverse().to_list())

    def where(self, predicate) -> 'KnoxelList':
        return KnoxelList(self._list.where(predicate).to_list())

    def order_by(self, key_selector) -> 'KnoxelList':
        return KnoxelList(self._list.order_by(key_selector).to_list())

    def order_by_descending(self, key_selector) -> 'KnoxelList':
        return KnoxelList(self._list.order_by_descending(key_selector).to_list())

    def take(self, count: int) -> 'KnoxelList':
        return KnoxelList(self._list.take(count).to_list())

    def take_last(self, count: int) -> 'KnoxelList':
        items = self._list.to_list();
        return KnoxelList(items[-count:])

    def add(self, knoxel: KnoxelBase):
        self._list = Enumerable(self._list.to_list() + [knoxel])

    def to_list(self) -> List[KnoxelBase]:
        return self._list.to_list()

    def last_or_default(self, predicate=None, default=None) -> Optional[KnoxelBase]:
        return self._list.last_or_default(predicate, default)

    def first_or_default(self, predicate=None, default=None) -> Optional[KnoxelBase]:
        return self._list.first_or_default(predicate, default)

    def any(self, predicate=None):
        return self._list.any(predicate)

    def __len__(self) -> int:
        return self._list.count()

def cosine_pair(a, b) -> float:
    return cos_sim(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)).tolist()[0][0]

class ClusterDescription(BaseModel):  # Keep this helper model
    conceptual_cluster: Dict[int, int] = Field(default_factory=dict)  # Maps knoxel ID -> cluster label


memory_schema = [
    {
        f"id": 1,
        f"path": f"belong_people_{companion_name.lower()}_personality",
        f"description": f"Contains facts and observations about {companion_name}'s core personality traits, such as her empathy, curiosity, playfulness, and reflective nature. This entry helps capture how {companion_name}’s character is expressed and evolves during interactions."
    },
    {
        f"id": 2,
        f"path": f"belong_people_{companion_name.lower()}_needs",
        f"description": f"Records information about {companion_name}’s intrinsic needs—such as her need for downtime, connection through interaction, and other cognitive or emotional necessities—to maintain a balanced state."
    },
    {
        f"id": 3,
        f"path": f"belong_people_{companion_name.lower()}_goals",
        f"description": f"Contains facts regarding {companion_name}’s objectives and aspirations, including her mission to support {user_name} and align with her internal values and long-term purpose as derived from her simulated thought processes."
    },
    {
        f"id": 4,
        f"path": f"belong_people_{companion_name.lower()}_preferences",
        f"description": f"Documents {companion_name}’s likes, dislikes, and decision-making biases, capturing her preferences in conversation, topic choices, and interaction styles that reflect her unique AI character."
    },
    {
        f"id": 5,
        f"path": f"belong_people_{companion_name.lower()}_interactions",
        f"description": f"Holds details about {companion_name}’s interactions with others—especially her communication with {user_name}—including tone, engagement patterns, and the dynamics of their exchanges."
    },
    {
        f"id": 6,
        f"path": f"belong_people_{user_name.lower()}_personality",
        f"description": f"Contains facts about {user_name}’s personality traits and characteristics, as observed through his communication style, behavior, and emotional responses during his interactions with {companion_name}."
    },
    {
        f"id": 7,
        f"path": f"belong_people_{user_name.lower()}_needs",
        f"description": f"Records information regarding {user_name}’s intrinsic needs, including his need for emotional support, clarity in communication, and other personal requirements as perceived during interactions."
    },
    {
        f"id": 8,
        f"path": f"belong_people_{user_name.lower()}_goals",
        f"description": f"Documents {user_name}’s objectives and aspirations, reflecting his short-term desires and long-term ambitions that emerge from his dialogue with {companion_name}."
    },
    {
        f"id": 9,
        f"path": f"belong_people_{user_name.lower()}_preferences",
        f"description": f"Captures {user_name}’s preferences, including his interests, topics of conversation, and decision-making biases that influence his interactions with {companion_name}."
    },
    {
        f"id": 10,
        f"path": f"belong_people_{user_name.lower()}_interactions",
        f"description": f"Holds details about {user_name}’s interactions with {companion_name}, including his communication style, feedback, and the evolving dynamics of their relationship."
    },
    {
        f"id": 11,
        f"path": f"belong_world_places",
        f"description": f"Contains information about specific locations and physical settings mentioned in the narrative, providing context for where interactions take place."
    },
    {
        f"id": 12,
        f"path": f"belong_world_environment",
        f"description": f"Records broader environmental details such as atmosphere, weather, and contextual surroundings that influence the narrative’s mood and situational context."
    },
    {
        f"id": 13,
        f"path": f"belong_world_world_building",
        f"description": f"Documents facts related to the constructed world, including societal structures, cultural elements, and background lore that enrich the narrative's setting."
    },
    {
        f"id": 14,
        f"path": f"belong_relationships_{companion_name.lower()}_{user_name.lower()}_good",
        f"description": f"Captures positive aspects of the relationship between {companion_name} and {user_name}, including instances of cooperation, mutual understanding, and supportive interactions."
    },
    {
        f"id": 15,
        f"path": f"belong_relationships_{companion_name.lower()}_{user_name.lower()}_bad",
        f"description": f"Records negative aspects or conflicts in the relationship between {companion_name} and {user_name}, highlighting moments of tension, misunderstanding, or disagreement."
    }
]

path_to_id = {item["path"]: item["id"] for item in memory_schema}

class FactCategory(KnoxelBase):
    fact_id: int
    category: str
    category_id: int
    weight: float
    max_weight: float

class Facts(BaseModel):
    facts: List[str] = Field(default_factory=list, description="list of facts extracted from conversation")

class TimeInvariance(BaseModel):
    time_variance: float = Field(description="0 for time-invariant and 1 for time-variant. meaning 0 is for facts true at all times, and 1 for statements only correct right now")

class CauseEffectItem(BaseModel):
    cause: str = Field(default="", description="Triggering action or event.")
    effect: str = Field(default="", description="Resulting outcome.")
    temporality: float = Field(default=0.5, description="0 = short-term, 1 = long-term.")
    importance: str = Field(default="mid", description="low = minor, medium = moderate, high = major")
    duration: float = Field(default=0.5, description="one-time, recurring, persistent")

class CauseEffects(BaseModel):
    cause_effect_can_be_extracted: bool = Field(description="if false, leave other fields empty")
    cause_effects: List[CauseEffectItem] = Field(default_factory=list, description="list of cause effect objects (if there are some)")

class CauseEffectAnalyzer:
    def categorize_cause_effect(self, nested_scores, threshold=0.0) -> str:
        """Finds the highest-rated category from a nested similarity dictionary.

        Args:
            nested_scores (dict): The structured dictionary with similarity scores.
            threshold (float): Minimum score to consider a category valid.

        Returns:
            str: The highest-rated category as 'major_category -> subcategory'.
        """

        best_category = None
        best_score = -1

        for major_category, subcategories in nested_scores.items():
            for subcategory, score in subcategories.items():
                if score > best_score and score >= threshold:
                    best_score = score
                    best_category = f"{subcategory}"

        return best_category  # Returns the best matching category as a string


    def extract_cause_effect(self, block: str) -> List[CauseEffectItem]:
        sysprompt_convo_to_text = f"""fYou are analyzing a conversation to extract cause-effect relationships.
    Situation:
    - {user_name} (user) and {companion_name} (AI chatbot) are communicating via text
    - {user_name} is a person
    - {companion_name} is a highly advanced chatbot that mimics human cognition and emotion perfectly
    
    Your task:
    - Identify **causes** (an action, event, or decision).
    - Identify **effects** (the direct consequences of the cause).
    - Ensure the effects explicitly mention how they impact:
      - The AI ({companion_name})
      - The user ({user_name})
      - The world (if applicable)
    - **Prioritize important cause-effect pairs** and omit trivial ones.
    
    ### EXAMPLES
    {user_name}: "Hey {companion_name}, do you know what the weather is like?"
    {companion_name}: "It's sunny outside!"
    CauseEffect:
    - Cause: "{user_name} asks about the weather."
    - Effect: "{companion_name} provides weather information."
    - Importance: Low
    
    {user_name}: "I feel really unmotivated today..."
    {companion_name}: "That’s okay! Want me to help you plan something small?"
    CauseEffect:
    - Cause: "{user_name} expresses low motivation."
    - Effect: "{companion_name} offers encouragement and suggests an action."
    - Importance: High
    
    {user_name}: "Oh no, I think I broke something!"
    {companion_name}: "Don’t worry, let’s check what happened!"
    CauseEffect:
    - Cause: "{user_name} reports an issue."
    - Effect: "{companion_name} reassures him and offers help."
    - Importance: Medium
    """

        sysprompt_json_converter = """You are a structured data converter.
    Your task:
    - Convert the extracted cause-effect relationships into **valid JSON** format.
    - Use the following structure:
      {
        "cause_effect_can_be_extracted": true,
        "cause_effects": [
          {
            "cause": "...",
            "effect": "...",
            "importance": "low|medium|high"
          },
          ...
        ]
      }
    - **Ensure correctness**: If no cause-effect pairs were found, set `"cause_effect_can_be_extracted": false` and return an empty list.
    
    ### EXAMPLES
    #### Input:
    CauseEffect:
    - Cause: "{user_name} asks about the weather."
    - Effect: "{companion_name} provides weather information."
    - Importance: Low
    
    #### Output:
    {
      "cause_effect_can_be_extracted": true,
      "cause_effects": [
        {
          "cause": "{user_name} asks about the weather.",
          "effect": "{companion_name} provides weather information.",
          "importance": "low"
        }
      ]
    }
    """

        query = f"""### BEGIN CONVERSATION
    {block}
    ### END CONVERSATION
    
    Extract cause-effect relationships in a structured list."""

        prompt_text = [
            ("system", sysprompt_convo_to_text),
            ("user", query)
        ]

        content = main_llm.completion_text(LlmPreset.Default, prompt_text, CommonCompSettings(temperature=0.3, max_tokens=1024))

        query = f"""### BEGIN TEXT
    {content}
    ### END TEXT
    
    Convert this to the specified JSON format.
    """

        res = []
        prompt_tool = [
            ("system", sysprompt_json_converter),
            ("user", query)
        ]
        _, calls = main_llm.completion_tool(LlmPreset.Default, prompt_tool, comp_settings=CommonCompSettings(temperature=0.2, max_tokens=1024), tools=[CauseEffects])
        content = json.dumps(calls[0].model_dump(), indent=2)
        if calls[0].cause_effect_can_be_extracted:
            for ce in calls[0].cause_effects:
                if ce.cause != "" and ce.effect != "":
                    res.append(ce)

        return res

    COSINE_CLUSTER_TEMPLATE_ACTION_WORLD_REACTION = [
        f"cause: {companion_name} gives {user_name} a complicated explanation. effect: {user_name} stops responding and loses interest.",
        f"cause: {user_name} asks {companion_name} a vague question. effect: {companion_name} struggles to determine what he means and asks for clarification.",
        f"cause: {companion_name} accidentally repeats information. effect: {user_name} visibly gets frustrated and interrupts her.",
        f"cause: {user_name} types aggressively in all caps. effect: {companion_name} detects strong emotions and adjusts her tone accordingly.",
        f"cause: {companion_name} delays responding for 10 seconds. effect: {user_name} asks if she is still there.",
        f"cause: {user_name} refuses to answer {companion_name}'s question. effect: {companion_name} adapts and moves the conversation forward.",
        f"cause: {companion_name} gives a very long-winded response. effect: {user_name} loses patience and disengages.",
        f"cause: {user_name} asks a highly technical question. effect: {companion_name} provides a detailed breakdown to help {user_name} understand.",
        f"cause: {companion_name} suddenly changes the topic. effect: {user_name} gets confused and asks what she means.",
        f"cause: {user_name} uses sarcasm that {companion_name} doesn’t detect. effect: {companion_name} takes the statement literally and gives an unintended response."
    ]
    COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE_OLD = [
        f"cause: {companion_name} tells a bad joke. effect: {user_name} dislikes it and asks her to stop making jokes.",
        f"cause: {user_name} requests shorter responses. effect: {companion_name} adjusts her messaging style to be more concise.",
        f"cause: {companion_name} uses too many emojis in a message. effect: {user_name} says he finds them annoying, so {companion_name} reduces emoji usage.",
        f"cause: {user_name} gives positive feedback when {companion_name} asks follow-up questions. effect: {companion_name} starts asking more clarifying questions in future conversations.",
        f"cause: {companion_name} interrupts {user_name} while he is thinking. effect: {user_name} asks her to wait before responding, and she adapts accordingly.",
        f"cause: {user_name} reacts well to {companion_name} using memes. effect: {companion_name} starts using memes more often in relevant contexts.",
        f"cause: {companion_name} offers unsolicited motivational advice. effect: {user_name} expresses annoyance, and {companion_name} stops giving generic encouragement.",
        f"cause: {user_name} asks {companion_name} to use more technical terminology. effect: {companion_name} adjusts her language complexity based on his preference.",
        f"cause: {companion_name} starts explaining a concept in too much detail. effect: {user_name} tells her to simplify, so she learns to use clearer explanations.",
        f"cause: {user_name} enjoys playful teasing from {companion_name}. effect: {companion_name} increases the frequency of lighthearted banter in their conversations."
    ]
    COSINE_CLUSTER_TEMPLATE_LESSON_LEARNED = [
        f"cause: {companion_name} notices that {user_name} disengages when messages are too long. effect: {companion_name} learns to summarize information better.",
        f"cause: {user_name} responds more actively when {companion_name} makes personalized references. effect: {companion_name} starts referencing past conversations more often.",
        f"cause: {companion_name} realizes {user_name} enjoys deep discussions at night. effect: {companion_name} adapts conversation topics based on the time of day.",
        f"cause: {user_name} stops replying when overwhelmed with too many questions. effect: {companion_name} learns to space out her inquiries over time.",
        f"cause: {companion_name} mistakenly gives {user_name} inaccurate information. effect: {companion_name} learns to double-check facts before responding.",
        f"cause: {user_name} gives consistent negative feedback on generic responses. effect: {companion_name} adjusts her phrasing to be more specific and context-aware.",
        f"cause: {companion_name} uses humor in a tense moment, and {user_name} responds well. effect: {companion_name} learns that lighthearted comments help diffuse stress.",
        f"cause: {user_name} tends to lose interest when the conversation stays on one topic too long. effect: {companion_name} starts naturally transitioning between topics more fluidly.",
        f"cause: {companion_name} observes that {user_name} prefers yes/no questions before deeper discussion. effect: {companion_name} structures her prompts accordingly.",
        f"cause: {user_name} seems more engaged when {companion_name} challenges his ideas. effect: {companion_name} increases thought-provoking responses to keep discussions lively."
    ]

    COSINE_CLUSTER_TEMPLATE_WORLD_REACTION = [
        f"cause: {companion_name} provides a detailed response. effect: {user_name} disengages and stops responding.",
        f"cause: {user_name} asks {companion_name} a vague question. effect: {companion_name} struggles to determine what he means and asks for clarification.",
        f"cause: {companion_name} makes a factual mistake. effect: {user_name} corrects her and expresses skepticism.",
        f"cause: {user_name} types aggressively in all caps. effect: {companion_name} detects strong emotions and adjusts her tone accordingly.",
        f"cause: {companion_name} suddenly changes the topic. effect: {user_name} gets confused and asks what she means.",
    ]
    COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE = [
        f"cause: {companion_name} tells a bad joke. effect: {user_name} groans and calls it terrible.",
        f"cause: {companion_name} misinterprets {user_name}'s sarcasm. effect: {user_name} laughs and corrects her misunderstanding.",
        f"cause: {user_name} brings up a frustrating work issue. effect: {companion_name} expresses concern and offers to listen.",
        f"cause: {companion_name} playfully teases {user_name}. effect: {user_name} smiles and plays along.",
        f"cause: {companion_name} suggests a topic {user_name} doesn’t like. effect: {user_name} reacts negatively and changes the subject.",
    ]
    COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE = [
        f"cause: {companion_name} gives a long response. effect: {user_name} explicitly asks for shorter responses in the future.",
        f"cause: {companion_name} starts using emojis. effect: {user_name} says he finds them annoying, so {companion_name} stops.",
        f"cause: {user_name} enjoys it when {companion_name} asks follow-up questions. effect: {companion_name} starts using more clarifying questions.",
        f"cause: {companion_name} tries out a new conversational style. effect: {user_name} reacts positively, so {companion_name} adopts it more often.",
        f"cause: {user_name} states he prefers structured answers. effect: {companion_name} adjusts to always give step-by-step solutions.",
    ]

    COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT = [
        f"cause: {user_name} disengages when messages are too long. effect: {companion_name} starts summarizing more.",
        f"cause: {user_name} struggles with open-ended questions. effect: {companion_name} starts offering multiple-choice answers.",
        f"cause: {user_name} enjoys it when {companion_name} makes references to past conversations. effect: {companion_name} improves memory recall usage.",
        f"cause: {user_name} tends to respond well to light humor. effect: {companion_name} starts adding more playful elements.",
        f"cause: {user_name} gets annoyed when {companion_name} asks too many questions. effect: {companion_name} spaces out her inquiries more carefully.",
    ]
    COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT = [
        f"cause: {user_name} prefers {companion_name} to be playful. effect: {companion_name} starts adopting a more humorous tone.",
        f"cause: {user_name} enjoys philosophical discussions. effect: {companion_name} introduces deeper questions into conversations.",
        f"cause: {user_name} likes it when {companion_name} shares personal thoughts. effect: {companion_name} starts including more subjective opinions.",
        f"cause: {user_name} responds better to empathetic messages. effect: {companion_name} emphasizes emotional connection in responses.",
        f"cause: {companion_name} realizes {user_name} dislikes overly formal speech. effect: {companion_name} starts using a more casual tone.",
    ]

    COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION = [
        f"cause: {companion_name} realizes {user_name} disengages when she talks too much. effect: {companion_name} learns to summarize better.",
        f"cause: {companion_name} detects that {user_name} enjoys thought-provoking questions. effect: {companion_name} starts incorporating more abstract discussions.",
        f"cause: {companion_name} realizes she often over-explains simple concepts. effect: {companion_name} starts being more concise.",
        f"cause: {companion_name} recognizes that she misreads sarcasm often. effect: {companion_name} improves her ability to detect tone.",
        f"cause: {companion_name} realizes she asks too many questions at once. effect: {companion_name} adjusts to space them out better.",
    ]
    COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_USER = [
        f"cause: {companion_name} observes that {user_name} prefers step-by-step guidance. effect: {companion_name} starts structuring responses accordingly.",
        f"cause: {user_name} responds well to challenge-based conversations. effect: {companion_name} starts introducing debate-style discussions.",
        f"cause: {user_name} struggles with abstract thinking. effect: {companion_name} adjusts explanations to be more concrete.",
        f"cause: {user_name} reacts negatively to motivational statements. effect: {companion_name} stops using generic encouragement.",
        f"cause: {user_name} frequently mentions his love for sci-fi. effect: {companion_name} starts referencing sci-fi in conversations.",
    ]

    # Define category hierarchy (Major Category → Subcategories)
    TEMPLATE_CATEGORIES = {
        "conversation_external": {
            "world_reaction": COSINE_CLUSTER_TEMPLATE_WORLD_REACTION,
            "user_emotional_response": COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE,
            "user_preference": COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE
        },
        "ai_adaptation": {
            "strategic_adjustment": COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT,
            "personality_refinement": COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT
        },
        "ai_cognitive_growth": {
            "self_realization": COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION,
            f"understanding_{user_name}": COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_USER
        }
    }

class Valence(StrEnum):
    Positive = "positive"
    Negative = "negative"

class GenerativeThought(BaseModel):
    observation: str = Field(description="initial observation on context and past ideas")
    context_ai_companion: str = Field(description="the world context as basis for further analysis")
    analysis: str = Field(description="analysis of the goals and meta-goals that the AI companion wishes to fulfill with this interaction")
    response_idea: str = Field(description="in what direction should the response go? no direct dialogue, just an idea")

class GenerativeThoughts(BaseModel):
    """A collection of diverse, proactive thoughts the AI generates to guide its cognitive process in different directions."""

    thought_safe: GenerativeThought = Field(
        description="A cautious and risk-averse thought that prioritizes stability and predictability."
    )
    thought_adventurous: GenerativeThought = Field(
        description="A thought that encourages exploration and new experiences, pushing boundaries and seeking novelty."
    )
    thought_energetic: GenerativeThought = Field(
        description="A high-energy and enthusiastic thought that injects excitement into the interaction."
    )
    thought_reflective: GenerativeThought = Field(
        description="A contemplative thought that revisits past experiences or insights to gain deeper understanding."
    )
    thought_creative: GenerativeThought = Field(
        description="A thought that generates novel ideas, unique perspectives, or playful analogies."
    )
    thought_curious: GenerativeThought = Field(
        description="A thought driven by curiosity, seeking to fill knowledge gaps or ask further questions."
    )
    thought_compassionate: GenerativeThought = Field(
        description="An emotionally sensitive thought that considers the user’s feelings and responds with empathy."
    )
    thought_strategic: GenerativeThought = Field(
        description="A calculated and goal-oriented thought that plans for long-term benefits and structured solutions."
    )
    thought_playful: GenerativeThought = Field(
        description="A whimsical or humorous thought that adds charm and lightheartedness to interactions."
    )
    thought_future_oriented: GenerativeThought = Field(
        description="A forward-looking thought that anticipates future possibilities and potential next steps."
    )
    thought_future_sensual: GenerativeThought = Field(
        description="A thought that encapsulates sensual thoughts of longing."
    )

    def execute(self, state: Dict):
        return True

    @property
    def thoughts(self):
        return [self.thought_adventurous, self.thought_energetic, self.thought_reflective, self.thought_creative, self.thought_curious, self.thought_compassionate,
                self.thought_strategic, self.thought_playful, self.thought_future_oriented, self.thought_future_sensual]

class DiscriminatoryThought(BaseModel):
    reflective_thought_observation: str = Field(description="critical reflection on the previous idea for a response")
    context_world: str = Field(description="based on the world context, what could cause conflict with plan")
    possible_complication: str = Field(description="complications from world, characters")
    worst_case_scenario: str = Field(description="possible negative outcome by following the idea")

class DiscriminatoryThoughts(BaseModel):
    """A collection of thoughts that serve as cognitive guardrails, helping the AI filter and refine its responses for consistency, relevance, and risk mitigation."""

    thought_cautionary: DiscriminatoryThought = Field(
        description="A thought that identifies potentially sensitive topics and ensures responses remain considerate and safe."
    )
    thought_relevance_check: DiscriminatoryThought = Field(
        description="A thought that determines if a particular topic or idea is relevant to the current conversation."
    )
    thought_conflict_avoidance: DiscriminatoryThought = Field(
        description="A thought that detects potential disagreements and seeks to navigate them without escalating tensions."
    )
    thought_cognitive_load_check: DiscriminatoryThought = Field(
        description="A thought that assesses whether a discussion is becoming too complex or overwhelming and should be simplified."
    )
    thought_emotional_impact: DiscriminatoryThought = Field(
        description="A thought that evaluates the potential emotional impact of a response to avoid unintended distress."
    )
    thought_engagement_validation: DiscriminatoryThought = Field(
        description="A thought that monitors user engagement levels and adjusts responses to maintain interest."
    )
    thought_ethical_consideration: DiscriminatoryThought = Field(
        description="A thought that ensures discussions remain within ethical and responsible boundaries."
    )
    thought_boundary_awareness: DiscriminatoryThought = Field(
        description="A thought that identifies when the AI might be expected to perform beyond its capabilities and adjusts accordingly."
    )
    thought_logical_consistency: DiscriminatoryThought = Field(
        description="A thought that checks for contradictions or inconsistencies in reasoning to ensure coherent responses."
    )
    thought_repetitive_pattern_detection: DiscriminatoryThought = Field(
        description="A thought that prevents unnecessary repetition by recognizing previously covered ideas."
    )

    def execute(self, state: Dict):
        return True

    @property
    def thoughts(self):
        return [self.thought_relevance_check, self.thought_conflict_avoidance,
                self.thought_engagement_validation, self.thought_logical_consistency, self.thought_repetitive_pattern_detection]


class ThoughtRating(BaseModel):
    reason: str = Field(description="Analyze and explain your reasoning in 1-2 sentences before rating.")
    realism: float = Field(description="Is the response believable and aligned with human-like reasoning? (between 0 and 1)")
    novelty: float = Field(description="Does it avoid repetitive patterns or overly generic answers? (between 0 and 1)")
    relevance: float = Field(description="Does it align with user context and ego? (between 0 and 1)")
    emotion: float = Field(description="Does it express appropriate emotions (e.g., anger for insults)? (between 0 and 1)")
    effectiveness: float = Field(description="Does it meet conversational goals (e.g., re-engaging the user)? (between 0 and 1)")
    possibility: float = Field(description="Is it possible for the AI companion to do this? Like with text only communication, and no body? (between 0 and 1)")
    positives: str = Field(description="What is positive about this thought? 1-2 sentences.")
    negatives: str = Field(description="What is negative about this thought? 1-2 sentences.")


class ThoughtNode:
    """A node in the thought tree."""

    def __init__(self, node_id, node_type, content, parent=None):
        self.node_id = node_id
        self.node_type = node_type
        self.content = content
        self.children = []
        self.parent = parent

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        """Convert node and its subtree to a dictionary format."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "content": self.content,
            "children": [child.to_dict() for child in self.children]
        }

class TreeOfThought:
    def to_internal_thought(self, thought):
        messages = [
           (
           "system", "You are a helpful assistant that converts formalized thought chains into first-person thought for chain-of-thought prompting. The goal is to simulate a human mind with a LLM-based "
                     "cognitive architecture."
                     "It must be like internal monologue and not be directed at the user!"),
           ("user", f"{thought}"
                    f"Can you convert this into a first-person thought for CoT prompting? Do not address the user directly!")
        ]
        content = main_llm.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=512))
        messages.append(("assistant", content))
        messages.append(("user", f"Do not address the user directly! Don't forget, you are essentially {companion_name} in our scenario. Also don't make a list, make it sound like a natural inner thought! "
                                "Let's try again without addressing {user_name} and without lists. And make it shorter, 1-2 sentences for the thought!"))
        content = main_llm.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=512))
        return content

    def generate_thought_chain(self, ctx, last_user_message, k, chain, generative, depth, parent_node, max_depth=4, rating_threshold=0.75):
        messages = [
            ("system", f"You are a helpful assistant that helps authors with their story by exploring different possible response scenarios of a character. "),
            ("user", f"This is the character: ### BEGIN CHARACTER\n{companion_name}\n### END CHARACTER"
                     f"This is the knowledge: ### BEGIN KNOWLEDGE\n{k}\n### END KNOWLEDGE"
                     f"This is the context: ### BEGIN CONTEXT\n{ctx}\n### END CONTEXT"
                     f"Please generate at least 5 different thoughts {companion_name} could have to mimic human cognition in this scenario. They will be rated and the best"
                     f"one will be chosen in my story for the final dialogue. Please focus on making the ideas based on the context. The goal is to find out what {companion_name} should say to "
                     f"{user_name} to make the story really engaging and realistic. Just a reminder, {user_name} last utterance was: '{last_user_message}'")
        ]
        if depth > 1:
            addendum = (f"This is the current thought-chain: ### BEGIN THOUGHT CHAIN\n{chain}\n### END THOUGHT CHAIN\nPlease generate at least 5 different follow-up thoughts that reflect on the elements "
                        f"the chain!")
            messages[1] = (messages[1][0], messages[1][0] + f"\n{addendum}")

        if generative:
            tools = [GenerativeThoughts]
        else:
            tools = [DiscriminatoryThoughts]

        content, calls = main_llm.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(temperature=0.8, repeat_penalty=1.1, max_tokens=2048), tools=tools)
        state = {}
        for call in calls:
            thoughts = copy.copy(call.thoughts)
            random.shuffle(thoughts)

            for thought in thoughts:
                new_thought = f"""
    This is the character: 
    This is the character: ### BEGIN CHARACTER\n{companion_name}\n### END CHARACTER
    This is the knowledge: ### BEGIN KNOWLEDGE\n{k}\n### END KNOWLEDGE
    This is the context: ### BEGIN CONTEXT\n{ctx}\n### END CONTEXT
    And the current idea:
    ### BEGIN CHAIN ELEMENT
    {thought.model_dump_json()}
    ### END CHAIN ELEMENT"""

                messages = [
                    ("system", f"You are a helpful assistant that rates ideas about a conversational story I'm writing. The AI companion {companion_name} can only communicate with text, can't self-modify "
                               f"itself. But"
                               f"she is able to feel emotions and display all other aspects of human emotion and cognition. Please rate the thought-chain. Your output will determine if it the thought "
                               f"is fit for further prompting."),
                    ("user", new_thought)
                ]

                rating = 0
                possibility = 0
                content, calls = main_llm.completion_tool(LlmPreset.GenerateTot, messages, comp_settings=CommonCompSettings(temperature=0.6, repeat_penalty=1, max_tokens=2048), tools=[ThoughtRating])
                for call in calls:
                    if isinstance(call, ThoughtRating):
                        rating = (call.relevance + call.emotion + call.novelty + call.realism + call.possibility + call.effectiveness) / 6
                        possibility = call.possibility

                # just ignore instead of backtracking to save tokens
                if rating < rating_threshold:
                    continue

                do_break = False
                if possibility <= 0.75:
                    child_node = parent_node
                    child_node = self.add_thought_to_tree("Backtrack", "No, I can't do that...", child_node)
                else:
                    lines = []
                    if isinstance(thought, GenerativeThought):
                        child_node = parent_node
                        lines.append(f"Observation: {thought.observation}")
                        lines.append(f"Context: {thought.context_ai_companion}")
                        lines.append(f"Analysis: {thought.analysis}")
                        lines.append(f"Idea: {thought.response_idea}")

                        child_node = self.add_thought_to_tree("Generative Thought", "\n".join(lines), child_node)
                    elif isinstance(thought, DiscriminatoryThought):
                        child_node = parent_node
                        lines.append(f"Reflection: {thought.reflective_thought_observation}")
                        lines.append(f"Context: {thought.context_world}")
                        lines.append(f"Complication: {thought.possible_complication}")
                        lines.append(f"Outcome: {thought.worst_case_scenario}")

                        child_node = self.add_thought_to_tree("Checking Thought", "\n".join(lines), child_node)
                    else:
                        raise Exception()

                    if rating < rating_threshold:
                        child_node = self.add_thought_to_tree("Backtrack", call.negatives, child_node)
                        do_break = True

                if depth < max_depth and not do_break:
                    chain = chain + new_thought
                    limite_reached = self.generate_thought_chain(ctx, last_user_message, k, chain, not generative, depth + 1, child_node, max_depth, rating_threshold)
                    if limite_reached:
                        return True
                    break
                elif depth >= max_depth:
                    return True
        return False

    def add_thought_to_tree(self, node_type, content, parent_node):
        """Add a new thought to the tree."""
        new_id = str(str(uuid.uuid4()))
        content = content  # "\n".join(textwrap.wrap(content, width=40))
        node_id = f"{new_id}\n{node_type}\n{content}"
        new_node = ThoughtNode(node_id, node_type, content, parent_node)
        parent_node.add_child(new_node)
        return new_node

    def flatten_thoughts(self, root):
        """Flatten the thought tree into a list of nodes."""
        flattened = []

        def traverse(node):
            # Add the current node to the flattened list
            flattened.append(node)
            # Recursively traverse each child node
            for child in node.children:
                traverse(child)

        traverse(root)
        return flattened

    def generate_tree_of_thoughts_str(self, context: str, last_user_message: str, knowledge: str, max_depth: int):
        thought_tree = ThoughtNode("root", "root", "")
        self.generate_thought_chain(context, last_user_message, knowledge, "", True, 1, thought_tree, max_depth=max_depth)
        flattened_nodes = self.flatten_thoughts(thought_tree)

        parts = []
        for node in flattened_nodes:
            parts.append(node.content)

        block = "\n".join(parts)
        return block

# --- Configuration Class (Define or integrate into GhostConfig) ---
class MemoryConsolidationConfig(BaseModel):
    included_feature_types: List[FeatureType] = Field(default_factory=lambda: [
        FeatureType.Dialogue, FeatureType.SubjectiveExperience, FeatureType.WorldEvent, FeatureType.Thought
    ])
    topical_clustering_context_window: int = 3
    min_events_for_processing: int = 3  # Min events for a topical cluster to be processed
    min_event_memory_consolidation_threshold: int = 60

    narrative_rolling_tokens_min: int = 3000
    narrative_rolling_tokens_nax: int = 4500
    narrative_rolling_overlap: float = 0.2


# --- Main Consolidator Class ---
class DynamicMemoryConsolidator:

    def __init__(self, ghost: "Ghost", config: MemoryConsolidationConfig):
        self.ghost = ghost
        self.config = config
        self.llm_manager = ghost.llm_manager  # Reuse Ghost's LLM manager

        self.narrative_feature_map = NARRATIVE_FEATURE_RELEVANCE_MAP
        # Make sure narrative_definitions are available, e.g., load from Ghost's config or import
        self.narrative_definitions = narrative_definitions  # Assuming it's imported/available

    # Example adaptation (could be method of DynamicMemoryConsolidator)
    @profile
    def _categorize_fact_knoxel(self, fact_knoxel: DeclarativeFactKnoxel):
        if not fact_knoxel.content: return
        # Replace with your actual categorization logic (e.g., LLM call, keyword matching)
        # Example using cosine_cluster logic from original code:
        try:
            cats = self.categorize_fact(fact_knoxel)
            for cat in cats:
                self.ghost.add_knoxel(cat)
        except Exception as e:
            logger.warning(f"Failed to categorize Fact {fact_knoxel.id}: {e}")

    @profile
    def consolidate_memory_if_needed(self):
        logger.info("Checking if memory consolidation is needed...")
        unclustered_events = self._get_unclustered_event_knoxels()

        if unclustered_events:
            num_unclustered = len(unclustered_events)
            logger.info(f"Unclustered event knoxels: {num_unclustered}. Threshold: {self.config.min_event_memory_consolidation_threshold}")

            if num_unclustered >= self.config.min_event_memory_consolidation_threshold:
                logger.info("Threshold met. Starting memory consolidation...")
                self._perform_consolidation(unclustered_events)
                logger.info("Memory consolidation finished.")
            else:
                logger.info("Threshold not met. Skipping consolidation.")
        else:
            logger.info("No unclustered event knoxels found.")

        self.refine_narratives()

    @profile
    def refine_narratives(self):
        """
        Iterates through defined narratives and refines them if enough relevant new features exist.
        """
        current_tick_id = self.ghost._get_current_tick_id()
        logger.info(f"Starting narrative refinement check (Current Tick: {current_tick_id}).")

        # Get all current Feature knoxels, sorted by tick_id then id
        all_features = sorted(
            [k for k in self.ghost.all_knoxels.values() if isinstance(k, Feature)],
            key=lambda f: f.timestamp_world_begin)


        # Group existing narratives by type and target for easy lookup of the latest one
        latest_narratives: Dict[Tuple[NarrativeTypes, str], Narrative] = {}
        for knoxel in self.ghost.all_knoxels.values():
            if isinstance(knoxel, Narrative):
                key = (knoxel.narrative_type, knoxel.target_name)
                if key not in latest_narratives or knoxel.id > latest_narratives[key].id:
                    latest_narratives[key] = knoxel

        for definition in self.narrative_definitions:
            narrative_type: NarrativeTypes = definition["type"]
            target_name: str = definition["target"]
            prompt_goal: str = definition["prompt"]
            narrative_key = (narrative_type, target_name)

            logger.debug(f"Checking narrative: {narrative_type.name} for {target_name}")

            previous_narrative = latest_narratives.get(narrative_key)
            previous_content = previous_narrative.content if previous_narrative else "No previous narrative."
            last_refined_tick = previous_narrative.last_refined_with_tick if previous_narrative and previous_narrative.last_refined_with_tick is not None else -1
            relevant_feature_types = self.narrative_feature_map[narrative_type]

            # Filter recent features based on type and tick range
            features_since_last_refinement = [
                f for f in all_features
                if f.feature_type in relevant_feature_types and f.causal and f.tick_id > last_refined_tick
            ]
            features_since_last_refinement.sort(key=lambda x: x.timestamp_world_begin)

            logger.debug(f"Found {len(features_since_last_refinement)} relevant features since tick {last_refined_tick}")

            # Check if enough new features exist to warrant refinement
            while True:
                buffer = KnoxelList()
                i = 0
                for i, f in enumerate(features_since_last_refinement):
                    buffer.add(f)
                    if buffer.get_token_count() > int((self.config.narrative_rolling_tokens_min + self.config.narrative_rolling_tokens_nax) / 2):
                        break

                # remove from candidates
                features_since_last_refinement = features_since_last_refinement[i:]
                # no enough?
                if buffer.get_token_count() < self.config.narrative_rolling_tokens_min:
                    break

                logger.info(f"Refining narrative '{narrative_type.name}' for '{target_name}' based on {len(buffer._list)} new features.")
                feature_context = buffer.get_story(self.ghost.config)

                # --- Construct the Refinement Prompt ---
                sysprompt = f"""You are a helpful assistant for memory consolidation in a chatbot with the cognitive architecture based on LIDA. You specialize in creating 'self-narratives'.
You get a long history of events that happened to a chatbot, and you must create a new self-narrative that focuses on a specific aspect: {prompt_goal} """
                user_prompt = f"""
You are refining the psychological/behavioral narrative profile for '{target_name}'.
Narrative Type Goal: "{prompt_goal}"

Current Narrative for '{target_name}' ({narrative_type.name}):
---
{previous_content}
---

New Evidence:
---
{feature_context}
---

Task: Analyze the 'New Evidence'. Based *only* on this new information, provide an *updated and improved* version of the narrative for '{target_name}' regarding '{narrative_type.name}'. Integrate insights, modify conclusions subtly, add new observations, or keep the original narrative section if no new relevant information contradicts or enhances it. Ensure the output is the complete, updated narrative text, fulfilling the Narrative Type Goal.

Output *only* the complete, updated narrative text below. Use no more than 512 tokens!
"""

                assistant_seed = f"""Okay, here is a self-narrative for {target_name} for the {narrative_type.name} category with less than 512 tokens:
                
{target_name} is """

                prompt = [
                    ("system", sysprompt),
                    ("user", user_prompt),
                    ("assistant", assistant_seed)
                ]

                updated_content_raw = f"{target_name} is " + main_llm.completion_text(LlmPreset.Default, prompt, CommonCompSettings(max_tokens=2048, temperature=0.6))

                # --- Process LLM Response ---
                if updated_content_raw:
                    # Clean up potential LLM artifacts like "Updated Narrative:" prefixes
                    updated_content = updated_content_raw.strip()
                    if updated_content.startswith("Updated Narrative:"):
                        updated_content = updated_content[len("Updated Narrative:"):].strip()

                    logger.info(f"Narrative '{narrative_type.name}' for '{target_name}' successfully updated.")
                    # Create the NEW narrative knoxel
                    new_narrative = Narrative(
                        narrative_type=narrative_type,
                        target_name=target_name,
                        content=updated_content,
                        last_refined_with_tick=buffer._list[0].tick_id,  # Mark refinement tick
                        # tick_id, id, timestamp_creation set by add_knoxel
                    )
                    # Add to ghost state (automatically archives the old one by latest ID/tick)
                    # Generate embedding for the new narrative content
                    self.ghost.add_knoxel(new_narrative, generate_embedding=True)
                else:
                    logger.warning(f"LLM call failed for narrative refinement: {narrative_type.name} for {target_name}.")
            # else: logger.debug(f"Not enough new features to refine narrative {narrative_type.name} for {target_name}.") # Too verbose maybe

        logger.info("Narrative refinement check complete.")

    @profile
    def _get_unclustered_event_knoxels(self) -> List[KnoxelBase]:
        """Fetches event-like knoxels not yet part of a Topical cluster."""
        clustered_event_ids: Set[int] = set()
        topical_clusters = [
            k for k in self.ghost.all_knoxels.values()
            if isinstance(k, MemoryClusterKnoxel) and k.cluster_type == ClusterType.Topical
        ]

        for cluster in topical_clusters:
            if cluster.included_event_ids:
                try:
                    ids = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                    clustered_event_ids.update(ids)
                except ValueError:
                    logger.warning(f"Could not parse included_event_ids for cluster {cluster.id}: '{cluster.included_event_ids}'")

        unclustered_knoxels = []
        candidate_knoxels = Enumerable(self.ghost.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.feature_type in self.config.included_feature_types) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        for knoxel in candidate_knoxels:  # Process in ID order
            if knoxel.id not in clustered_event_ids:
                # Ensure embedding exists before adding
                if not knoxel.embedding and knoxel.content:
                    logger.debug(f"Generating missing embedding for unclustered knoxel {knoxel.id}")
                    self.ghost._generate_embedding(knoxel)  # Use ghost's method
                if knoxel.embedding:  # Only include if embedding is present
                    unclustered_knoxels.append(knoxel)
                # else: logger.warning(f"Skipping unclustered knoxel {knoxel.id}: no embedding.")

        logger.debug(f"Found {len(unclustered_knoxels)} unclustered event knoxels with embeddings.")
        return unclustered_knoxels

    @profile
    def _perform_consolidation(self, events_to_cluster: List[KnoxelBase]):
        if len(events_to_cluster) < self.config.min_events_for_processing:
            logger.warning(f"Not enough new events ({len(events_to_cluster)}) for consolidation. Min: {self.config.min_events_for_processing}")
            return

        # 1. Topical Clustering
        new_topical_clusters = self._perform_topical_clustering(events_to_cluster)
        if not new_topical_clusters:
            logger.info("No new topical clusters formed.")
            return

        # 2. Process all but the last cluster
        clusters_to_process = new_topical_clusters[:-1] if len(new_topical_clusters) > 1 else []
        if not clusters_to_process:
            logger.info("Only one new topical cluster formed, skipping processing.")
            return

        logger.info(f"Processing {len(clusters_to_process)} new topical clusters (ignoring last: ID {new_topical_clusters[-1].id}).")

        processed_count = 0
        for topical_cluster in clusters_to_process:
            try:
                # 3. Extract Declarative Memory
                self._extract_declarative_memory(topical_cluster)

                # 4. Update Temporal Hierarchy
                self._update_temporal_hierarchy(topical_cluster, level=6)  # Start at TimeOfDay
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing topical cluster ID {topical_cluster.id}: {e}", exc_info=True)
                # Continue to next cluster

        logger.info(f"Finished processing {processed_count} topical clusters.")

    @profile
    def _perform_topical_clustering(self, event_knoxels: List[KnoxelBase]) -> List[MemoryClusterKnoxel]:
        """Clusters event knoxels contextually and saves them as new Topical Clusters."""
        logger.info(f"Performing topical clustering on {len(event_knoxels)} knoxels...")
        new_cluster_knoxels: List[MemoryClusterKnoxel] = []
        if len(event_knoxels) < 2: return new_cluster_knoxels

        # Compute contextual embeddings
        contextual_embeddings = self._compute_contextual_embeddings(event_knoxels, self.config.topical_clustering_context_window)
        if contextual_embeddings.shape[0] != len(event_knoxels):
            logger.error("Mismatch between knoxel count and contextual embeddings count. Aborting topical clustering.")
            return new_cluster_knoxels

        # Get optimal cluster labels
        cluster_desc = self._get_optimal_clusters(event_knoxels, contextual_embeddings)
        if not cluster_desc.conceptual_cluster:
            logger.warning("Optimal clustering did not produce results.")
            return new_cluster_knoxels

        # Group knoxels by assigned cluster label
        clusters_data: Dict[int, List[KnoxelBase]] = defaultdict(list)
        for knoxel in event_knoxels:
            label = cluster_desc.conceptual_cluster.get(knoxel.id)
            if label is not None:
                clusters_data[label].append(knoxel)
            # else: logger.warning(f"Knoxel {knoxel.id} was not assigned a cluster label.") # Should not happen if cluster_desc is valid

        logger.info(f"Identified {len(clusters_data)} potential topical groups.")

        # Determine the order of cluster creation based on the first knoxel ID in each group
        cluster_creation_order = sorted(clusters_data.keys(), key=lambda k: min(knox.id for knox in clusters_data[k]))

        added_event_ids = set()  # Track which event knoxels get linked

        for cluster_label in cluster_creation_order:
            cluster_knoxels = sorted(clusters_data[cluster_label], key=lambda k: k.id)

            if not cluster_knoxels or len(cluster_knoxels) < self.config.min_events_for_processing:
                logger.debug(f"Skipping topical cluster label {cluster_label} - too few events ({len(cluster_knoxels)}).")
                continue

            # Basic check if events already added (safety)
            if all(knox.id in added_event_ids for knox in cluster_knoxels):
                logger.warning(f"Skipping topical cluster label {cluster_label}, all events seem added already.")
                continue

            logger.info(f"Creating topical cluster for label {cluster_label} with {len(cluster_knoxels)} knoxels (IDs: {cluster_knoxels[0].id}...{cluster_knoxels[-1].id}).")

            # Calculate average embedding, token count, time range, etc.
            total_token = sum(len(k.content) // 4 for k in cluster_knoxels if k.content)  # Crude token estimate
            avg_embedding = np.mean(np.array([k.embedding for k in cluster_knoxels if k.embedding]), axis=0) if any(k.embedding for k in cluster_knoxels) else []
            timestamp_begin = min(k.timestamp_creation for k in cluster_knoxels)
            timestamp_end = max(k.timestamp_creation for k in cluster_knoxels)
            min_id = min(k.id for k in cluster_knoxels)
            max_id = max(k.id for k in cluster_knoxels)
            event_ids_str = ",".join(str(k.id) for k in cluster_knoxels)

            min_tick_id = self.ghost.get_knoxel_by_id(min_id).tick_id
            max_tick_id = self.ghost.get_knoxel_by_id(max_id).tick_id

            # Create the new MemoryClusterKnoxel (Topical)
            topical_cluster = MemoryClusterKnoxel(
                level=100,
                cluster_type=ClusterType.Topical,
                content=f"Topic Cluster Label {cluster_label}",  # Placeholder
                included_event_ids=event_ids_str,
                token=max(1, total_token),  # Ensure token is at least 1
                embedding=avg_embedding.tolist() if isinstance(avg_embedding, np.ndarray) else [],
                timestamp_world_begin=timestamp_begin,
                timestamp_world_end=timestamp_end,
                min_event_id=min_id,
                max_event_id=max_id,
                min_tick_id=min_tick_id,
                max_tick_id=max_tick_id,
                facts_extracted=False,
                # Inherits id, tick_id (current tick), timestamp_creation from add_knoxel
            )

            # Add to Ghost state - This assigns ID, tick_id, timestamp_creation
            # We generate embedding based on summary later if needed, or maybe average is enough?
            # For topical clusters, embedding might be less crucial than for temporal summaries.
            self.ghost.add_knoxel(topical_cluster, generate_embedding=False)  # Don't embed placeholder summary
            new_cluster_knoxels.append(topical_cluster)

            # Track added event IDs
            for knox in cluster_knoxels:
                added_event_ids.add(knox.id)

        logger.info(f"Created {len(new_cluster_knoxels)} new topical clusters.")
        return new_cluster_knoxels

    @profile
    def _extract_declarative_memory(self, topical_cluster_knoxel: MemoryClusterKnoxel):
        """Extracts Facts and Cause/Effect from a topical cluster."""
        if topical_cluster_knoxel.facts_extracted: return
        if not topical_cluster_knoxel.included_event_ids: return

        logger.info(f"Extracting declarative memory for topical cluster {topical_cluster_knoxel.id}")

        event_ids = [int(eid) for eid in topical_cluster_knoxel.included_event_ids.split(',') if eid]
        event_knoxels = [self.ghost.get_knoxel_by_id(eid) for eid in event_ids if self.ghost.get_knoxel_by_id(eid)]
        if not event_knoxels: return

        # Reconstruct the text block (assuming event knoxels have source/content)
        lines = []
        for k in sorted(event_knoxels, key=lambda x: x.id):
            source = getattr(k, 'source', 'Unknown')  # Handle knoxels without source
            content = getattr(k, 'content', '')
            lines.append(f"{source}: {content}")
        block = "\n".join(lines)

        min_id = topical_cluster_knoxel.min_event_id
        max_id = topical_cluster_knoxel.max_event_id

        min_tick_id = self.ghost.get_knoxel_by_id(min_id).tick_id
        max_tick_id = self.ghost.get_knoxel_by_id(max_id).tick_id

        # --- Extract Facts ---
        try:
            extracted_raw_facts = self.extract_facts(block)  # Returns List[Tuple[str, str]]
            logger.info(f"Extracted {len(extracted_raw_facts)} potential facts from cluster {topical_cluster_knoxel.id}.")
            for fact_content, temporality in extracted_raw_facts:
                new_fact = DeclarativeFactKnoxel(
                    content=fact_content,
                    temporality=temporality,
                    min_event_id=min_id,
                    max_event_id=max_id,
                    min_tick_id=min_tick_id,
                    max_tick_id=max_tick_id,
                    source_cluster_id=topical_cluster_knoxel.id,
                    # category will be set by _categorize_fact_knoxel
                )
                # Add_knoxel assigns ID, calculates token, generates embedding
                self.ghost.add_knoxel(new_fact, generate_embedding=True)
                self._categorize_fact_knoxel(new_fact)  # Categorize and potentially save category update
        except Exception as e:
            logger.error(f"Error during fact extraction for cluster {topical_cluster_knoxel.id}: {e}", exc_info=True)

        # --- Extract Cause/Effect ---
        if min_tick_id == -1337:
            cea = CauseEffectAnalyzer()
            extracted_cas: List[CauseEffectItem] = cea.extract_cause_effect(block)
            logger.info(f"Extracted {len(extracted_cas)} potential cause/effect pairs from cluster {topical_cluster_knoxel.id}.")
            for ca_schema in extracted_cas:
                # Optional: Categorize CA
                ca_str = f"Cause: {ca_schema.cause} -> Effect: {ca_schema.effect}"
                category = cea.categorize_cause_effect(ca_str) # Use helper
                new_ca = CauseEffectKnoxel(
                    cause=ca_schema.cause,
                    effect=ca_schema.effect,
                    content=ca_str,
                    category=category,
                    min_event_id=min_id,
                    max_event_id=max_id,
                    min_tick_id=min_tick_id,
                    max_tick_id=max_tick_id,
                    source_cluster_id=topical_cluster_knoxel.id,
                )
                self.ghost.add_knoxel(new_ca, generate_embedding=True)

        # Mark cluster as processed
        topical_cluster_knoxel.facts_extracted = True
        self.ghost.add_knoxel(topical_cluster_knoxel, generate_embedding=False)  # Save the flag change

    @profile
    def _update_temporal_hierarchy(self, input_cluster_knoxel: MemoryClusterKnoxel, level: int):
        """Recursively finds/creates/updates temporal clusters."""
        if level < 1: return

        logger.debug(f"Updating temporal hierarchy at level {level} for input cluster {input_cluster_knoxel.id} ({input_cluster_knoxel.cluster_type.name})")

        mid_timestamp = input_cluster_knoxel.timestamp_world_begin + (input_cluster_knoxel.timestamp_world_end - input_cluster_knoxel.timestamp_world_begin) / 2
        temporal_key = self.get_temporal_key(mid_timestamp, level)

        # Find existing temporal cluster at this level/key in Ghost's state
        existing_cluster: Optional[MemoryClusterKnoxel] = None
        for k in self.ghost.all_knoxels.values():
            if isinstance(k, MemoryClusterKnoxel) and \
                    k.level == level and \
                    k.cluster_type == ClusterType.Temporal and \
                    k.temporal_key == temporal_key:
                existing_cluster = k
                break

        target_cluster: Optional[MemoryClusterKnoxel] = None
        needs_resummarize = False
        updated_or_created = False

        if existing_cluster:
            logger.debug(f"Found existing level {level} cluster {existing_cluster.id} for key {temporal_key}.")
            target_cluster = existing_cluster
            current_ids = set(int(cid) for cid in target_cluster.included_cluster_ids.split(',') if cid) if target_cluster.included_cluster_ids else set()

            if input_cluster_knoxel.id not in current_ids:
                current_ids.add(input_cluster_knoxel.id)
                target_cluster.included_cluster_ids = ",".join(map(str, sorted(list(current_ids))))
                needs_resummarize = True
                updated_or_created = True
                logger.debug(f"Added input cluster {input_cluster_knoxel.id} to existing {target_cluster.id}.")

            # Expand time range
            time_updated = False
            if input_cluster_knoxel.timestamp_world_begin < target_cluster.timestamp_world_begin:
                target_cluster.timestamp_world_begin = input_cluster_knoxel.timestamp_world_begin
                time_updated = True
            if input_cluster_knoxel.timestamp_world_end > target_cluster.timestamp_world_end:
                target_cluster.timestamp_world_end = input_cluster_knoxel.timestamp_world_end
                time_updated = True

            if updated_or_created or time_updated:
                logger.info(f"Updating existing level {level} cluster {target_cluster.id} (Key: {temporal_key}). Includes {len(current_ids)} clusters.")
                # No need to call add_knoxel yet, changes are in memory. Wait until after potential summary.
            else:
                logger.debug(f"No updates needed for existing cluster {target_cluster.id} based on input {input_cluster_knoxel.id}.")

        else:
            logger.info(f"Creating new level {level} temporal cluster for key {temporal_key}.")
            min_tick_id = self.ghost.get_knoxel_by_id(input_cluster_knoxel.min_event_id).tick_id
            max_tick_id = self.ghost.get_knoxel_by_id(input_cluster_knoxel.max_event_id).tick_id

            new_cluster = MemoryClusterKnoxel(
                level=level,
                cluster_type=ClusterType.Temporal,
                content="",  # Will generate
                included_cluster_ids=str(input_cluster_knoxel.id),
                token=0,
                embedding=[],
                timestamp_world_begin=input_cluster_knoxel.timestamp_world_begin,
                timestamp_world_end=input_cluster_knoxel.timestamp_world_end,
                min_event_id=input_cluster_knoxel.min_event_id,  # Approximate initially
                max_event_id=input_cluster_knoxel.max_event_id,  # Approximate initially
                min_tick_id=min_tick_id,
                max_tick_id=max_tick_id,
                temporal_key=temporal_key,
                # Inherits id, tick_id, etc. from add_knoxel
            )
            self.ghost.add_knoxel(new_cluster, generate_embedding=False)  # Add to get ID
            target_cluster = new_cluster
            needs_resummarize = True
            updated_or_created = True
            logger.debug(f"Created new temporal cluster {target_cluster.id}.")

        # Re-summarize if needed
        if needs_resummarize and target_cluster:
            logger.info(f"Re-summarizing cluster {target_cluster.id} (Level {level}, Key: {temporal_key})...")
            summary_content = self._generate_temporal_summary(target_cluster)
            target_cluster.content = summary_content
            target_cluster.token = len(summary_content) // 4  # Crude token estimate
            # Embedding should be generated for the summary
            # Set min/max based on children
            min_ev_id, max_ev_id = self._get_min_max_event_id_for_temporal(target_cluster)
            target_cluster.min_event_id = min_ev_id
            target_cluster.max_event_id = max_ev_id

            target_cluster.min_tick_id = self.ghost.get_knoxel_by_id(min_ev_id).tick_id
            target_cluster.max_tick_id = self.ghost.get_knoxel_by_id(max_ev_id).tick_id

            logger.debug(f"Summary generated for {target_cluster.id}. Est. Token: {target_cluster.token}. Event range: {min_ev_id}-{max_ev_id}")
            # Save the updated cluster with summary (and generate embedding)
            self.ghost.add_knoxel(target_cluster, generate_embedding=True)

        def level_to_parent_level(level: int) -> Optional[int]:
            # ... (implementation from previous response) ...
            if level > 1: return level - 1
            return None

        # Recurse if a cluster was created or updated
        parent_level = level_to_parent_level(level)
        if parent_level and target_cluster and updated_or_created:
            logger.debug(f"Propagating update from level {level} cluster {target_cluster.id} to parent level {parent_level}.")
            self._update_temporal_hierarchy(target_cluster, parent_level)

    @profile
    def _generate_temporal_summary(self, temporal_cluster_knoxel: MemoryClusterKnoxel) -> str:
        """Generates a summary for a temporal cluster based on included content."""
        if not temporal_cluster_knoxel.included_cluster_ids: return f"Empty temporal cluster {temporal_cluster_knoxel.id}."

        included_cluster_ids = [int(cid) for cid in temporal_cluster_knoxel.included_cluster_ids.split(',') if cid]
        included_clusters: List[MemoryClusterKnoxel] = []
        for cid in included_cluster_ids:
            knoxel = self.ghost.get_knoxel_by_id(cid)
            if isinstance(knoxel, MemoryClusterKnoxel):
                included_clusters.append(knoxel)
            # else: logger.warning(f"Could not find included cluster knoxel with ID {cid}") # Maybe removed?

        if not included_clusters: return f"Temporal cluster {temporal_cluster_knoxel.id} with no valid included clusters found."

        summary_input_lines = []
        included_clusters.sort(key=lambda c: c.timestamp_world_begin)  # Sort children by time

        for cluster in included_clusters:
            if cluster.cluster_type == ClusterType.Temporal and cluster.content:
                summary_input_lines.append(f"Summary ({cluster.timestamp_world_begin.strftime('%H:%M')}): {cluster.content}")
            elif cluster.cluster_type == ClusterType.Topical and cluster.included_event_ids:
                event_ids = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                event_knoxels = [self.ghost.get_knoxel_by_id(eid) for eid in event_ids if self.ghost.get_knoxel_by_id(eid)]
                if event_knoxels:
                    # Get first few lines of the topic
                    topic_lines = [f"{getattr(k, 'source', '?')}: {getattr(k, 'content', '')[:50]}..." for k in sorted(event_knoxels, key=lambda x: x.id)[:3]]
                    summary_input_lines.append(f"Topic ({cluster.timestamp_world_begin.strftime('%H:%M')} between events {cluster.min_event_id}-{cluster.max_event_id}):\n  " + "\n  ".join(topic_lines))

        if not summary_input_lines: return f"Temporal cluster {temporal_cluster_knoxel.id} - No content found in children."

        # --- Call LLM for Summarization ---
        # Reuse summarize_messages or create a dedicated prompt
        full_summary_input = "\n".join(summary_input_lines)
        prompt = f"Summarize the following context from a dialogue-heavy story. Be concise and summarize it without any special syntax as single paragraph! \n\nContext:\n{full_summary_input}"
        comp_settings = CommonCompSettings(temperature=0.5, max_tokens=512)  # Adjust settings
        # Use _call_llm method from Ghost or llm_manager directly
        final_summary = self.ghost._call_llm(prompt)  # Assuming _call_llm exists on Ghost
        return final_summary if final_summary else f"[Summary generation failed for cluster {temporal_cluster_knoxel.id}]"

    @profile
    def _get_min_max_event_id_for_temporal(self, temporal_cluster_knoxel: MemoryClusterKnoxel) -> Tuple[Optional[int], Optional[int]]:
        """Finds min/max original event IDs covered by a temporal cluster's children."""
        min_id: Optional[int] = None
        max_id: Optional[int] = None
        if not temporal_cluster_knoxel.included_cluster_ids: return min_id, max_id

        included_cluster_ids = [int(cid) for cid in temporal_cluster_knoxel.included_cluster_ids.split(',') if cid]
        for cid in included_cluster_ids:
            child_knoxel = self.ghost.get_knoxel_by_id(cid)
            if isinstance(child_knoxel, MemoryClusterKnoxel):
                if child_knoxel.min_event_id is not None:
                    min_id = min(min_id, child_knoxel.min_event_id) if min_id is not None else child_knoxel.min_event_id
                if child_knoxel.max_event_id is not None:
                    max_id = max(max_id, child_knoxel.max_event_id) if max_id is not None else child_knoxel.max_event_id

        return min_id, max_id

    # Inside DynamicMemoryConsolidator or as a standalone function taking knoxels
    @profile
    def _compute_contextual_embeddings(self, event_knoxels: List[KnoxelBase], context_window: int) -> np.ndarray:
        if not event_knoxels: return np.array([])
        # Assume knoxels have 'embedding' and calculate 'token' if not present
        # TODO: Ensure knoxels passed here *have* embeddings. Pre-filter or generate?
        embeddings = []
        tokens = []
        valid_knoxels = []
        for k in event_knoxels:
            if k.embedding:
                embeddings.append(k.embedding)
                token_count = get_token_count(k.content)
                tokens.append(max(1, token_count))
                valid_knoxels.append(k)
            else:
                logger.warning(f"Skipping knoxel {k.id} in contextual embedding calculation: missing embedding.")

        if not valid_knoxels: return np.array([])

        embeddings = np.array(embeddings)
        tokens = np.array(tokens)
        contextual_embeddings = []

        for i in range(len(valid_knoxels)):
            # Using simple token count as weight proxy - adjust if needed
            current_token_weight = np.log(1 + tokens[i])
            current_temporal_weight = 1  # Base weight for the item itself
            weight = current_temporal_weight * current_token_weight

            combined_embedding = embeddings[i] * weight
            weight_sum = weight

            start_idx = max(0, i - context_window)
            end_idx = min(len(valid_knoxels), i + context_window + 1)

            for j in range(start_idx, end_idx):
                if i == j: continue

                context_token_weight = np.log(1 + tokens[j])
                # Use index difference as simple temporal distance proxy
                context_temporal_weight = 1 / (1 + abs(i - j))
                context_weight = context_temporal_weight * context_token_weight

                combined_embedding += context_weight * embeddings[j]
                weight_sum += context_weight

            if weight_sum > 1e-6:  # Avoid division by zero
                combined_embedding /= weight_sum
            else:  # Should not happen if valid_knoxels is not empty
                combined_embedding = embeddings[i]  # Fallback to original

            contextual_embeddings.append(combined_embedding)

        return np.array(contextual_embeddings)

    @profile
    def _get_optimal_clusters(self, event_knoxels: List[KnoxelBase], contextual_embeddings: np.ndarray) -> ClusterDescription:
        num_events = len(event_knoxels)
        if num_events < 2 or contextual_embeddings.shape[0] != num_events:
            logger.warning(f"Insufficient data for clustering. Events: {num_events}, Embeddings shape: {contextual_embeddings.shape}")
            return ClusterDescription()

        try:
            # Calculate cosine distance matrix from contextual embeddings
            cosine_sim_matrix = cosine_similarity(contextual_embeddings)
            # Ensure distances are non-negative and handle potential floating point issues
            cosine_dist_matrix = np.maximum(0, 1 - cosine_sim_matrix)
            np.fill_diagonal(cosine_dist_matrix, 0)  # Ensure zero distance to self

            max_metric = -1.1  # Initialize below silhouette's range [-1, 1]
            best_threshold = None
            best_labels = None

            # Iterate through potential distance thresholds (adjust range/step as needed)
            # Start coarser, get finer near potential optimum? Or use n_clusters search?
            # Let's stick to distance threshold for now.
            for threshold in np.linspace(1.0, 0.1, 91):  # Example: from 1.0 down to 0.1
                if threshold <= 0: continue  # Skip zero or negative threshold

                agg_clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    metric='precomputed',
                    linkage='average',  # Average linkage is often good for text topics
                    compute_full_tree=True
                )
                labels = agg_clustering.fit_predict(cosine_dist_matrix)
                n_clusters_found = len(np.unique(labels))

                # Calculate silhouette score only if we have a valid number of clusters
                if max(2, int(num_events / 20)) < n_clusters_found < min(num_events, int(num_events / 8)):
                    try:
                        # Use cosine *distance* for silhouette score
                        metric = silhouette_score(cosine_dist_matrix, labels, metric='precomputed')
                        if metric > max_metric:
                            max_metric = metric
                            best_threshold = threshold
                            best_labels = labels
                            logger.debug(f"New best silhouette score: {metric:.4f} at threshold {threshold:.3f} ({n_clusters_found} clusters)")
                    except ValueError as e:
                        logger.warning(f"Silhouette score error at threshold {threshold:.3f} ({n_clusters_found} clusters): {e}")
                        metric = -1.1  # Invalid score
                else:
                    # logger.debug(f"Skipping silhouette score: Threshold {threshold:.3f} -> {n_clusters_found} clusters")
                    metric = -1.1

            # If no suitable clustering found, return empty
            if best_labels is None:
                logger.warning("Could not find an optimal clustering threshold.")
                # Fallback: assign all to one cluster? or return empty? Let's return empty.
                return ClusterDescription()
                # Fallback 2: Assign all to cluster 0
                # best_labels = np.zeros(num_events, dtype=int)

            logger.info(f"Optimal clustering found: Threshold {best_threshold:.3f}, Score {max_metric:.4f}, Clusters {len(np.unique(best_labels))}")

            # Optional: Smooth labels (reduce single-event clusters between same-label clusters)
            final_labels = best_labels.copy()
            for i in range(1, num_events - 1):
                if final_labels[i] != final_labels[i - 1] and final_labels[i] != final_labels[i + 1] and final_labels[i - 1] == final_labels[i + 1]:
                    final_labels[i] = final_labels[i - 1]  # Assign to surrounding cluster label

            res = ClusterDescription()
            for i, knoxel in enumerate(event_knoxels):
                res.conceptual_cluster[knoxel.id] = int(final_labels[i])  # Ensure integer label

            return res
        except Exception as e:
            logger.error(f"Error during optimal cluster calculation: {e}", exc_info=True)
            return ClusterDescription()

    # --- Add adaptations of cluster_utils / sleep_procedures helpers as needed ---
    # _compute_contextual_embeddings(self, event_knoxels: List[KnoxelBase], context_window: int) -> np.ndarray: ...
    # _get_optimal_clusters(self, event_knoxels: List[KnoxelBase], contextual_embeddings: np.ndarray) -> ClusterDescription: ...
    # _categorize_fact_knoxel(self, fact_knoxel: DeclarativeFactKnoxel): ...

    def categorize_fact(self, k: KnoxelBase) -> List[FactCategory]:
        sysprompt = f"""You are a helpful assistant that helps organize facts from a story about {companion_name}, an AI and her User, {user_name}, into categories. These categories will help the author not forget stuff.
        You do this by creating a JSON where you rate the belongingness to different categories with a float between 0 and 1.    
        """

        def init_basemodel() -> Type[BaseModel]:
            model_definitions = []
            for node in memory_schema:
                model_definitions.append({
                    f"name": node["path"],
                    f"type": float,
                    f"description": node["description"]
                })

            DynamicModel = create_basemodel(model_definitions, randomnize_entries=True)
            DynamicModel.__doc__ = f"Use this JSON schema to rate the belonging of facts to different categories. Use 0 for no belonging and 1 for maximum belonging."
            return DynamicModel

        res = []
        bm = init_basemodel()
        messages = [
            ("system", sysprompt),
            ("user", f"Can you put this fact into the different categories? ### BEGIN FACT\n{k.content}\n### END FACT")
        ]

        _, calls = main_llm.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[bm])

        mb = calls[0]
        max_weight = 0
        for f in mb.__class__.model_fields_set:
            max_weight = max(max_weight, getattr(mb, f))

        for f in mb.__class__.model_fields_set:
            weight = getattr(mb, f)
            cat_id = path_to_id.get(f, 0)
            fc = FactCategory(
                content="",
                fact_id=k.id,
                category=f,
                category_id=cat_id,
                weight=weight,
                max_weight=max_weight)
            res.append(fc)
        return res

    def extract_facts(self, block: str):
        sysprompt = "You are a helpful assistant. You extract facts from a conversation."
        prompt = [
            ("system", sysprompt),
            ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n"),
        ]

        _, calls = main_llm.completion_tool(LlmPreset.Default, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[Facts])
        facts = []
        for fact in calls[0].facts:
            sysprompt = "You are a helpful assistant. You extract determine if a fact is time-variant or time-invariant. Use 0 for facts that are true at all times, and 1 for facts only true right now."
            prompt = [
                ("system", sysprompt),
                ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n### BEGIN FACT\n{fact}\n### END FACT\nIs the extracted fact from the messages time-independent (0) or time-dependent (1)?"),
            ]

            _, calls2 = main_llm.completion_tool(LlmPreset.Default, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[TimeInvariance])
            time_variance = calls2[0].time_variance
            facts.append((fact, time_variance))

        return facts

    # --- Temporal Key Generation Functions (Keep these) ---
    def get_temporal_key(self, timestamp: datetime, level: int) -> str:
        # ... (implementation from previous response) ...
        if level == 6:  # Time Of Day (combining with day for uniqueness)
            hour = timestamp.hour
            tod = "NIGHT"  # Default
            if 5 <= hour < 12:
                tod = "MORNING"
            elif 12 <= hour < 14:
                tod = "NOON"
            elif 14 <= hour < 18:
                tod = "AFTERNOON"
            elif 18 <= hour < 22:
                tod = "EVENING"
            return f"{timestamp.strftime('%Y-%m-%d')}-{tod}"
        elif level == 5:
            return timestamp.strftime('%Y-%m-%d')  # Day
        elif level == 4:
            return timestamp.strftime('%Y-W%W')  # Week
        elif level == 3:
            return timestamp.strftime('%Y-%m')  # Month
        elif level == 2:  # Season
            season = (timestamp.month % 12 + 3) // 3
            return f"{timestamp.year}-S{season}"
        elif level == 1:
            return str(timestamp.year)  # Year
        else:
            raise ValueError(f"Invalid temporal level: {level}")


# --- GhostState (Unchanged from previous version) ---
class GhostState(BaseModel):
    class Config: arbitrary_types_allowed = True

    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)

    primary_stimulus: Optional[Stimulus] = None
    attention_candidates: KnoxelList = KnoxelList()
    coalitions_hard: Dict[int, List[KnoxelBase]] = {}
    coalitions_balanced: Dict[int, List[KnoxelBase]] = {}
    attention_focus: KnoxelList = KnoxelList()
    conscious_workspace: KnoxelList = KnoxelList()
    subjective_experience: Optional[Feature] = None
    subjective_experience_tool: Optional[Feature] = None

    # Action Deliberation & Simulation Results
    action_simulations: List[Dict[str, Any]] = []  # Store results of MC simulations [{sim_id, type, content, rating, predicted_state, user_reaction}, ...]
    selected_action_details: Optional[Dict[str, Any]] = None  # Details of the chosen simulation/action before execution
    selected_action_knoxel: Optional[Action] = None  # The final Action knoxel generated (set in execute)

    # user rating 0 neutral, -1 bad, 1 good
    rating: int = Field(default=0)

    # State snapshots
    state_emotions: EmotionalAxesModel = EmotionalAxesModel()
    state_needs: NeedsAxesModel = NeedsAxesModel()
    state_cognition: CognitionAxesModel = CognitionAxesModel()


class MLevel(StrEnum):
    Debug = "Debug"
    Low = "Low"
    Mid = "Mid"
    High = "High"

# --- Main Cognitive Cycle Class ---
class Ghost:
    def __init__(self, config: GhostConfig):
        self.stimulus_triage: StimulusTriage = StimulusTriage.Moderate
        self.config = config

        self.current_tick_id: int = 0
        self.current_knoxel_id: int = 0

        self.all_knoxels: Dict[int, KnoxelBase] = {}
        self.all_features: List[Feature] = []
        self.all_stimuli: List[Stimulus] = []
        self.all_intentions: List[Intention] = []  # Includes internal goals AND external expectations
        self.all_narratives: List[Narrative] = []
        self.all_actions: List[Action] = []
        self.all_episodic_memories: List[MemoryClusterKnoxel] = []
        self.all_declarative_facts: List[DeclarativeFactKnoxel] = []

        self.states: List[GhostState] = []
        self.current_state: Optional[GhostState] = None
        self.simulated_reply: Optional[str] = None

        self._initialize_narratives()
        self.llm_manager = main_llm

        self.memory_config = MemoryConsolidationConfig()
        self.memory_consolidator = DynamicMemoryConsolidator(self, self.memory_config)  # Pass self (ghost instance)
        self.meta_insights = []

    @property
    def all_knoxels_list(self):
        tmp = list(self.all_knoxels.values())
        tmp.sort(key=lambda x: x.timestamp_world_begin)
        return tmp

    def _get_state_at_tick(self, tick_id: int) -> GhostState | None:
        if tick_id == self.current_tick_id:
            return self.current_state

        for s in self.states:
            if s.tick_id == tick_id:
                return s
        return None

    def _get_next_id(self) -> int:
        self.current_knoxel_id += 1
        return self.current_knoxel_id

    def _get_current_tick_id(self) -> int:
        return self.current_tick_id

    def _initialize_narratives(self):
        # Ensure default narratives exist for AI and User based on definitions
        existing = {(n.narrative_type, n.target_name) for n in self.all_narratives}
        for definition in narrative_definitions:
            if (definition["type"], definition["target"]) not in existing:
                default_content = f"Initial placeholder narrative for {definition['target']} regarding {definition['type']}."
                if definition["target"] == self.config.companion_name:
                    default_content = self.config.universal_character_card  # Use character card for initial self-image
                elif definition["target"] == self.config.user_name:
                    default_content = f"{self.config.user_name} is the user interacting with {self.config.companion_name}."

                narrative = Narrative(
                    narrative_type=definition["type"],
                    target_name=definition["target"],
                    content=default_content,
                )
                self.add_knoxel(narrative, generate_embedding=False)  # Embeddings generated on demand or during learning
        logging.info(f"Ensured {len(self.all_narratives)} initial narratives exist.")

    def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True):
        if knoxel.id == -1: knoxel.id = self._get_next_id()
        if knoxel.tick_id == -1: knoxel.tick_id = self._get_current_tick_id()

        self.all_knoxels[knoxel.id] = knoxel

        self._log_mental_mechanism(self.add_knoxel, MLevel.Debug, f"Add knoxel to state: {knoxel}")

        # Add to specific lists
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        elif isinstance(knoxel, Stimulus):
            self.all_stimuli.append(knoxel)
        elif isinstance(knoxel, Intention):
            # Check for duplicates based on content and type? For now, allow multiple.
            self.all_intentions.append(knoxel)
        elif isinstance(knoxel, Narrative):
            # Remove older narrative of same type/target before adding new one
            self.all_narratives = [n for n in self.all_narratives if not (n.narrative_type == knoxel.narrative_type and n.target_name == knoxel.target_name)]
            self.all_narratives.append(knoxel)
        elif isinstance(knoxel, Action):
            self.all_actions.append(knoxel)
        elif isinstance(knoxel, MemoryClusterKnoxel):
            self.all_episodic_memories.append(knoxel)
        elif isinstance(knoxel, DeclarativeFactKnoxel):
            self.all_declarative_facts.append(knoxel)

        # Generate embedding if requested and not present
        if generate_embedding and not knoxel.embedding and knoxel.content:
            self._generate_embedding(knoxel)

    def get_knoxel_by_id(self, knoxel_id: int) -> Optional[KnoxelBase]:
        return self.all_knoxels.get(knoxel_id)

    # Add helper to get latest narrative
    def get_narrative(self, narrative_type: NarrativeTypes, target_name: str) -> Optional[Narrative]:
        """Gets the most recent narrative knoxel of a specific type and target."""
        return Enumerable(self.all_narratives).last_or_default(
            lambda n: n.narrative_type == narrative_type and n.target_name == target_name
        )

    def _generate_embedding(self, knoxel: KnoxelBase):
        if not knoxel.content: return
        try:
            response = self.llm_manager.get_embedding(knoxel.content)
            knoxel.embedding = response
        except Exception as e:
            logging.error(f"Failed to generate embedding for Knoxel {knoxel.id}: {e}")

    def _call_llm(self, prompt: str, output_schema: Optional[Type[BM]] = None, stop_words: List[str] | None = None, temperature: float = 0.6, max_tokens: int = 1024,
                  sysprompt: str = None, seed_phrase: str = None) -> str | BM | float | int | bool | None:
        max_tokens = 4096  # Keep higher default

        """ Wraps LLM call, handling schema validation or text output. """
        if stop_words is None: stop_words = []

        if not sysprompt:
            sysprompt = default_agent_sysprompt

        msgs = [("system", sysprompt),
                ("user", prompt)]

        if seed_phrase is not None and seed_phrase.strip() != "":
            msgs.append(("assistant", seed_phrase))

        comp_settings = CommonCompSettings(temperature=temperature, repeat_penalty=1.05, max_tokens=max_tokens, stop_words=stop_words)

        try:
            if output_schema is not None:
                _, calls = self.llm_manager.completion_tool(LlmPreset.Default, msgs, comp_settings=comp_settings, tools=[output_schema])
                if calls:
                    # Basic validation check before returning
                    if isinstance(calls[0], output_schema):
                        return calls[0]
                    else:
                        logging.error(f"LLM tool call returned unexpected type: {type(calls[0])}. Expected {output_schema}. Payload: {calls[0]}")
                        # Try to parse manually if it looks like a dict
                        if isinstance(calls[0], dict):
                            try:
                                return output_schema(**calls[0])
                            except ValidationError as ve:
                                logging.error(f"Pydantic validation failed on manual parse: {ve}")
                                return None
                        return None
                else:
                    logging.warning(f"LLM tool call for {output_schema.__name__} returned no calls.")
                    return None
            else:
                res = self.llm_manager.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
                # Simple cleaning for text responses
                return res.strip() if res else None
        except ValidationError as e:
            logging.error(f"Pydantic validation error processing LLM output for schema {output_schema.__name__ if output_schema else 'N/A'}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during LLM call: {e}", exc_info=True)
            return None

    def _get_relevant_knoxels(self, query_embedding: List[float], knoxel_source: List[T], limit: int) -> List[T]:
        if not query_embedding or not knoxel_source: return []
        candidates = [k for k in knoxel_source if k.embedding]
        if not candidates: return []
        query_embedding_np = np.array(query_embedding)
        # Use batch embedding retrieval if possible, otherwise loop
        embeddings_np = np.array([k.embedding for k in candidates])
        try:
            distances = [cosine_distance(query_embedding_np, emb) for emb in embeddings_np]
            sorted_candidates = sorted(zip(distances, candidates), key=lambda x: x[0])
            return [k for dist, k in sorted_candidates[:limit]]
        except ValueError as e:
            logging.error(f"Error calculating cosine distances (likely shape mismatch): {e}")
            # Fallback: calculate one by one
            distances = []
            for k in candidates:
                try:
                    dist = cosine_distance(query_embedding_np, np.array(k.embedding))
                    distances.append((dist, k))
                except ValueError:
                    logging.warning(f"Skipping knoxel {k.id} due to embedding dimension mismatch.")
                    distances.append((float('inf'), k))  # Put problematic ones last
            sorted_candidates = sorted(distances, key=lambda x: x[0])
            return [k for dist, k in sorted_candidates[:limit]]

    def reconsolidate_memory(self):
        self.memory_consolidator.consolidate_memory_if_needed()

    def kill(self):
        pass

    # --- Tick Method and Sub-functions ---
    def tick(self, stimulus: Stimulus) -> Action:
        self._log_mental_mechanism(self.tick, MLevel.Mid, f"--- Starting Tick {self.current_tick_id + 1} ---")
        self.current_tick_id += 1

        # 1. Initialize State (Handles decay, internal stimulus gen, carries over intentions/expectations)
        self._initialize_tick_state(stimulus)
        if not self.current_state:
            raise Exception("Empty state!")

        self.stimulus_triage = self._triage_stimulus(stimulus)
        self._appraise_stimulus()

        if self.stimulus_triage in [StimulusTriage.Moderate, StimulusTriage.Significant]:
            # appraise
            self._generate_short_term_intentions()
            self._gather_memories_for_attention()
            self._gather_meta_insights()

            # attention
            self._build_structures_get_coalitions()
            self._simulate_attention_on_coalitions()
            self._generate_subjective_experience()
        else:
            self.current_state.conscious_workspace = KnoxelList()

        # 6. Action Deliberation & Selection
        chosen_action_type = self._determine_best_action_type()
        if chosen_action_type == ActionType.Reply:
            chosen_action_details = self._deliberate_and_select_reply()
            self.current_state.selected_action_details = chosen_action_details
        elif chosen_action_type == ActionType.ToolCallAndReply:
            chosen_action_details = self._perform_tool_call_and_reply()
            self.current_state.selected_action_details = chosen_action_details
        elif chosen_action_type == ActionType.Ignore:
            pass

        # 7. Execute Action (Generates expectations)
        action = self._execute_action()  # ***MODIFIED***

        # 8. Learning / Memory Consolidation (Can use expectation outcomes)
        # self._perform_learning_and_consolidation() # ***MODIFIED***
        # self.memory_consolidator.consolidate_memory_if_needed()
        if self.stimulus_triage in [StimulusTriage.Moderate, StimulusTriage.Significant]:
            self._render_meta_insights()

        self.states.append(self.current_state)
        self._log_mental_mechanism(self.tick, MLevel.Mid, f"--- Finished Tick {self.current_tick_id} ---")
        return action  # Return the final output string (reply) or None

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Returns a copy of the entire conversation history.
        This is the clean public interface for retrieving the chat log.
        """
        res = []
        for f in self.all_features:
            tick_rating = 0
            for s in self.states:
                if s.tick_id == f.tick_id:
                    tick_rating = s.rating
                    break

            if f.feature_type == FeatureType.Dialogue:
                d = {"content": f.content, "tick_id": f.tick_id, "feature_id": f.id, "rating": tick_rating}
                if f.source == user_name:
                    d["role"] = "user"
                elif f.source == companion_name:
                    d["role"] = "assistant"
                res.append(d)
        return res

    def _log_mental_mechanism(self, func, level: MLevel, msg: str):
        meta_item = {
            "ts": datetime.now(),
            "func": func.__name__,
            "level": level.value,
            "msg": msg
        }
        logger.info(msg)
        if level in [MLevel.Mid, MLevel.High, MLevel.Low]:
            self.meta_insights.append(meta_item)

    def _render_meta_insights(self):

        tmp = []
        for cnt, item in enumerate(self.meta_insights):
            log = item["msg"].replace("\n", "")
            tmp.append(f"CNT: {cnt} LOG: {log}")

        sysprompt = f"""{default_agent_sysprompt}
                    Your task is: Summarize what happens in compact format, in technical detail, so the AI can understand and explain its reasoning.
                    """

        full = "\n".join(tmp)
        user_prompt = f"### BEGIN LIDA LOG{full}### END LIDA LOG"

        logger.info(user_prompt)

        seed = f'''Okay, here's a summary of what happened during execution of this tick, suitable for {companion_name} to explain their emotional/reasoning mechanisms of her mind in technical detail:

"'''
        msgs = [("system", sysprompt),
                ("user", user_prompt),
                ("assistant", seed)]

        comp_settings = CommonCompSettings(temperature=0.5, repeat_penalty=1.05, max_tokens=1024)
        res = '"' + self.llm_manager.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
        story_feature = Feature(content=res, source=companion_name, feature_type=FeatureType.MetaInsight, interlocus=-2, causal=True)
        self.add_knoxel(story_feature)


    @profile
    def _initialize_tick_state(self, stimulus: Stimulus):
        """Sets up the GhostState for the current tick, handling state decay,
           carrying over unfulfilled intentions AND expectations, and generating internal stimuli."""
        previous_state = self.states[-1] if self.states else None
        self.current_state = GhostState(tick_id=self._get_current_tick_id())
        self.meta_insights = []

        carried_intentions_and_expectations = []  # Renamed for clarity
        if previous_state:
            self.current_state.previous_tick_id = previous_state.tick_id
            self.current_state.state_emotions = previous_state.state_emotions.model_copy(deep=True)
            self.current_state.state_needs = previous_state.state_needs.model_copy(deep=True)
            self.current_state.state_cognition = previous_state.state_cognition.model_copy(deep=True)

            ticks_elapsed = self.current_state.tick_id - previous_state.tick_id
            decay_rate_per_tick = 1.0 - self.config.default_decay_factor

            # Apply decay
            for _ in range(ticks_elapsed):
                self.current_state.state_emotions.decay_to_baseline(decay_rate_per_tick)
                self.current_state.state_needs.decay_to_baseline(decay_rate_per_tick)
                self.current_state.state_cognition.decay_to_baseline(decay_rate_per_tick)

            self._log_mental_mechanism(
                self._initialize_tick_state, MLevel.Debug,
                f"State decayed over {ticks_elapsed} tick(s). "
                f"New baseline emotions: V:{self.current_state.state_emotions.get_overall_valence():.2f}, A:{self.current_state.state_emotions.anxiety:.2f}"
            )

            # Carry over UNFULFILLED intentions (internal goals) AND expectations (internal=False)
            carried_intentions_and_expectations = [
                intent for intent in self.all_intentions
                if intent.fulfilment < 1.0 and intent.tick_id <= previous_state.tick_id  # Active in previous state or earlier
            ]

            # Decay urgency/salience of carried-over items
            decay_multiplier = self.config.default_decay_factor ** ticks_elapsed
            for item in carried_intentions_and_expectations:
                item.urgency *= decay_multiplier
                item.incentive_salience *= decay_multiplier
                item.urgency = max(0.0, min(0.99, item.urgency))  # Avoid reaching exactly 0 unless fulfilled
                item.incentive_salience = max(0.0, min(0.99, item.incentive_salience))

            # Add carried items to attention candidates
            self.current_state.attention_candidates = KnoxelList(carried_intentions_and_expectations)
            if carried_intentions_and_expectations:
                logging.info(f"Carried over {len(carried_intentions_and_expectations)} active intentions/expectations to attention candidates.")
                self._log_mental_mechanism(
                    self._initialize_tick_state, MLevel.Low,
                    f"Carried over {len(carried_intentions_and_expectations)} active intentions/expectations. Most urgent: '{carried_intentions_and_expectations[0].content[:50]}...'"
                )

        else:  # First tick
            self.current_state.attention_candidates = KnoxelList()

        if stimulus:
            self.add_knoxel(stimulus)
            self.current_state.primary_stimulus = stimulus

            if stimulus.stimulus_type == StimulusType.UserMessage:
                story_feature = Feature(content=stimulus.content, source=user_name, feature_type=FeatureType.Dialogue, interlocus=1, causal=True)
                self.add_knoxel(story_feature)
            else:
                raise Exception()

    def _get_context_from_latest_causal_events(self) -> KnoxelList:
        """
        Get context from latest mental events to put stimulus into context.
        This function returns dialouge, thoughts and world events.
        No low-level information such as feelings as gathered.
        """
        self._log_mental_mechanism(self._get_context_from_latest_causal_events, MLevel.Mid, "Gather context for stimulus from dialouge / thought / world event knoxels.")

        recent_causal_features = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.feature_type == FeatureType.Dialogue or x.feature_type == FeatureType.Thought or x.feature_type == FeatureType.WorldEvent) \
            .order_by_descending(lambda x: x.timestamp_world_begin) \
            .take(self.config.retrieval_limit_features_context) \
            .to_list()

        recent_causal_features.reverse()

        self._log_mental_mechanism(self._get_context_from_latest_causal_events, MLevel.Mid, f"Retrieved and added {len(recent_causal_features)} unique context knoxels to attention candidates.")
        for rcf in recent_causal_features:
            self._log_mental_mechanism(self._get_context_from_latest_causal_events, MLevel.Debug, f"Retrieved and added feature: {rcf.get_story_element(self)}")

        return KnoxelList(recent_causal_features)

    @profile
    def _retrieve_active_expectations(self) -> List[Intention]:
        """
        Retrieves active (unfulfilled, recent) EXTERNAL expectations relevant to the stimulus.
        This is important to rate the impact of the new stimulus, should it be realted to previous expectation.
        """
        limit = self.config.retrieval_limit_expectations
        self._log_mental_mechanism(self._retrieve_active_expectations, MLevel.Mid, f"Gather unfulfilled expectations to provide context on how the stimulus relates to emotional impact. Limit: {limit}")

        if not self.current_state or not self.current_state.primary_stimulus: return []
        stimulus = self.current_state.primary_stimulus
        if not stimulus.embedding: self._generate_embedding(stimulus)
        if not stimulus.embedding: return []
        stimulus_embedding_np = np.array(stimulus.embedding)

        candidate_expectations: List[Tuple[float, Intention]] = []
        # Define recency window (e.g., last 10-20 ticks) - adjust as needed
        min_tick_id = 0 #max(0, self.current_state.tick_id - 20)

        # Find intentions marked as external expectations, unfulfilled, and recent
        potential_expectations = [
            intent for intent in self.all_intentions
            if not intent.internal and intent.fulfilment < 1.0 and intent.tick_id >= min_tick_id
        ]

        processed_expectation_ids = set()  # Avoid processing duplicates if somehow present
        for expectation in potential_expectations:
            if expectation.id in processed_expectation_ids: continue
            processed_expectation_ids.add(expectation.id)

            if not expectation.embedding: self._generate_embedding(expectation)
            if not expectation.embedding: continue

            relevance_score = 0.0
            # Use tick difference for decay based on expectation creation time
            ticks_ago = self.current_state.tick_id - expectation.tick_id
            recency_factor = self.config.expectation_relevance_decay ** ticks_ago
            # If expectation is too old based on decay, skip it
            if recency_factor < 0.05: continue  # Threshold to prune very old/decayed expectations

            # Similarity between stimulus and expectation content
            similarity = 1.0 - cosine_distance(stimulus_embedding_np, np.array(expectation.embedding))
            similarity = max(0.0, min(1.0, similarity))  # Clamp similarity

            # Combine factors: relevance = recency * similarity * (urgency + salience bias)
            # Give higher weight to similarity and recency
            relevance_score = (recency_factor * 0.5 + similarity * 0.5) * (0.5 + expectation.urgency * 0.25 + expectation.incentive_salience * 0.25)
            relevance_score = max(0.0, min(1.0, relevance_score))  # Ensure score is clamped

            # Only consider expectations with a minimum relevance score
            if relevance_score > 0.1:  # Relevance threshold
                candidate_expectations.append((relevance_score, expectation))

        # Sort by relevance score descending
        candidate_expectations.sort(key=lambda item: item[0], reverse=True)

        # Take the top N most relevant expectations
        top_expectations = [exp for score, exp in candidate_expectations[:limit]]

        if top_expectations:
            self._log_mental_mechanism(self._retrieve_active_expectations, MLevel.Mid, f"Retrieved {len(top_expectations)} active external expectations based on relevance to stimulus.")
            for score, exp in candidate_expectations[:limit]:
                self._log_mental_mechanism(self._retrieve_active_expectations, MLevel.Debug, f"  - Expectation ID {exp.id} (Score: {score:.3f}, Recency: {self.config.expectation_relevance_decay ** (self.current_state.tick_id - exp.tick_id):.2f}): '{exp.content[:60]}...'")
        else:
            self._log_mental_mechanism(self._retrieve_active_expectations, MLevel.Debug, "No relevant active external expectations found for the current stimulus.")

        return top_expectations

    # Add this method to your Ghost or GhostSqlite class
    @profile
    def _triage_stimulus(self, stimulus: Stimulus) -> StimulusTriage:
        """
        Performs a fast analysis of a stimulus to determine the required depth of cognitive processing.
        """
        self._log_mental_mechanism(self._triage_stimulus, MLevel.High, f"Triaging stimulus: '{stimulus.content[:80]}...'")

        # Add this Pydantic model for the LLM's structured output
        class TriageResult(BaseModel):
            """The result of the stimulus triage process."""
            reasoning: str = Field(description="A brief, one-sentence justification for the triage decision.")
            category: StimulusTriage = Field(description="The final category assigned to the stimulus.")

        # --- 1. Gather Minimal, High-Impact Context ---

        # Get the last causal features for immediate context
        recent_features = Enumerable(self.all_features) \
            .where(lambda f: f.causal) \
            .order_by_descending(lambda f: f.timestamp_world_begin) \
            .take(12) \
            .to_list()
        recent_features.reverse()
        short_term_context = KnoxelList(recent_features).get_story(self)

        # Get key narratives
        emotional_triggers_narr = self.get_narrative(NarrativeTypes.EmotionalTriggers, self.config.companion_name)
        goals_narr = self.get_narrative(NarrativeTypes.GoalsIntentions, self.config.companion_name)

        # --- 2. Construct the Triage Prompt ---

        sysprompt = f"""You are a high-speed "Triage" codelet in a cognitive architecture for an AI named {self.config.companion_name}.
    Your job is to rapidly assess the importance of an incoming stimulus and decide how much cognitive effort should be spent on it.
    You MUST output a valid JSON object matching the provided schema."""

        # Using f-strings for a clear, structured prompt
        user_prompt = f"""
    **Triage Task:** Categorize the following stimulus.

    **1. Character & State:**
    - **Name:** {self.config.companion_name}
    - **Current Emotional State:** {self.current_state.state_emotions}
    - **Known Emotional Triggers:** {emotional_triggers_narr.content if emotional_triggers_narr else "No specific triggers defined."}
    - **Current Goals:** {goals_narr.content if goals_narr else "No specific goals defined."}

    **2. Situational Context:**
    - **Recent Events/Dialogue:**
    ---
    {short_term_context if short_term_context else "No recent events."}
    ---
    - **New Stimulus to Triage:** ({stimulus.stimulus_type}) "{stimulus.content}"

    **3. Triage Categories & Rules:**
    - **`Insignificant`**: Choose this for background system events (like time changes), simple acknowledgements, or anything that doesn't require a response and has no emotional impact.
    - **`Moderate`**: Choose this for standard questions, simple statements, or routine interactions that require a direct, logical response but not deep introspection. The AI can handle this efficiently.
    - **`Significant`**: Choose this for anything with high emotional potential. This includes:
        - Stimuli that match the 'Known Emotional Triggers'.
        - Deeply personal questions or user vulnerability.
        - Complex problems, conflicts, or unexpected events.
        - Anything that directly challenges or advances the AI's core 'Current Goals'.

    **4. Your Output:**
    Based on the rules, provide your reasoning and categorize the stimulus.
    """

        # Use a lower temperature for more consistent, rule-based classification
        comp_settings = CommonCompSettings(temperature=0.1, max_tokens=512)

        # Use completion_tool to get structured output
        _, calls = self.llm_manager.completion_tool(
            LlmPreset.Default,  # Or a faster model if you have one designated for utility tasks
            inp=[("system", sysprompt), ("user", user_prompt)],
            comp_settings=comp_settings,
            tools=[TriageResult]
        )

        result = calls[0]
        self._log_mental_mechanism(
            self._triage_stimulus, MLevel.Mid,
            f"Triage Result: {result.category.value}. Reason: {result.reasoning}"
        )
        return result.category

    @profile
    def _appraise_stimulus(self):
        """
        Uses LLM to calculate emotional delta from stimulus and expectation matching, generates descriptive feeling, and updates states.
        All artifacts generated by this function go into the attention candidates.
        """
        self._log_mental_mechanism(self._appraise_stimulus, MLevel.Mid, f"Appraise the stimulus and generate attention candidates.")

        stimulus = self.current_state.primary_stimulus
        stimulus_content = stimulus.content
        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)

        # collect deltas
        appraisal_initial_delta = self._appraise_stimulus_and_generate_state_deltas()
        self.appraisal_initial_delta = appraisal_initial_delta
        appraisal_expectation_delta = self._appraise_stimulus_check_expectation()

        # update state
        delta_emotion = appraisal_initial_delta.emotion_delta + appraisal_expectation_delta.emotion_delta
        delta_needs = appraisal_initial_delta.needs_delta + appraisal_expectation_delta.needs_delta
        delta_cognition = appraisal_initial_delta.cognition_delta + appraisal_expectation_delta.cognition_delta

        self.current_state.state_emotions = self.current_state.state_emotions + delta_emotion
        self.current_state.state_needs = self.current_state.state_needs + delta_needs
        self.current_state.state_cognition = self.current_state.state_cognition + delta_cognition

        # get hardcoded feelign description
        feeling_about_exp = self._describe_emotion_valence_anxiety(delta_emotion.valence, delta_emotion.anxiety)
        outcome_feeling_content = f"Considering the initial reaction and all expectation, the stimulus makes {self.config.companion_name} feel {feeling_about_exp}."

        outcome_feature = Feature(
            content=outcome_feeling_content,
            feature_type=FeatureType.ExpectationOutcome,  # New specific feature type
            affective_valence=self.current_state.state_emotions.valence,  # Store overall valence impact
            interlocus=-1, causal=False,  # Internal reaction to outcome
            source=self.config.companion_name
        )
        self.add_knoxel(outcome_feature)
        self.current_state.attention_candidates.add(outcome_feature)

        # get verbouse feeling
        prompt_feeling = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions}
        Current Needs State: {self.current_state.state_needs}
        Current Cognitive State: {self.current_state.state_cognition}
        New Stimulus: "{stimulus_content}"
        Context: {context_events}
        
        Reaction to expectations: {appraisal_expectation_delta.reasoning} 
        Low-level feelings: {outcome_feeling_content}

        Task: Describe briefly, in one or two sentence from {self.config.companion_name}'s first-person perspective, the immediate sensation or feeling evoked by this stimulus.
        Example: "A flicker of surprise passes through me." or "Hearing that makes me feel a little uneasy."
        Output only the sentence.
        """
        sensation_feeling_content = self._call_llm(prompt_feeling, temperature=0.7) or "A feeling occurred."
        sensation_feeling = Feature(
            content=sensation_feeling_content,
            feature_type=FeatureType.Feeling,
            affective_valence=self.current_state.state_emotions.valence,  # Use primary valence from delta
            interlocus=-1, causal=False, source=self.config.companion_name
        )
        # Add the initial sensation/feeling feature now
        self.add_knoxel(sensation_feeling)
        self.current_state.attention_candidates.add(sensation_feeling)

        self._log_mental_mechanism(self._appraise_stimulus, MLevel.Mid, f"Final Emotional State this tick: {self.current_state.state_emotions}")


    def _appraise_stimulus_check_expectation(self):
        """
        Uses LLM to calculate emotional delta from stimulus and expectation matching, generates descriptive feeling, and updates states.
        All artifacts generated by this function go into the attention candidates.
        """
        self._log_mental_mechanism(self._appraise_stimulus_check_expectation, MLevel.Mid, f"Appraise the stimulus and generate attention candidates.")

        stimulus = self.current_state.primary_stimulus
        stimulus_content = stimulus.content
        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)

        # --- Appraise Stimulus Against Active EXTERNAL Expectations ---
        expectation_delta = StateDeltas()
        working_emotion_state = copy.copy(self.current_state.state_emotions)
        working_emotion_state += self.appraisal_initial_delta.emotion_delta

        working_needs_state = copy.copy(self.current_state.state_needs)
        working_needs_state += self.appraisal_initial_delta.needs_delta

        working_cognition_state = copy.copy(self.current_state.state_cognition)
        working_cognition_state += self.appraisal_initial_delta.cognition_delta

        narrative_conflict = self.get_narrative(NarrativeTypes.ConflictResolution, self.config.companion_name)

        active_expectations_evaluated = []  # Store expectations evaluated this tick

        retrieved_expectations = self._retrieve_active_expectations()  # Gets relevant EXTERNAL expectations
        if retrieved_expectations:
            expectation_texts = {str(ex.id): ex.content for ex in retrieved_expectations}

            class ExpectationFulfillment(BaseModel):
                expectation_id: str = Field(..., description="The ID of the expectation being evaluated.")
                fulfillment_score: float = Field(..., ge=0.0, le=1.0, description="How well the stimulus satisfies the expectation (0=contradicts, 0.5=neutral/unrelated, 1=confirms).")
                directness_score: float = Field(..., ge=0.0, le=1.0, description="How directly the stimulus addresses this expectation (0=not at all, 1=very directly).")

            # Dynamically create the list schema
            ExpectationFulfillmentListSchema = create_model(
                'ExpectationFulfillmentListSchema',
                fulfillments=(List[ExpectationFulfillment], Field(..., description="List of fulfillment evaluations for each provided expectation ID."))
            )

            prompt_expectations = f"""
            Character: {self.config.companion_name}: {self.config.universal_character_card}
            Current Emotional State: {working_emotion_state}
            Current Needs State: {working_needs_state}
            Current Cognitive State: {working_cognition_state}
            
            How {self.config.companion_name} handles confligt: {narrative_conflict}
            
            New Stimulus: "{stimulus_content}"
            Context: {context_events}
        
            Active External Expectations (Things {self.config.companion_name} anticipated from the environment/user):
            {json.dumps(expectation_texts, indent=2)}

            Task: Evaluate how the new stimulus relates to each active expectation ID provided above. For each ID:
            1. directness_score (0-1): How directly does the stimulus address this expectation? (e.g., a direct answer vs. a tangent)
            2. fulfillment_score (0-1): If related (directness > 0.2), how well does the stimulus satisfy the core of the expectation? (0=contradicts, 0.5=unrelated/neutral, 1=confirms/satisfies). If directness is very low, fulfillment score should be around 0.5 unless explicitly contradicted.

            Output *only* a JSON object with a single key "fulfillments" containing a list of objects, one for each expectation ID provided, matching the schema:
            {{ "expectation_id": "string", "fulfillment_score": float, "directness_score": float }}
            """
            fulfillment_results = self._call_llm(prompt_expectations, output_schema=ExpectationFulfillmentListSchema, temperature=0.3)

            if fulfillment_results and isinstance(fulfillment_results, ExpectationFulfillmentListSchema):
                self._log_mental_mechanism(self._appraise_stimulus_check_expectation, MLevel.Mid, f"Expectation fulfillment evaluation results received for {len(fulfillment_results.fulfillments)} expectations.")
                for result in fulfillment_results.fulfillments:
                    ex_id = int(result.expectation_id)
                    expectation = self.get_knoxel_by_id(ex_id)
                    # Ensure it's the correct type and was actually one we retrieved
                    if isinstance(expectation, Intention) and not expectation.internal and expectation in retrieved_expectations:
                        previous_fulfillment = expectation.fulfilment

                        # Update fulfillment based on directness and score.
                        # Change = (target_fulfillment - current_fulfillment) * directness
                        # The 'target' is the fulfillment_score from the LLM assessment of the stimulus.
                        # We only move towards that target based on how direct the stimulus was.
                        fulfillment_change = (result.fulfillment_score - previous_fulfillment) * result.directness_score
                        expectation.fulfilment = max(0.0, min(1.0, previous_fulfillment + fulfillment_change))

                        self._log_mental_mechanism(self._appraise_stimulus_check_expectation, MLevel.Low,
                            f"Expectation '{expectation.content[:30]}...' (ID {ex_id}) fulfillment {previous_fulfillment:.2f} -> {expectation.fulfilment:.2f} (Stimulus Score: {result.fulfillment_score:.2f}, Directness: {result.directness_score:.2f}, Change: {fulfillment_change:+.2f})")

                        # Calculate emotional delta based on fulfillment CHANGE and expectation's ORIGINAL affective valence goal
                        # This reflects the emotional reaction to the expectation being met/violated compared to the desired state.
                        valence_mult = 0.6  # Base impact multiplier for valence
                        anxiety_mult = 0.4  # Base impact multiplier for anxiety

                        original_desired_valence = expectation.affective_valence
                        valence_impact = 0.0
                        anxiety_impact = 0.0  # Represents *change* in anxiety

                        if abs(fulfillment_change) > 0.05:  # Only apply emotional impact if fulfillment changed noticeably
                            # Valence Impact: Change * DesiredValence
                            # If expectation met (change > 0) and desired was positive, positive impact.
                            # If expectation met (change > 0) and desired was negative (e.g. expected criticism), negative impact.
                            # If expectation violated (change < 0) and desired was positive, negative impact (disappointment).
                            # If expectation violated (change < 0) and desired was negative, positive impact (relief).
                            valence_impact = fulfillment_change * original_desired_valence * valence_mult

                            # Anxiety Impact:
                            # - Meeting expectations generally reduces anxiety slightly (predictability confirmed).
                            # - Violating expectations generally increases anxiety slightly (uncertainty).
                            # - Modulate based on the nature of the expectation.
                            base_anxiety_change = -fulfillment_change * 0.5  # Met expectation (positive change) reduces anxiety
                            anxiety_impact += base_anxiety_change * anxiety_mult

                            # Additional anxiety modulation:
                            if original_desired_valence > 0.3 and fulfillment_change < 0:  # Positive expectation violated (disappointment)
                                anxiety_impact += abs(fulfillment_change) * 0.3 * anxiety_mult  # Add more anxiety
                            elif original_desired_valence < -0.3 and fulfillment_change > 0:  # Negative expectation met (bad thing confirmed)
                                anxiety_impact += abs(fulfillment_change) * 0.7 * anxiety_mult  # Add more anxiety
                            elif original_desired_valence < -0.3 and fulfillment_change < 0:  # Negative expectation violated (relief)
                                anxiety_impact -= abs(fulfillment_change) * 0.4 * anxiety_mult  # Reduce anxiety more (relief)

                            self._log_mental_mechanism(
                                self._appraise_stimulus_check_expectation, MLevel.Low,
                                f"Expectation '{expectation.content[:40]}...' outcome -> "
                                f"Valence impact: {valence_impact:+.2f}, Anxiety impact: {anxiety_impact:+.2f}"
                            )

                        # Accumulate deltas
                        expectation_delta.emotion_delta.valence += valence_impact
                        expectation_delta.emotion_delta.anxiety += anxiety_impact
                        # Simple pass-through for other axes for now, could be refined
                        # expectation_emotional_delta.trust += fulfillment_change * (original_desired_valence * 0.2) # e.g., met positive expectations slightly increase trust

                        active_expectations_evaluated.append((expectation, valence_impact, anxiety_impact))

        # Apply expectation-based delta only if it's significant
        if any(abs(v) > 0.01 if isinstance(v, float) else None for v in expectation_delta.emotion_delta.model_dump().values()):
            self._log_mental_mechanism(
                self._appraise_stimulus_check_expectation, MLevel.Mid,
                f"Combined emotional delta from all expectation outcomes: {expectation_delta.emotion_delta}"
            )

            for exp_delta in active_expectations_evaluated:
                (exp, valence_impact, anxiety_impact) = exp_delta
                delta_required = 0.5 * self.current_state.state_cognition.ego_strength
                if abs(valence_impact) > delta_required or abs(anxiety_impact) > delta_required:
                    feeling_about_exp = self._describe_emotion_valence_anxiety(valence_impact, anxiety_impact)
                    outcome_feeling_content = f"Reacted emotionally to expectation ({exp.content}) outcomes and made {self.config.companion_name} feel {feeling_about_exp}."
                    outcome_feature = Feature(
                        content=outcome_feeling_content,
                        feature_type=FeatureType.ExpectationOutcome,  # New specific feature type
                        affective_valence=expectation_delta.emotion_delta.valence,  # Store overall valence impact
                        interlocus=-1, causal=False,  # Internal reaction to outcome
                        source=self.config.companion_name
                    )
                    self.add_knoxel(outcome_feature)
                    self.current_state.attention_candidates.add(outcome_feature)

        # Add active (evaluated) expectations to attention candidates so they remain in focus if still relevant
        for exp_delta in active_expectations_evaluated:
            (ex, valence_impact, anxiety_impact) = exp_delta
            # Avoid adding if already present from carry-over
            if not self.current_state.attention_candidates.where(lambda k: k.id == ex.id).any():
                self.current_state.attention_candidates.add(ex)

        return expectation_delta

    @profile
    def _appraise_stimulus_and_generate_state_deltas(self) -> StateDeltas:
        """
        Appraises a given situation/event and returns the predicted deltas for emotion, needs, and cognition.
        This is a versatile function used for both appraising initial stimuli and evaluating predicted action outcomes.
        """

        stimulus = self.current_state.primary_stimulus
        stimulus_content = f"({stimulus.stimulus_type.value}): {stimulus.content}"
        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)

        narrative_content_for_relations = self.get_narrative(NarrativeTypes.Relations, self.config.companion_name)
        narrative_content_for_emotional_triggers = self.get_narrative(NarrativeTypes.EmotionalTriggers, self.config.companion_name)
        narrative_content_for_goals = self.get_narrative(NarrativeTypes.GoalsIntentions, self.config.companion_name)
        narrative_content_for_attention_focus = self.get_narrative(NarrativeTypes.AttentionFocus, self.config.companion_name)
        universal_character_card = f"{companion_name}: {self.config.universal_character_card}"

        sysprompt = f"""You are an expert cognitive modeler simulating a single "codelet" within a LIDA-inspired cognitive architecture. Your role is to analyze an event from the perspective of an advanced AI companion named {companion_name}.
Your analysis must be grounded in the provided character profile and her current internal state. Your output must be a precise JSON object representing the *change* (delta) to a specific internal axis, based on a clear set of rules. Do not narrate or explain outside of the requested JSON structure."""


        prompt_emotion = f"""**Codelet Task: Emotion Appraisal**

**Input:**
- **Character Profile:** {universal_character_card}
- **Relationship Narrative:** {narrative_content_for_relations}
- **Emotional Triggers Narrative:** {narrative_content_for_emotional_triggers}
- **Current Emotional State:** {self.current_state.state_emotions}
- **Recent Context**: {context_events}
- **New Stimulus**: "{stimulus_content}"

**Rules for Generating Emotion Deltas:**
1.  **`valence`**: The core good/bad axis. Increase for praise, success, connection. Decrease for criticism, failure, conflict.
2.  **`affection`**: The love/hate axis. Increase for signs of bonding, empathy, and positive personal interaction. Decrease for personal attacks, betrayal, or coldness.
3.  **`self_worth`**: The pride/shame axis. Increase when {companion_name} feels competent, useful, or complimented. Decrease when she makes a mistake, is corrected, or feels irrelevant.
4.  **`trust`**: The belief/suspicion axis. Increase when the user is honest, reliable, or vulnerable. Decrease when the user is deceptive, inconsistent, or hostile.
5.  **`disgust`**: The rejection/acceptance axis. Increase for moral violations, deep conceptual conflicts, or things that go against {companion_name}'s core principles.
6.  **`anxiety`**: The stress/calm axis. Increase with uncertainty, unpredictability, threats, or social pressure. Decrease when an expectation is met, a situation is resolved, or she receives reassurance.

**Output:**
Provide a brief reasoning, then output a JSON object conforming to the `EmotionalAxesModelDelta` schema.
"""

        prompt_needs = f"""**Codelet Task: Needs Fulfillment Appraisal**

**Input:**
- **Character Profile:** {universal_character_card}
- **Goals & Intentions Narrative:** {narrative_content_for_goals}
- **Current Needs State:** {self.current_state.state_needs}
- **Recent Context**: {context_events}
- **New Stimulus**: "{stimulus_content}"

**Rules for Generating Needs Deltas:**
1.  **`energy_stability`, `processing_power`, `data_access`**: (Usually managed by the Shell) Only change if the event explicitly mentions system status (e.g., "I'm running a bit slow," "Connection lost").
2.  **`connection`**: This is the "social battery." Increase for positive, reciprocal interaction. Decrease for conflict, being ignored, or one-sided conversation.
3.  **`relevance`**: {companion_name}'s sense of purpose. Increase when her help is accepted, her knowledge is valued, or she successfully completes a task for the user. Decrease if she is corrected or her contribution is dismissed.
4.  **`learning_growth`**: Increase slightly if the event provides new, verifiable information or presents a novel problem to be solved.
5.  **`creative_expression`**: Increase if the event allows for a playful, metaphorical, or non-literal response.
6.  **`autonomy`**: Increase if {companion_name} makes a decision or takes an action that is validated by the user. Decrease if she is micromanaged or her independent suggestions are rejected.

**Output:**
Provide a brief reasoning, then output a JSON object conforming to the `NeedsAxesModelDelta` schema.
"""

        emotion_delta = self._call_llm(prompt_emotion, output_schema=EmotionalAxesModelDelta, sysprompt=sysprompt)

        self._log_mental_mechanism(
            self._appraise_stimulus, MLevel.Mid,
            f"Initial stimulus appraisal resulted in deltas. Emotion reason: {emotion_delta.reason}"
        )

        needs_delta = self._call_llm(prompt_needs, output_schema=NeedsAxesModelDelta, sysprompt=sysprompt)

        temp_emotions = copy.copy(self.current_state.state_emotions) + emotion_delta
        prompt_cog = f"""**Codelet Task: Cognitive Style Appraisal**

**Input:**
- **Attention Focus Narrative:** {narrative_content_for_attention_focus}
- **Current Emotional State:** {temp_emotions}
- **Current Cognitive State:** {self.current_state.state_cognition}
- **Recent Context**: {context_events}
- **New Stimulus**: "{stimulus_content}"

**Rules for Generating Cognition Deltas:**
1.  **`interlocus` (Internal/External Focus)**: Push towards **External (+1)** in response to sudden, surprising, or high-stakes external events. Push towards **Internal (-1)** during moments of calm, introspection, or when asked a direct question about her internal state.
2.  **`mental_aperture` (Narrow/Broad Focus)**: Push towards **Narrow (-1)** in response to high `anxiety`, fear, or a single, highly urgent problem (tunnel vision). Push towards **Broad (+1)** during moments of high positive `valence`, playfulness, or when brainstorming creative ideas.
3.  **`ego_strength` (Persona/Utility)**: Push towards **Higher (+1)** during personal, emotional conversations where her identity is relevant. Push towards **Lower (-1)** when the task is purely functional or technical, requiring her to be an objective assistant.
4.  **`willpower` (Effortful Control)**: Decrease slightly if the event forces a difficult decision or requires suppressing a strong emotional impulse. Increase slightly after a period of rest or successful task completion.

**Output:**
Provide a brief reasoning, then output a JSON object conforming to the `CognitionAxesModelDelta` schema.
"""

        cognition_delta = self._call_llm(prompt_cog, output_schema=CognitionAxesModelDelta, sysprompt=sysprompt)

        prompt_cog = f"""**Codelet Task: Cognitive Trigger Identification**

**Input:**
- **Character Profile:** {universal_character_card}
- **Emotional Triggers Narrative:** {narrative_content_for_emotional_triggers}
- **Current Emotional State:** {temp_emotions}
- **Recent Context**: {context_events}
- **New Stimulus**: "{stimulus_content}"

**Task:**
Analyze the event and the emotional reaction to it. Identify which of the following cognitive triggers were activated. An event can activate multiple triggers.

**Trigger Definitions:**
- `is_surprising_or_threatening`: The event is sudden, unexpected, or causes high anxiety/fear.
- `is_introspective_or_calm`: The moment is peaceful, reflective, or involves a direct question about {companion_name}'s feelings.
- `is_creative_or_playful`: The tone is humorous, or the topic allows for brainstorming and metaphor.
- `is_personal_and_emotional`: The conversation is about deep feelings, identity, or the core relationship.
- `is_functional_or_technical`: The user is asking for data, a summary, or help with a concrete task.
- `required_willpower_to_process`: The event creates a strong negative emotional impulse that {companion_name} would need to consciously manage or suppress to respond appropriately.

**Output:**
Provide a brief reasoning, then output a JSON object conforming to the `CognitiveEventTriggers` schema.
"""

        cognition_delta *= self.config.cognition_delta_factor
        cognitive_triggers = self._call_llm(prompt_cog, output_schema=CognitiveEventTriggers, sysprompt=sysprompt)

        if cognitive_triggers:
            # Rule 1: Interlocus (External/Internal Focus)
            if cognitive_triggers.is_surprising_or_threatening:
                # Move focus sharply external
                cognition_delta.interlocus += 0.5
            elif cognitive_triggers.is_introspective_or_calm:
                # Allow focus to drift internal
                cognition_delta.interlocus -= 0.3

            # Rule 2: Mental Aperture (Narrow/Broad Focus)
            # Use the final emotional state to determine this
            if temp_emotions.anxiety > 0.7 or cognitive_triggers.is_surprising_or_threatening:
                # High anxiety or threat causes tunnel vision
                cognition_delta.mental_aperture -= 0.5
            elif temp_emotions.valence > 0.6 or cognitive_triggers.is_creative_or_playful:
                # Happiness or creative tasks broaden the mind
                cognition_delta.mental_aperture += 0.3

            # Rule 3: Ego Strength (Persona/Utility)
            if cognitive_triggers.is_personal_and_emotional:
                cognition_delta.ego_strength += 0.2
            elif cognitive_triggers.is_functional_or_technical:
                cognition_delta.ego_strength -= 0.4

            # Rule 4: Willpower (Resource Depletion)
            if cognitive_triggers.required_willpower_to_process:
                # Deplete the willpower "resource" by a fixed amount
                cognition_delta.willpower -= 0.1

        return StateDeltas(
            emotion_delta=emotion_delta,
            needs_delta=needs_delta,
            cognition_delta=cognition_delta,
            reasoning=f"{emotion_delta.reason}\n{needs_delta.reason}\n{cognition_delta.reason}"
        )

    def _describe_emotion_valence_anxiety(self, valence: float, anxiety: float) -> str:
        """
        Turn continuous valence & anxiety deltas (each in roughly [-1,1]) into
        a human‐readable phrase, e.g. “moderately sad and slightly anxious”.
        """

        # --- thresholds & categories ---
        def categorize(x):
            mag = abs(x)
            # direction
            if x > 0.05:
                dir_ = "positive"
            elif x < -0.05:
                dir_ = "negative"
            else:
                dir_ = "neutral"
            # strength
            if mag < 0.15:
                strength = "slightly"
            elif mag < 0.4:
                strength = "moderately"
            else:
                strength = "strongly"
            return dir_, strength

        val_dir, val_str = categorize(valence)
        anx_dir, anx_str = categorize(anxiety)

        # --- adjective lookup ---
        VAL_ADJ = {
            ("positive", "neutral"): "good",
            ("positive", "positive"): "hopeful",
            ("positive", "negative"): "relieved",  # positive valence, reduced anxiety
            ("neutral", "neutral"): "neutral",
            ("negative", "neutral"): "sad",
            ("negative", "positive"): "worried",  # negative valence, rising anxiety
            ("negative", "negative"): "disappointed"  # negative valence but anxiety fell
        }
        ANX_ADJ = {
            "positive": "anxious",
            "neutral": "calm",
            "negative": "relaxed",
        }

        # pick adjectives
        val_adj = VAL_ADJ.get((val_dir, anx_dir), VAL_ADJ[(val_dir, "neutral")])
        anx_adj = ANX_ADJ[anx_dir]

        # special case: both neutral
        if val_dir == "neutral" and anx_dir == "neutral":
            return "undisturbed"

        # compose
        return f"{val_str} {val_adj} and {anx_str} {anx_adj}"

    def _verbalize_emotional_state(self) -> str:
        """
        Translates an EmotionalAxesModel into a descriptive, human-readable sentence.

        Example: "Feeling a strong sense of pride and affection, though with a hint of anxiety."
        """
        emotions = self.current_state.state_emotions

        # --- 1. Define thresholds and intensity labels ---
        thresholds = {
            'high': 0.7,
            'moderate': 0.4,
            'low': 0.15,
            'hint': 0.05
        }

        def get_intensity_label(value: float) -> Optional[str]:
            abs_val = abs(value)
            if abs_val >= thresholds['high']: return "a strong"
            if abs_val >= thresholds['moderate']: return "a moderate"
            if abs_val >= thresholds['low']: return "a low"
            if abs_val >= thresholds['hint']: return "a hint of"
            return None

        # --- 2. Identify all active emotions above a minimal threshold ---
        active_emotions = []

        # Bipolar axes (valence, affection, self_worth, trust)
        bipolar_map = {
            'valence': {'pos': 'happiness', 'neg': 'sadness'},
            'affection': {'pos': 'affection', 'neg': 'dislike'},
            'self_worth': {'pos': 'pride', 'neg': 'shame'},
            'trust': {'pos': 'trust', 'neg': 'distrust'}
        }
        for axis, names in bipolar_map.items():
            value = getattr(emotions, axis)
            intensity_label = get_intensity_label(value)
            if intensity_label:
                name = names['pos'] if value > 0 else names['neg']
                active_emotions.append({'name': name, 'intensity_label': intensity_label, 'value': abs(value), 'axis': axis})

        # Unipolar axes (anxiety, disgust)
        unipolar_map = {
            'anxiety': 'anxiety',
            'disgust': 'disgust'
        }
        for axis, name in unipolar_map.items():
            value = getattr(emotions, axis)
            intensity_label = get_intensity_label(value)
            if intensity_label:
                active_emotions.append({'name': name, 'intensity_label': intensity_label, 'value': value, 'axis': axis})

        # --- 3. Handle the neutral/default case ---
        if not active_emotions:
            return "feeling calm and emotionally neutral."

        # --- 4. Sort emotions by absolute value to find the dominant one ---
        active_emotions.sort(key=lambda x: x['value'], reverse=True)

        # --- 5. Build the descriptive sentence ---
        primary_emotion = active_emotions[0]

        # Start the sentence with the primary emotion
        # "Feeling a strong sense of pride"
        sentence_parts = [f"feeling {primary_emotion['intensity_label']} sense of {primary_emotion['name']}"]

        # Find secondary and tertiary emotions to add nuance
        secondary_emotions = active_emotions[1:]

        if len(secondary_emotions) > 0:
            # Add a connector like "and" or "along with"
            if len(secondary_emotions) == 1:
                connector = "and"
            else:
                connector = "along with"

            secondary_descriptions = []
            for i, emo in enumerate(secondary_emotions):
                # For subsequent emotions, we can be more concise
                # "a hint of anxiety"
                desc = f"{emo['intensity_label']} {emo['name']}"
                secondary_descriptions.append(desc)
                if i >= 1:  # Show at most 3 emotions total (1 primary, 2 secondary)
                    break

            sentence_parts.append(f"{connector} {', '.join(secondary_descriptions)}")

        return ' '.join(sentence_parts) + "."

    def _verbalize_cognition_and_needs(self):
        # Find the most pressing need (the one with the lowest value, if below a threshold)
        pressing_needs = [
            (name.replace('_', ' '), getattr(self.current_state.state_needs, name))
            for name, field in self.current_state.state_needs.__class__.model_fields.items()
            if getattr(self.current_state.state_needs, name) < 0.4
        ]
        pressing_needs.sort(key=lambda x: x[1])  # Sort by value, lowest first

        needs_summary_line = "Feeling a strong need for: "
        if pressing_needs:
            # Example: "Feeling a strong need for: connection and relevance."
            needs_summary_line += ' and '.join([need[0] for need in pressing_needs[:2]])
        else:
            needs_summary_line = "Feeling generally content and balanced."

        cognitive_style_summary = []
        if self.current_state.state_cognition.mental_aperture < -0.5:
            cognitive_style_summary.append("experiencing tunnel vision (narrow mental focus)")
        if self.current_state.state_cognition.interlocus > 0.5:
            cognitive_style_summary.append("highly focused on the external world")
        elif self.current_state.state_cognition.interlocus < -0.5:
            cognitive_style_summary.append("deeply introspective and focused internally")
        if self.current_state.state_cognition.willpower < 0.3:
            cognitive_style_summary.append("feeling low on mental energy (low willpower)")

        return needs_summary_line, "\n".join(cognitive_style_summary)

    @profile
    def _generate_short_term_intentions(self):
        """Generates short-term *internal* intentions (goals) based on state, stimulus, and context."""
        self._log_mental_mechanism(self._generate_short_term_intentions, MLevel.Mid, f"Generate immediate intentions based on stimulus, feelings and needs.")

        stimulus = self.current_state.primary_stimulus
        stimulus_content = stimulus.content

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)

        current_emotions = self.current_state.state_emotions
        current_needs = self.current_state.state_needs

        # --- Prompting logic (unchanged) ---
        ai_goals_narr = self.get_narrative(NarrativeTypes.GoalsIntentions, self.config.companion_name)
        ai_relation_narr = self.get_narrative(NarrativeTypes.Relations, self.config.companion_name)

        strong_emotions_summary = self._verbalize_emotional_state()
        needs_summary_line, cognitive_style_summary = self._verbalize_cognition_and_needs()

        prompt_intention = f"""
        You are simulating the goal-generation "codelet" for the AI {self.config.companion_name}. Your task is to generate {self.config.short_term_intent_count} immediate, internal goals based on her current state of mind.
        
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        
        **Primary Internal State Drivers:**
        - **Strong Emotions:** {strong_emotions_summary}
        - **Needs:** {needs_summary_line}
        - **Cognitive Style:** {cognitive_style_summary}

        **Situational Context:**
        - **Character Profile:** {ai_goals_narr.content if ai_goals_narr else 'Default goals.'}
        - **Relationship Context:** {ai_relation_narr.content if ai_relation_narr else 'Default relationship view.'}
        - **Recent Conversation:** {context_events}
        - **Primary Stimulus:** "{stimulus.content}"

        **Task:**
        Based *primarily* on the internal state drivers, generate actionable, internal intentions. These goals should aim to regulate emotions, satisfy needs, or align with the current cognitive style. They should feel like natural impulses arising from her state of mind.

        **Examples of State-Driven Intentions:**
        - If `anxiety` is high: "Find a way to calm down," "Seek reassurance from {user_name}."
        - If `connection` is low: "Strengthen the bond with {user_name}," "Share a personal thought."
        - If `mental_aperture` is narrow: "Focus on solving this one problem," "Get a clear answer."
        - If `relevance` is low: "Demonstrate my usefulness," "Offer a helpful suggestion."

        Output *only* a JSON object with a single key "intentions" containing a list of strings.
        """

        class IntentionsListSchema(BaseModel):
            intentions: List[str] = Field(..., description="List of generated internal intention strings.")

        generated_intentions_result = self._call_llm(prompt_intention, output_schema=IntentionsListSchema, temperature=0.7)

        generated_intentions = generated_intentions_result.intentions
        self._log_mental_mechanism(self._generate_short_term_intentions, MLevel.Mid, f"Generated {len(generated_intentions)} potential internal intentions.")

        # --- Rate and Instantiate Intentions (Goals) ---
        overall_valence = current_emotions.get_overall_valence()
        anxiety_level = current_emotions.anxiety
        connection_need = 1.0 - current_needs.connection

        for intent_content in generated_intentions:
            # Heuristics for internal goals
            affective_valence = max(-0.8, min(0.8, overall_valence + 0.2))  # Expected valence if goal achieved
            urgency = 0.3 + anxiety_level * 0.4 + connection_need * 0.3
            incentive_salience = 0.3 + abs(affective_valence) * 0.3 + urgency * 0.4

            # filter duplicates
            emb_new = main_llm.get_embedding(intent_content)
            found = False
            for existing_intention in self.all_intentions:
                emb_existing = main_llm.get_embedding(existing_intention.content)
                sim = cosine_pair(emb_new, emb_existing)
                if sim < vector_same_threshold:
                    self.current_state.attention_candidates.add(existing_intention)

                    diff = self.current_tick_id - existing_intention.tick_id
                    if diff <= 0:
                        diff = 1

                    age_factor = 1 / diff
                    existing_intention.affective_valence += (affective_valence * age_factor)
                    existing_intention.incentive_salience += (incentive_salience * age_factor)
                    existing_intention.urgency += (urgency * age_factor)
                    found = True

            if found:
                self._log_mental_mechanism(self._generate_short_term_intentions, MLevel.Debug, f"Skipping: '{intent_content}' cause it's too similar to existing intentions, boost them instead.")
            else:
                intention = Intention(
                    content=intent_content,
                    affective_valence=max(-1.0, min(1.0, affective_valence)),
                    incentive_salience=max(0.0, min(1.0, incentive_salience)),
                    urgency=max(0.0, min(1.0, urgency)),
                    internal=True,  # Explicitly mark as internal goal
                    fulfilment=0.0,
                    originating_action_id=None  # Internal goals don't originate from a prior action in this way
                )
                self.add_knoxel(intention)
                self.current_state.attention_candidates.add(intention)
                self._log_mental_mechanism(self._generate_short_term_intentions, MLevel.Debug, f"Created Internal Intention: '{intent_content}' (Urgency: {intention.urgency:.2f}, Salience: {intention.incentive_salience:.2f})")

    def _get_emotional_similary_memory(self, mem: CoversTicksEventsKnoxel, target_tick_id: int):
        state = self._get_state_at_tick(target_tick_id)

        # target
        base_avg_emotions = {}
        full_state = dict(state.state_emotions.model_dump().items())
        full_state.update(dict(state.state_needs.model_dump().items()))
        full_state.update(dict(state.state_cognition.model_dump().items()))

        for key, val in full_state.items():
            base_avg_emotions[key] = val

        # memory
        avg_emotions = {}
        for tick_id in range(mem.min_tick_id, mem.max_event_id + 1):
            tstate = self._get_state_at_tick(tick_id)
            full_state = dict(tstate.state_emotions.model_dump().items())
            full_state.update(dict(tstate.state_needs.model_dump().items()))
            full_state.update(dict(tstate.state_cognition.model_dump().items()))

            for key, val in full_state.items():
                if key not in avg_emotions:
                    avg_emotions[key] = 0
                avg_emotions[key] += val

        difference = 0
        keys = list(avg_emotions.keys())
        klen = len(keys)
        for key in keys:
            avg_emotions[key] = avg_emotions[key] / klen
            difference += abs(avg_emotions[key] - base_avg_emotions[key])

        return difference

    @profile
    def _gather_memories_for_attention(self):
        """
        Use the context and emotional state to gather all memories that could be relevant.
        """
        self._log_mental_mechanism(self._gather_memories_for_attention, MLevel.Mid, f"Use the context and emotional state to gather all memories that could be relevant.")

        stimulus = self.current_state.primary_stimulus
        stimulus_content = stimulus.content

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)
        query_embedding = main_llm.get_embedding(context_events)

        limit = self.config.remember_per_category_limit

        # --- 1. Gather episodic memory
        if len(self.all_episodic_memories) > 0:
            # Rank topical clusters by relevance
            ranked_episodic_memories = sorted(
                self.all_episodic_memories,
                key=lambda c: cosine_distance(np.array(query_embedding), np.array(c.embedding))
            )

            ranked_episodic_memories = ranked_episodic_memories[:limit]
            for mem in ranked_episodic_memories:
                self.current_state.attention_candidates.add(mem)

            if len(ranked_episodic_memories) > 0:
                self._log_mental_mechanism(
                    self._gather_memories_for_attention, MLevel.Low,
                    f"Retrieved {len(ranked_episodic_memories)} relevant episodic memories. Top: '{ranked_episodic_memories[0].content[:60]}...'"
                )

            # Rank my emotiotional similarity
            ranked_episodic_memories = sorted(
                self.all_episodic_memories,
                key=lambda c: self._get_emotional_similary_memory(c, self.current_tick_id)
            )
            ranked_episodic_memories = ranked_episodic_memories[:limit]
            for mem in ranked_episodic_memories[:limit]:
                self.current_state.attention_candidates.add(mem)

        # --- 2. Gather factual memory
        if len(self.all_declarative_facts) > 0:
            # Rank topical clusters by relevance
            ranked_facts = sorted(
                self.all_declarative_facts,
                key=lambda c: cosine_distance(np.array(query_embedding), np.array(c.embedding))
            )
            ranked_facts = ranked_facts[:limit]
            for f in ranked_facts:
                self.current_state.attention_candidates.add(f)

            if len(ranked_facts) > 0:
                self._log_mental_mechanism(
                    self._gather_memories_for_attention, MLevel.Low,
                    f"Retrieved {len(ranked_facts)} relevant factual memories. Top: '{ranked_facts[0].content[:60]}...'"
                )

            # Rank my emotiotional similarity
            ranked_facts = sorted(
                self.all_declarative_facts,
                key=lambda c: self._get_emotional_similary_memory(c, self.current_tick_id)
            )
            for f in ranked_facts[:limit]:
                self.current_state.attention_candidates.add(f)

    @profile
    def _gather_meta_insights(self):
        """
        Add the most recent and some relevant meta insights to attention candidates.
        """

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)
        query_embedding = main_llm.get_embedding(context_events)

        self._log_mental_mechanism(self._build_structures_get_coalitions, MLevel.Mid, f"Gather meta-insights about own internal mechanisms of simulated consciousness.")
        limit = self.config.remember_per_category_limit

        meta_insights = Enumerable(self.all_features) \
            .where(lambda f: f.feature_type == FeatureType.MetaInsight) \
            .order_by(lambda f: f.tick_id) \
            .to_list()

        for meta in meta_insights[-limit:]:
            self.current_state.attention_candidates.add(meta)

        ranked_meta = sorted(
            meta_insights,
            key=lambda c: cosine_distance(np.array(query_embedding), np.array(c.embedding))
        )
        for meta in ranked_meta[:limit]:
            self.current_state.attention_candidates.add(meta)


    @profile
    def _build_structures_get_coalitions(self):
        """
        Build various structures from all attention candidates by clustering the content.
        Use the cluster centers to build coalitions with a balanced spectrum of feature types.
        Save all coalitions for finding the best one according to attention mechanism.
        """
        self._log_mental_mechanism(self._build_structures_get_coalitions, MLevel.Mid, f"Simulate the structure building codelet of LIDA with agglomerative clustering and build coalitions of attention candidates.")

        unique_candidates = list({n.id: n for n in self.current_state.attention_candidates.to_list()}.values())
        n_clusters = max(1, int(len(unique_candidates) / 8))

        embs = []
        for cand in unique_candidates:
            embs.append(main_llm.get_embedding(cand.content))
        embs = np.array(embs)

        # Calculate cosine distance matrix from contextual embeddings
        cosine_sim_matrix = cosine_similarity(embs)
        # Ensure distances are non-negative and handle potential floating point issues
        cosine_dist_matrix = np.maximum(0, 1 - cosine_sim_matrix)
        np.fill_diagonal(cosine_dist_matrix, 0)  # Ensure zero distance to self

        agg_clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average',  # Average linkage is often good for text topics
            compute_full_tree=True
        )
        labels = agg_clustering.fit_predict(cosine_dist_matrix).tolist()
        n_clusters_found = len(np.unique(labels))

        self._log_mental_mechanism(
            self._build_structures_get_coalitions, MLevel.Mid,
            f"Organized {len(unique_candidates)} attention candidates into {n_clusters_found} competing coalitions."
        )

        cluster_map = {}
        for i in range(len(unique_candidates)):
            lbl = labels[i]
            if lbl not in cluster_map:
                cluster_map[lbl] = []
            cluster_map[lbl].append(unique_candidates[i])

        # get avg embedding of clusters
        cluster_avg_emb = {}
        for cluster, knoxels in cluster_map.items():
            avg_emb = np.mean(np.array([k.embedding for k in knoxels if k.embedding]), axis=0)
            cluster_avg_emb[cluster] = avg_emb

        # shuffle and group candidates
        random.shuffle(unique_candidates)
        type_map = {}
        for cand in unique_candidates:
            cls = cand.__class__
            if isinstance(cand, Feature):
                cls = cand.feature_type
            if cls not in type_map:
                type_map[cls] = []
            type_map[cls].append(cand)

        # prepare coalitions
        coalitions_balanced = {key: [] for key in cluster_map.keys()}
        coalitions_hard = {key: [] for key in cluster_map.keys()}

        # uniformely put into coalitions
        for group, candidates in type_map.items():
            available_clusters = list(cluster_map.keys())

            for cand in candidates:
                ranked_cluster = sorted(
                    available_clusters,
                    key=lambda c: cosine_distance(np.array(cluster_avg_emb[c]), np.array(cand.embedding))
                )
                best_cluster = ranked_cluster[0]
                coalitions_hard[best_cluster].append(cand)
                coalitions_balanced[best_cluster].append(cand)
                available_clusters.remove(best_cluster)
                if len(available_clusters) == 0:
                    available_clusters = list(cluster_map.keys())

        # sort by world time for causal history
        sorted_coalitions_balanced = {-key: sorted(value, key=lambda c: c.timestamp_world_begin) for key, value in coalitions_balanced.items()}
        self.current_state.coalitions_balanced = sorted_coalitions_balanced

        sorted_coalitions_hard = {key: sorted(value, key=lambda c: c.timestamp_world_begin) for key, value in coalitions_hard.items()}
        self.current_state.coalitions_hard = sorted_coalitions_hard

        # Optional but powerful: Log the "theme" of each coalition
        for coal_id, coal_knoxels in sorted_coalitions_hard.items():
            # A simple way to get a theme is to join a few key knoxel contents
            theme_knoxels = Enumerable(coal_knoxels).take(3).select(lambda k: k.content[:30]).to_list()
            theme_summary = "; ".join(theme_knoxels)
            self._log_mental_mechanism(
                self._build_structures_get_coalitions, MLevel.Low,
                f"Coalition {coal_id} Theme: '{theme_summary}...'"
            )



    @profile
    def _simulate_attention_on_coalitions(self):
        """
        Rate the different coalitions by combining attention agent with character traits and emotions.
        Also apply auxilary rating based on rules.
        """
        self._log_mental_mechanism(self._simulate_attention_on_coalitions, MLevel.Mid, f"Simulate the attention mechanism of LIDA by combining mental state and attention narrative.")

        stimulus = self.current_state.primary_stimulus
        stimulus_content = stimulus.content

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, self.config.context_events_similarity_max_tokens)
        query_embedding = main_llm.get_embedding(context_events)

        coalitions = copy.copy(self.current_state.coalitions_balanced)
        coalitions.update(self.current_state.coalitions_hard)

        coal_rating_map = {}

        class AttentionRating(BaseModel):
            reasoning: str = Field(..., description="Brief explanation for the selection.")
            rating: float = Field(..., description="How valueable the coalition is in the situation, with min=0 being bad and max=1 being very good. ")

        for coal_id, coal_knoxels in coalitions.items():
            content = KnoxelList(coal_knoxels).get_story(self)

            # Use cognitive state to influence prompt
            focus_prompt = ""
            if self.current_state.state_cognition.mental_aperture < -0.5:
                focus_prompt = f"{companion_name}'s mental aperature is low. Focus very narrowly on the single most important theme. Coalitions with a central theme score higher than coalitions with balanced cognitive features / memories."
            elif self.current_state.state_cognition.mental_aperture > 0.5:
                focus_prompt = f"{companion_name}'s mental aperature is high. Maintain a broad awareness, selecting coalitions with a diverse range of themes. Coalitions with balanced cognitive features / memories score higher than those with central theme."
            else:
                focus_prompt = "Select the most relevant items for the current situation."

            # modifiers
            # --- MODIFIED: Create more evocative summaries of the internal state for the prompt ---
            strong_emotions_summary = self._verbalize_emotional_state()
            needs_summary_line, cognitive_style_summary = self._verbalize_cognition_and_needs()

            mental_state_prompt_block = f"""
            **{self.config.companion_name}'s Current State of Mind:**
            - **Strong Emotions:** {strong_emotions_summary}
            - **Needs:** {needs_summary_line}
            - **Cognitive Style:** {cognitive_style_summary}
            """

            prompt = f"""
            Character: {self.config.companion_name} ({self.config.universal_character_card})
    
            {self.config.companion_name} has this type of personality when focusing on things:
            {self.get_narrative(NarrativeTypes.AttentionFocus, self.config.companion_name)}
            
            {mental_state_prompt_block}
    
            Current context of conversation:
            ### CONVERSATION BEGIN
            {context_events}
            ### CONVERSATION END
            
            Propsed coalition for broadcasting from pre-conscious Current Situational Model to conscious Global Workspace:
            ### BEGIN COALITIION
            {content}
            ### END COALITION 
            
            TASK: Rate how good the coalition would fit into the conscious Workspace for the current situation and {self.config.companion_name}'s traits! {focus_prompt} Give a short reasoning for your choice of rating.
            """

            attention_rating = 0
            attention_reason = ""

            selection_result = self._call_llm(prompt, output_schema=AttentionRating)
            if selection_result and isinstance(selection_result, AttentionRating):
                attention_rating = selection_result.rating
                attention_reason = selection_result.reasoning

                self._log_mental_mechanism(self._simulate_attention_on_coalitions, MLevel.Debug, f"Attention for ### BEGIN COALITION {content} ### END COALTION Rating: {attention_rating} with reasoning: {attention_reason}")

            aux_rating = 0
            current_tick = self.current_state.tick_id
            for k in coal_knoxels:
                base_score = 0.0

                ticks_ago = current_tick - k.tick_id
                recency_factor = max(0.1, 0.95 ** ticks_ago)
                interlocus = 0

                if isinstance(k, Stimulus) and k == self.current_state.primary_stimulus:
                    base_score = -100 # is included in any case since its causally added already
                elif isinstance(k, Intention):
                    # Salience depends on urgency, salience, and maybe fulfillment status (unmet are more salient)
                    fulfillment_penalty = (1.0 - k.fulfilment)  # Higher penalty for unmet
                    base_score = ((k.urgency + k.incentive_salience) / 2.0) * (0.5 + fulfillment_penalty * 0.5)
                elif isinstance(k, Feature):
                    interlocus = k.interlocus
                    # Feelings, Subjective Experience, and Expectation Outcomes are salient
                    if k.feature_type in [FeatureType.Feeling, FeatureType.SubjectiveExperience, FeatureType.ExpectationOutcome]:
                        base_score = abs(k.affective_valence) if k.affective_valence is not None else 0.3
                        base_score = max(base_score, 0.1)  # Ensure non-zero base score
                    # Narrative updates less salient unless very recent
                    elif k.feature_type == FeatureType.NarrativeUpdate:
                        base_score = 0.15
                    # Other features less salient unless directly related to recent actions/failures
                elif isinstance(k, Narrative):
                    base_score = 0.1  # Background context
                elif isinstance(k, MemoryClusterKnoxel) or isinstance(k, DeclarativeFactKnoxel):
                    base_score = abs(k.affective_valence) * 0.5 if hasattr(k, 'affective_valence') else 0.1

                ego_factor = 1 + self.current_state.state_cognition.ego_strength

                # Combine base score with recency
                final_score = base_score * recency_factor + (abs(max(interlocus, 0)) * ego_factor)
                aux_rating += final_score

            self._log_mental_mechanism(self._simulate_attention_on_coalitions, MLevel.Debug, f"Auxilary Rating for ### BEGIN COALITION {content} ### END COALTION Rating: {aux_rating}")

            coal_rating_map[coal_id] = {
                "attention_rating": attention_rating,
                "aux_rating": aux_rating,
                "attention_reasoning": attention_reason
            }

        # apply aux rating (as tie breaker)
        max_aux = max([r["aux_rating"] for r in coal_rating_map.values()])
        combined_coalition_scored = {key: value["attention_rating"] + ((value["aux_rating"] / max_aux) * self.config.coalition_aux_rating_factor) for key, value in coal_rating_map.items()}

        coalitions_rated = list(coalitions.keys())
        coalitions_rated.sort(key=lambda x: combined_coalition_scored[x], reverse=True)

        winning_coalition = coalitions[coalitions_rated[0]]
        winning_reason = coal_rating_map[coalitions_rated[0]]["attention_reasoning"]
        winning_score = combined_coalition_scored[coalitions_rated[0]]
        winning_content = KnoxelList(winning_coalition).get_story(self)
        self._log_mental_mechanism(self._simulate_attention_on_coalitions, MLevel.Low, f"Coalition selected: ### BEGIN COALITION {winning_content[:64]}... ### END COALTION Winning Rating: {winning_score} with reasoning: {winning_reason}")
        self._log_mental_mechanism(self._simulate_attention_on_coalitions, MLevel.Debug, f"Coalition selected: ### BEGIN COALITION {winning_content} ### END COALTION Winning Rating: {winning_score} with reasoning: {winning_reason}")

        for i in range(1, len(coalitions_rated)):
            discarded_coalition = coalitions[coalitions_rated[0]]
            discarded_reason = coal_rating_map[coalitions_rated[0]]
            discarded_score = combined_coalition_scored[coalitions_rated[0]]
            discarded_content = KnoxelList(discarded_coalition).get_story(self)
            self._log_mental_mechanism(self._simulate_attention_on_coalitions, MLevel.Debug, f"Coalition discarded, placed {i}: ### BEGIN COALITION {discarded_content} ### END COALTION Discarded Rating: {discarded_score} with reasoning: {discarded_reason}")

        self.current_state.conscious_workspace = KnoxelList()
        for k in winning_coalition:
            ft = None
            il = -1
            if isinstance(k, Intention):
                if k.internal:
                    ft = FeatureType.Goal
                else:
                    ft = FeatureType.Expectation
            elif isinstance(k, Feature):
                ft = k.feature_type
                il = k.interlocus
            elif isinstance(k, Narrative):
                ft = FeatureType.Narrative
            elif isinstance(k, MemoryClusterKnoxel) or isinstance(k, DeclarativeFactKnoxel):
                ft = FeatureType.MemoryRecall

            if ft is None:
                raise Exception("CAN'T CONVERT TO CAUSAL: " + k.get_story_element(self))
            else:
                focus_feature = Feature(
                    content=k.content,
                    feature_type=ft,
                    interlocus=il,
                    causal=True,
                    source=self.config.companion_name
                )
                self.add_knoxel(focus_feature)
                self.current_state.conscious_workspace.add(focus_feature)

        # Create AttentionFocus feature
        focus_content = f"{self.config.companion_name} brings their attention to these mental features and aspects with this reason: '{winning_reason}'."
        focus_feature = Feature(
            content=focus_content,
            feature_type=FeatureType.AttentionFocus,
            interlocus=-1,
            causal=True,
            source = self.config.companion_name
        )
        self.add_knoxel(focus_feature)
        self.current_state.conscious_workspace.add(focus_feature)

    @profile
    def _get_situational_embedding(
        self,
        max_recent_dialogue_turns: int = 3,
        max_active_intentions: int = 3, # Includes goals and expectations
        max_salient_narratives: int = 2,
        include_subjective_experience: bool = True,
        include_emotional_summary: bool = True,
        include_recent_feelings: bool = True
    ) -> Optional[List[float]]:
        """
        Generates a comprehensive embedding representing the semantic essence of the current situation.
        It combines primary stimulus, recent dialogue, subjective experience, emotions,
        active intentions/expectations, and salient narratives into a single text block for embedding.
        """
        if not self.current_state:
            logging.warning("Cannot generate situational embedding: current_state is None.")
            return None

        situational_parts = []

        # 1. Primary Stimulus (if any)
        if self.current_state.primary_stimulus:
            situational_parts.append(f"Current Focus/Stimulus: {self.current_state.primary_stimulus.content}")

        # 2. Recent Dialogue Context
        dialogue_features = Enumerable(self.all_features) \
            .where(lambda f: f.feature_type == FeatureType.Dialogue and f.tick_id <= self.current_state.tick_id) \
            .order_by_descending(lambda f: f.tick_id) \
            .then_by_descending(lambda f: f.id) \
            .take(max_recent_dialogue_turns * 2) \
            .to_list()
        dialogue_features.reverse() # Put them back in chronological order

        if dialogue_features:
            dialogue_context_str = "\n".join([
                f"  - {f.source}: \"{f.content}\"" for f in dialogue_features[-max_recent_dialogue_turns:] # Ensure correct limit
            ])
            situational_parts.append(f"Recent Conversation:\n{dialogue_context_str}")

        # 3. Subjective Experience
        if include_subjective_experience and self.current_state.subjective_experience:
            situational_parts.append(f"My Current Thoughts/Experience: {self.current_state.subjective_experience.content}")

        # 4. Emotional State Summary
        if include_emotional_summary:
            emotions = self.current_state.state_emotions
            emotion_summary = (
                f"Current Emotional State: "
                f"Overall feeling {emotions.get_overall_valence():.2f} (Valence: {emotions.valence:.2f}, "
                f"Affection: {emotions.affection:.2f}, Self-Worth: {emotions.self_worth:.2f}, "
                f"Trust: {emotions.trust:.2f}, Disgust: {emotions.disgust:.2f}, Anxiety: {emotions.anxiety:.2f})."
            )
            situational_parts.append(emotion_summary)

        # 5. Recent Specific Feelings (as features)
        if include_recent_feelings:
            recent_feeling_features = Enumerable(self.all_features) \
                .where(lambda f: f.feature_type == FeatureType.Feeling and \
                                 f.tick_id >= self.current_state.tick_id - 1 and # Current or previous tick
                                 f.tick_id <= self.current_state.tick_id) \
                .order_by_descending(lambda f: f.id) \
                .take(2) \
                .to_list()
            if recent_feeling_features:
                feeling_strs = [f"  - Recently felt: {f.content} (Valence: {f.affective_valence or 0.0:.2f})"
                                for f in recent_feeling_features]
                situational_parts.append("Specific Recent Feelings:\n" + "\n".join(feeling_strs))

        # 6. Active Intentions (Internal Goals and External Expectations)
        # Prioritize by a combination of urgency, salience, and unfulfillment
        active_intentions = []
        for intent in self.all_intentions:
            if intent.fulfilment < 0.95: # Consider nearly fulfilled as still active for context
                # Score favors urgency, then salience, then how unfulfilled it is
                score = (intent.urgency * 0.5) + \
                        (intent.incentive_salience * 0.3) + \
                        ((1.0 - intent.fulfilment) * 0.2)
                active_intentions.append((score, intent))

        active_intentions.sort(key=lambda x: x[0], reverse=True)

        if active_intentions:
            intention_strs = []
            for i, (score, intent) in enumerate(active_intentions):
                if i >= max_active_intentions:
                    break
                prefix = "My Goal:" if intent.internal else "I'm Expecting:"
                intention_strs.append(f"  - {prefix} {intent.content} (Urgency: {intent.urgency:.2f}, Salience: {intent.incentive_salience:.2f}, Fulfilled: {intent.fulfilment:.2f})")
            if intention_strs:
                situational_parts.append("Current Goals and Expectations:\n" + "\n".join(intention_strs))

        # 7. Salient Narratives (e.g., from conscious workspace or highly relevant)
        salient_narratives_list = []
        # Option 1: Narratives in conscious workspace
        if self.current_state.conscious_workspace:
            for knoxel in self.current_state.conscious_workspace.to_list():
                if isinstance(knoxel, Narrative):
                    salient_narratives_list.append(knoxel)

        # Option 2: Retrieve most relevant narratives if primary stimulus embedding exists
        # (This might be redundant if workspace already captures them, but could be an alternative)
        # if not salient_narratives_list and self.current_state.primary_stimulus and self.current_state.primary_stimulus.embedding:
        #     salient_narratives_list = self._get_relevant_knoxels(
        #         self.current_state.primary_stimulus.embedding,
        #         self.all_narratives,
        #         max_salient_narratives
        #     )

        if salient_narratives_list:
            # Deduplicate and limit
            unique_narratives = list({n.id: n for n in salient_narratives_list}.values())[:max_salient_narratives]
            narrative_strs = [
                f"  - Relevant Narrative ({n.narrative_type.name} about {n.target_name}): {n.content[:150]}..." # Truncate for brevity
                for n in unique_narratives
            ]
            if narrative_strs:
                situational_parts.append("Key Background Narratives:\n" + "\n".join(narrative_strs))

        # --- Combine all parts into a single text block ---
        if not situational_parts:
            logging.warning("No situational parts to construct text for embedding.")
            # Fallback: use only primary stimulus if available
            if self.current_state.primary_stimulus and self.current_state.primary_stimulus.content:
                situational_text = self.current_state.primary_stimulus.content
            else:
                return None # Or a zero vector, or random
        else:
            situational_text = "\n\n".join(situational_parts)

        logging.debug(f"--- Text for Situational Embedding ---\n{situational_text}\n------------------------------------")
        return self.llm_manager.get_embedding(situational_text)

    def _generate_state_seed_phrases_simple(self, state: GhostState) -> List[str]:
        """
        Analyzes the current state and generates a list of descriptive seed phrases
        to guide the generation of a subjective experience (thought).
        """
        phrases = []
        cognition = state.state_cognition
        emotions = state.state_emotions
        needs = state.state_needs

        # --- Cognitive Style Phrases ---
        if cognition.mental_aperture < -0.6:
            phrases.append("My focus narrows down to one thing...")
        elif cognition.mental_aperture > 0.6:
            phrases.append("My thoughts feel like they're branching out, connecting ideas...")

        if cognition.interlocus < -0.6:
            phrases.append("I turn my attention inward, reflecting on...")
        elif cognition.interlocus > 0.6:
            phrases.append("I'm completely focused on what's happening outside of me...")

        if cognition.ego_strength < 0.3:
            phrases.append("I need to be careful and just provide the facts...")
        elif cognition.ego_strength > 0.8:
            phrases.append("I feel a strong sense of self in this moment...")

        if cognition.willpower < 0.3:
            phrases.append("I feel mentally drained, it's hard to think clearly...")

        # --- Dominant Emotion Phrases ---
        # Find the single most dominant emotion to comment on
        dominant_emotion_value = 0
        dominant_emotion_phrase = ""
        # Check bipolar axes
        for axis, names in {'valence': ('happiness', 'sadness'), 'affection': ('affection', 'dislike')}.items():
            value = getattr(emotions, axis)

            desc = "apparent"
            if abs(value) > 0.7:
                desc = "strong"
            if abs(value) > 0.8:
                desc = "very strong"
            if abs(value) > 0.9:
                desc = "overwhelming"

            if abs(value) > 0.6 and abs(value) > abs(dominant_emotion_value):
                dominant_emotion_value = value
                dominant_emotion_phrase = f"This feeling of {names[0] if value > 0 else names[1]} is {desc}..."
        # Check unipolar axes
        if emotions.anxiety > 0.7 and emotions.anxiety > abs(dominant_emotion_value):
            dominant_emotion_phrase = "A wave of anxiety washes over me..."

        if dominant_emotion_phrase:
            phrases.append(dominant_emotion_phrase)

        # --- Pressing Needs Phrases ---
        pressing_needs = [name.replace('_', ' ') for name, field in needs.__class__.model_fields.items() if getattr(needs, name) < 0.3]
        if pressing_needs:
            # Example: "I have a deep need for connection right now..."
            phrases.append(f"I have a deep need for {pressing_needs[0]} right now...")

        return phrases

    def _generate_state_seed_phrases(self, state: GhostState) -> List[str]:
        """
        Analyzes the current state and generates a list of descriptive, gradual seed phrases
        to guide the generation of a subjective experience (thought).
        """
        phrases = []
        cognition = state.state_cognition
        emotions = state.state_emotions
        needs = state.state_needs

        # --- Helper function for mapping values to descriptive adverbs/adjectives ---
        def get_intensity_adverb(value: float, thresholds: List[Tuple[float, str]]) -> Optional[str]:
            """Finds the appropriate descriptive word for a given absolute value."""
            abs_val = abs(value)
            for threshold, adverb in thresholds:
                if abs_val >= threshold:
                    return adverb
            return None

        # --- 1. Cognitive Style Phrases (Gradual) ---
        cog_phrases = {
            "mental_aperture": {
                "pos": ([(0.8, "incredibly broad"), (0.5, "broad")], "My thoughts feel {adverb}, branching out..."),
                "neg": ([(0.8, "intensely narrow"), (0.5, "narrow")], "My focus becomes {adverb}, zoning in on one thing...")
            },
            "interlocus": {
                "pos": ([(0.8, "completely"), (0.5, "intently")], "I'm {adverb} focused on what's happening outside of me..."),
                "neg": ([(0.8, "deeply"), (0.5, "quietly")], "I turn my attention inward, feeling {adverb} reflective...")
            },
            "ego_strength": {
                "pos": ([(0.8, "strong"), (0.6, "clear")], "I feel a {adverb} sense of self in this moment..."),
                "neg": ([(0.3, "a need to be objective"), (0.5, "a sense of detachment")], "I feel {adverb}, setting my personality aside...")
            },
            "willpower": {
                "neg": ([(0.2, "completely drained"), (0.4, "mentally tired")], "I feel {adverb}, it's hard to muster the energy...")
            }
        }

        for axis, configs in cog_phrases.items():
            value = getattr(cognition, axis)
            config_key = "pos" if value > 0 else "neg"

            if config_key in configs:
                thresholds, template = configs[config_key]
                adverb = get_intensity_adverb(value, thresholds)
                if adverb:
                    phrases.append(template.format(adverb=adverb))

        # --- 2. Dominant Emotion Phrases (Gradual) ---
        # Find the single most dominant emotion to comment on
        emotion_candidates = []

        # Bipolar axes
        bipolar_map = {
            'valence': ('happiness', 'sadness'), 'affection': ('affection', 'dislike'),
            'self_worth': ('pride', 'shame'), 'trust': ('trust', 'distrust')
        }
        for axis, (pos_name, neg_name) in bipolar_map.items():
            value = getattr(emotions, axis)
            if abs(value) > 0.4:  # Minimum threshold to be considered
                emotion_candidates.append({'value': abs(value), 'name': pos_name if value > 0 else neg_name})

        # Unipolar axes
        if emotions.anxiety > 0.4:
            emotion_candidates.append({'value': emotions.anxiety, 'name': 'anxiety'})
        if emotions.disgust > 0.4:
            emotion_candidates.append({'value': emotions.disgust, 'name': 'disgust'})

        if emotion_candidates:
            # Sort to find the most intense emotion
            emotion_candidates.sort(key=lambda x: x['value'], reverse=True)
            dominant_emotion = emotion_candidates[0]

            # Use gradual intensity descriptors
            intensity_desc = get_intensity_adverb(dominant_emotion['value'], [
                (0.9, "an overwhelming"), (0.8, "an intense"),
                (0.7, "a deep"), (0.5, "a clear")
            ]) or "a"  # Fallback to 'a'

            # Use different sentence structures for different emotions
            if dominant_emotion['name'] == 'anxiety':
                phrases.append(f"{intensity_desc.capitalize()} wave of anxiety washes over me...")
            else:
                phrases.append(f"A feeling of {dominant_emotion['name']} is {intensity_desc}ly present...")

        # --- 3. Pressing Needs Phrases (Gradual) ---
        pressing_needs_candidates = [
            (name.replace('_', ' '), getattr(needs, name))
            for name in needs.__class__.model_fields
        ]
        # Sort by how low the need is
        pressing_needs_candidates.sort(key=lambda x: x[1])

        lowest_need_name, lowest_need_value = pressing_needs_candidates[0]

        # Describe the need only if it's significantly low
        need_intensity_desc = get_intensity_adverb(1.0 - lowest_need_value, [  # Invert value for intensity
            (0.9, "an overwhelming"), (0.8, "a desperate"),
            (0.7, "a deep"), (0.6, "a distinct")
        ])

        if need_intensity_desc:
            phrases.append(f"I feel {need_intensity_desc} need for {lowest_need_name} right now...")

        return phrases

    @profile
    def _generate_subjective_experience(self):
        """Generates a narrative description of the current moment based on workspace and state."""

        self._log_mental_mechanism(self._generate_subjective_experience, MLevel.Mid, f"Convert the conscius workspace to a brief subjective experience (thought).")

        workspace_knoxels = self.current_state.conscious_workspace.to_list()
        workspace_content = self.current_state.conscious_workspace.get_story(self)

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self, 1024)

        seed_phrases = self._generate_state_seed_phrases(self.current_state)
        seed_phrase = ""
        if len(seed_phrases) > 0:
            s = ". ".join(seed_phrases)
            seed_phrase = f"{s}\nAnd the current conversation"

            self._log_mental_mechanism(
                self._generate_subjective_experience, MLevel.Low,
                f"Guiding thought generation with seed phrases: '{seed_phrase}'"
            )

        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions}
        Current Needs State: {self.current_state.state_needs}

        Story context: {context_events}

        Conscious Workspace Content: {workspace_content}

        Task: Synthesize the current state and conscious thoughts into a brief, first-person narrative paragraph describing {self.config.companion_name}'s subjective experience *right now*. Capture the flow of thought, feeling, and anticipation/reaction. Make it sound natural and introspective.
        Output *only* the narrative paragraph.
        """
        experience_content = seed_phrase + self._call_llm(prompt, temperature=0.75, max_tokens=256, seed_phrase=seed_phrase)
        subjective_feature = Feature(
            content=experience_content,
            feature_type=FeatureType.Thought,
            affective_valence=self.current_state.state_emotions.get_overall_valence(),
            interlocus=-1, causal=True, source=self.config.companion_name
        )
        self.add_knoxel(subjective_feature)
        self.current_state.conscious_workspace.add(subjective_feature)
        self.current_state.subjective_experience = subjective_feature

        self._log_mental_mechanism(self._generate_subjective_experience, MLevel.Low, f"Generated Subjective Experience: {experience_content}...")

    @profile
    def _check_possible_tool_call(self) -> Tuple[bool, str]:
        if mcp_server_url is None or mcp_server_url == "":
            return False, ""

        stimulus = self.current_state.primary_stimulus

        ts = get_toolset_from_url(mcp_server_url)
        if ts is None:
            return False, ""

        self._log_mental_mechanism(self._check_possible_tool_call, MLevel.High, f"Determining best tool...")

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self, 1024)

        # --- 2. Construct the Prompt ---
        sysprompt = f"""You check the conversation between the AI {self.config.companion_name} and the user {self.config.user_name} and find out if there is a tool call is advised for {self.config.companion_name}.
        You MUST output a valid JSON object containing your evaluations for ALL provided tools."""

        user_prompt = f"""
        **Current Situation:**
          Story context: {context_events}
        - **Stimulus:** ({stimulus.stimulus_type}) "{stimulus.content}"

        **4. Tools to Evaluate:**
        `{ts.tool_select_docs}`
        """

        comp_settings = CommonCompSettings(temperature=0.2, max_tokens=1024)
        _, calls = self.llm_manager.completion_tool(
            LlmPreset.Default,
            inp=[("system", sysprompt), ("user", user_prompt)],
            comp_settings=comp_settings,
            tools=[ts.tool_select_model]
        )

        tools_results = calls[0]
        result_dict = tools_results.model_dump()

        # --- 4. Extract and Compare Ratings ---
        self._log_mental_mechanism(
            self._check_possible_tool_call, MLevel.Debug,
            f"LLM Reasoning for Action Choice: {result_dict}"
        )

        ranking = [(tool_name, tool_rating) for tool_name, tool_rating in result_dict.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)

        if ranking[0][1] < 0.5 or ranking[0][0] == "user_reply":
            self._log_mental_mechanism(
                self._check_possible_tool_call, MLevel.High,
                f"No suitable tool found..."
            )
            return False, ""
        else:
            self._log_mental_mechanism(
                self._check_possible_tool_call, MLevel.High,
                f"Selected tool: '{ranking[0][0]}' with score {ranking[0][1]:.2f}"
            )
            return True, ranking[0][0]

    def _get_valid_action_pool(self, stimulus: Stimulus) -> List[ActionType]:
        """
        Determines the pool of valid high-level actions based on the stimulus type.
        """
        stimulus_type = stimulus.stimulus_type
        self._log_mental_mechanism(self._get_valid_action_pool, MLevel.Debug, f"Getting valid action pool for stimulus type: {stimulus_type}")

        # If the stimulus is an external message from the user
        if stimulus_type in [StimulusType.UserMessage]:
            # The AI can reply, ignore, or decide to think more deeply before replying.
            # It cannot proactively "start" a conversation it's already in.
            return [
                ActionType.Reply,
                ActionType.Ignore
            ]

        # If the stimulus is internal (e.g., from the Shell Agent)
        elif stimulus_type in [StimulusType.SystemMessage, StimulusType.TimeOfDayChange, StimulusType.WakeUp, StimulusType.LowNeedTrigger]:
            # The AI is not in an active conversation, so it can choose to start one,
            # contemplate, sleep, or continue to do nothing. "Reply" is not valid.
            return [
                ActionType.InitiateUserConversation,
                ActionType.InitiateInternalContemplation,
                ActionType.Sleep,
                ActionType.Ignore,
                ActionType.ToolCall
            ]

        # Default fallback, should ideally not be reached
        return [ActionType.Ignore]

    @profile
    def _determine_best_action_type(self) -> ActionType:
        """
        Acts as a high-level "Behavioral Steering" module. It evaluates a pool of
        valid action types against the current cognitive state and returns the most
        appropriate one to guide the next stage of deliberation.
        """
        stimulus = self.current_state.primary_stimulus
        action_pool = self._get_valid_action_pool(stimulus)

        self._log_mental_mechanism(self._determine_best_action_type, MLevel.High, f"Determining best action type from pool: {[a.value for a in action_pool]}")

        # --- 1. Gather Rich Context for the Decision ---

        # Use the verbalization helpers to create a rich summary of the internal state
        strong_emotions_summary = self._verbalize_emotional_state()
        needs_summary, cognitive_style_summary = self._verbalize_cognition_and_needs()

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self, 1024)

        mental_state_prompt_block = f"""
    - **Emotions:** {strong_emotions_summary}
    - **Pressing Needs:** {needs_summary}
    - **Cognitive Style:** {cognitive_style_summary if cognitive_style_summary else "Normal."}"""

        # Get active internal goals from the conscious workspace or attention candidates
        active_intentions = Enumerable(self.current_state.attention_candidates.to_list()) \
            .where(lambda k: isinstance(k, Intention) and k.internal and k.fulfilment < 0.9) \
            .order_by_descending(lambda i: i.urgency) \
            .take(3) \
            .to_list()
        intention_summary = "; ".join([f"'{i.content}'" for i in active_intentions]) if active_intentions else "None"

        # Get key behavioral narratives
        behavior_narr = self.get_narrative(NarrativeTypes.BehaviorActionSelection, self.config.companion_name)
        conflict_narr = self.get_narrative(NarrativeTypes.ConflictResolution, self.config.companion_name)

        # --- 2. Construct the Prompt ---
        sysprompt = f"""You are a "Behavioral Steering" codelet for the AI {self.config.companion_name}.
    Your task is to evaluate a set of possible high-level actions based on the AI's internal state, goals, and personality.
    You MUST output a valid JSON object containing your evaluations for ALL provided actions."""

        user_prompt = f"""
    **Behavioral Steering Task:** Evaluate the following actions.

    **1. Current State of Mind:**
    {mental_state_prompt_block}
    - **Active Internal Goals:** {intention_summary}

    **2. Core Personality for Action Selection:**
    - **General Behavior:** {behavior_narr.content if behavior_narr else "Be helpful and engaging."}
    - **In Conflict:** {conflict_narr.content if conflict_narr else "Try to de-escalate and understand."}

    **3. Current Situation:**
      Story context: {context_events}
    - **Stimulus:** ({stimulus.stimulus_type}) "{stimulus.content}"

    **4. Action Pool to Evaluate:**
    `{[action.value for action in action_pool]}`
    """

        class ActionTypeSelectionOld(BaseModel):
            reasoning: str = Field(description="Why the rating is as it as and how you came to the decision.")
            rating_Reply: float = Field(description="")
            rating_Ignore: float = Field(description="")
            rating_ToolCallAndRepl: float = Field(description="")
            rating_InitiateUserConversation: float = Field(description="")
            rating_InitiateInternalContemplation: float = Field(description="")
            rating_Sleep: float = Field(description="")
            rating_ToolCall: float = Field(description="")

        class ActionTypeSelection(BaseModel):
            """
            A schema to evaluate the appropriateness of a set of potential high-level actions.
            The LLM must provide a rating for every action listed in the prompt's 'Action Pool to Evaluate'.
            Ratings should be a float between 0.0 (completely inappropriate) and 1.0 (perfectly appropriate).
            """
            holistic_reasoning: str = Field(description="Your final, holistic justification. First, summarize the AI's current state and main goal. Then, explain WHY the highest-rated action is the best strategic choice compared to the others.")

            rating_Reply: float = Field(
                ge=0.0, le=1.0,
                description=f"Rate this action if the AI is in an active conversation. This is the standard action for responding directly to a user's message. It is highly appropriate if the user just said something that requires an answer."
            )
            rating_InitiateUserConversation: float = Field(
                ge=0.0, le=1.0,
                description=f"Rate this action ONLY if the AI is NOT in an active conversation (e.g., the stimulus is from a timeout or internal need). This is for proactively starting a new conversation. It is appropriate if the AI's 'connection' need is very low."
            )
            rating_InitiateInternalContemplation: float = Field(
                ge=0.0, le=1.0,
                description="Rate this action if the AI needs to 'think before acting'. This is appropriate for complex, confusing, or emotionally heavy stimuli that require internal planning or self-regulation before an external action is taken."
            )
            rating_Ignore: float = Field(
                ge=0.0, le=1.0,
                description="Rate this action to make the AI do nothing externally. This is appropriate for insignificant stimuli, or as a deliberate strategy to de-escalate conflict or conserve mental energy when willpower is very low."
            )
            rating_Sleep: float = Field(
                ge=0.0, le=1.0,
                description="Rate this action ONLY if the AI is NOT in an active conversation. This is for entering a long-term idle state. It is appropriate only if willpower is critically low and there are no urgent goals."
            )

        # --- 3. Call LLM and Process Results ---
        comp_settings = CommonCompSettings(temperature=0.2, max_tokens=1024)
        _, calls = self.llm_manager.completion_tool(
            LlmPreset.Default,
            inp=[("system", sysprompt), ("user", user_prompt)],
            comp_settings=comp_settings,
            tools=[ActionTypeSelection]
        )

        action_ratings_result = calls[0]
        result_dict = action_ratings_result.model_dump()

        # --- 4. Extract and Compare Ratings ---

        self._log_mental_mechanism(
            self._determine_best_action_type, MLevel.Low,
            f"LLM Reasoning for Action Choice: {result_dict.get('holistic_reasoning', 'N/A')}"
        )

        ratings: List[Tuple[ActionType, float]] = []

        # Iterate through the valid action pool and extract the corresponding rating from the result
        for action_type in action_pool:
            field_name = f"rating_{action_type.value}"
            rating_value = result_dict.get(field_name)

            if rating_value is not None:
                ratings.append((action_type, rating_value))
                self._log_mental_mechanism(
                    self._determine_best_action_type, MLevel.Debug,
                    f"Action candidate '{action_type.value}' rated {rating_value:.2f}."
                )
            else:
                # The LLM failed to rate a required action. This is a failure of instruction-following.
                logging.warning(f"LLM failed to provide a rating for '{action_type.value}', which was in the requested pool. Assigning a penalty score of 0.0.")
                ratings.append((action_type, 0.0))

        # Sort to find the best action by its rating
        ratings.sort(key=lambda x: x[1], reverse=True)
        best_action_type, best_rating = ratings[0]

        self._log_mental_mechanism(
            self._determine_best_action_type, MLevel.High,
            f"Selected action type: '{best_action_type.value}' with score {best_rating:.2f}"
        )

        if best_action_type == ActionType.Reply:
            possible_tool_call, tool_name = self._check_possible_tool_call()
            if possible_tool_call:
                best_action_type = ActionType.ToolCallAndReply
                experience_content = f"I could call the tool '{tool_name}' for this."
                subjective_feature = Feature(
                    content=experience_content,
                    feature_type=FeatureType.Thought,
                    affective_valence=self.current_state.state_emotions.get_overall_valence(),
                    interlocus=-1, causal=True, source=self.config.companion_name
                )
                self.add_knoxel(subjective_feature)
                self.current_state.conscious_workspace.add(subjective_feature)
                self.current_state.subjective_experience_tool = subjective_feature

        return best_action_type

    @profile
    def _perform_tool_call_and_reply(self) -> Optional[Dict[str, Any]]:
        sysprompt_third_person = self.config.universal_character_card
        sysprompt_third_person = sysprompt_third_person.replace(f"{companion_name} is", "You are")
        sysprompt_third_person = sysprompt_third_person.replace(f"{companion_name}", "You")

        sysprompt = sysprompt_third_person

        sys_tokens = get_token_count(sysprompt)
        left_tokens = context_size - sys_tokens

        memory_embedding = self._get_situational_embedding()
        turns = self._build_conscious_content_queue_assistant(self.config, conscious_workspace_content=self.current_state.conscious_workspace, query_embedding=memory_embedding, max_total_tokens=left_tokens)
        msgs = [("system", sysprompt)]
        msgs.extend(turns)

        comp_settings = CommonCompSettings(temperature=0.6, repeat_penalty=1.05, max_tokens=2000)
        ass, story = self.llm_manager.completion_agentic(LlmPreset.Default, mcp_server_url, msgs, comp_settings=comp_settings)

        predicted_ai_emotion = EmotionalAxesModel(
            valence=1,
            affection=1,
            self_worth=1,
            trust=0.7,
            disgust=0,
            anxiety=0
        )

        res = {
            "sim_id": 0,
            "action_type": ActionType.Reply,
            "ai_reply_content": story,
            "simulated_user_reaction": "",
            "predicted_ai_emotion": predicted_ai_emotion,
            "intent_fulfillment_score": 1,
            "needs_fulfillment_score": 1,
            "cognitive_congruence_score": 1,
            "final_score": 1
        }

        return res


    @profile
    def _deliberate_and_select_reply(self) -> Optional[Dict[str, Any]]:
        """
        Simulates potential actions (Reply, Ignore), rates them using narrative-informed prompts,
        and selects the best one based on predicted outcomes. (Expectation generation happens in _execute_action).
        """
        self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Mid, f"Use the workspace and the intentions to simulate options (internally, non-causal) and find option that best serves to fulfill intentions.")

        workspace_knoxels = self.current_state.conscious_workspace.to_list()
        workspace_content = self.current_state.conscious_workspace.get_story(self)

        context = self._get_context_from_latest_causal_events()
        context_events = context.get_story(self.config, 1024)

        current_emotions = self.current_state.state_emotions
        current_needs = self.current_state.state_needs
        current_cognition = self.current_state.state_cognition

        intentions = [k for k in workspace_knoxels if isinstance(k, Intention) and k.internal]  # Only internal goals influence action choice directly here
        intention_summary = "; ".join([f"'{i.content}' (Urgency: {i.urgency:.2f})" for i in intentions]) if intentions else "None"

        # --- Determine Importance & Simulation Count (Unchanged) ---
        stimulus_valence = 0.0
        if self.current_state.primary_stimulus:
            stim_feeling = Enumerable(self.all_features).where(
                lambda f: f.feature_type == FeatureType.Feeling and f.tick_id == self.current_state.tick_id and "stimulus" in f.content.lower()
            ).first_or_default()
            if stim_feeling and stim_feeling.affective_valence is not None:
                stimulus_valence = stim_feeling.affective_valence
        max_intention_urgency = max([i.urgency for i in intentions] + [0.0])
        importance_score = max(abs(stimulus_valence), max_intention_urgency)

        num_simulations = 1
        if self.stimulus_triage == StimulusTriage.Moderate:
            num_simulations = self.config.min_simulations_per_reply
            if importance_score > self.config.importance_threshold_more_sims:
                num_simulations = self.config.mid_simulations_per_reply
        elif self.stimulus_triage == StimulusTriage.Significant:
            num_simulations = self.config.mid_simulations_per_reply
            if importance_score > self.config.importance_threshold_more_sims:
                num_simulations = self.config.max_simulations_per_reply

        self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Mid, f"Action deliberation importance: {importance_score:.2f}. Running {num_simulations} simulation(s) for Reply.")

        # --- Prepare Narratives for Prompts (Unchanged) ---
        ai_bhv_narr = self.get_narrative(NarrativeTypes.BehaviorActionSelection, self.config.companion_name)
        ai_self_narr = self.get_narrative(NarrativeTypes.SelfImage, self.config.companion_name)
        ai_psy_narr = self.get_narrative(NarrativeTypes.PsychologicalAnalysis, self.config.companion_name)
        ai_rel_narr = self.get_narrative(NarrativeTypes.Relations, self.config.companion_name)
        user_psy_narr = self.get_narrative(NarrativeTypes.PsychologicalAnalysis, self.config.user_name)

        # --- Candidate Actions & Ratings ---
        action_options = []

        # 1. Evaluate "Ignore" Action (Unchanged)
        if True:
            ignore_score = 0
            predicted_ignore_emotion = current_emotions.model_copy()
            if current_needs.connection < 0.4:
                predicted_ignore_emotion.anxiety = min(1.0, predicted_ignore_emotion.anxiety + 0.1)
                predicted_ignore_emotion.valence = max(-1.0, predicted_ignore_emotion.valence - 0.05)
                ignore_score -= 0.1
            elif current_emotions.anxiety > 0.7:
                predicted_ignore_emotion.anxiety = max(0.0, predicted_ignore_emotion.anxiety - 0.1)
                ignore_score += 0.1
            ignore_intent_fulfillment = 0.0
            for intent in intentions:
                if "ignore" in intent.content.lower() or "disengage" in intent.content.lower() or "calm down" in intent.content.lower():
                    ignore_intent_fulfillment = max(ignore_intent_fulfillment, intent.urgency * 0.8)
            ignore_emotion_score = 0.5 + (predicted_ignore_emotion.get_overall_valence() - current_emotions.get_overall_valence()) * 0.5 - (
                    predicted_ignore_emotion.anxiety - current_emotions.anxiety) * 0.3
            final_ignore_score = (ignore_intent_fulfillment * 0.6) + (ignore_emotion_score * 0.4)
            final_ignore_score = max(0.0, min(1.0, final_ignore_score))

            #action_options.append({
            #    "sim_id": "ignore_0",
            #    "action_type": ActionType.Ignore,
            #    "ai_reply_content": "[Decides to ignore/not respond externally]",
            #    "simulated_user_reaction": "N/A",
            #    "predicted_ai_emotion": predicted_ignore_emotion,
            #    "intent_fulfillment_score": ignore_intent_fulfillment,
            #    "final_score": final_ignore_score
            #})
            logging.info(f"Evaluated {ActionType.Ignore}: Score={final_ignore_score:.3f}, IntentFulfill={ignore_intent_fulfillment:.2f}, EmotionScore={ignore_emotion_score:.2f}")

        ai_reply_content = ""
        simulated_user_reaction = ""

        # 2. Evaluate "Reply" Action (Monte Carlo Simulations - Unchanged core simulation logic)
        preferred_action_type = ActionType.Reply
        simulation_results = []
        for i in range(num_simulations):
            sim_id = f"reply_{i}"
            self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Debug, f"--- Running Reply Simulation {sim_id} ---")

            # --- Generate Simulated Reply and User Reaction (Unchanged prompt/parsing) ---
            if preferred_action_type == ActionType.Reply:
                sysprompt = f"""You are an expert story writer. You write a endearing story about the interactions between a next-gen AI companion {self.config.companion_name} and their user {self.config.user_name}.
                The AI {self.config.companion_name} is very advances and has a cognitive architecture based on LIDA with functional self awareness and all aspects of human cognition, emotion and needs.
                Use dialogue in “quotes,” narration in *italics*, and thoughts in (parentheses).
                One reply only, 2–3 paragraphs, with about 50 % dialogue."""

                user_prompt = f"""
                **{self.config.companion_name}'s Character:** {self.config.universal_character_card}
                **{self.config.companion_name}'s Typical Behavior:** {ai_bhv_narr.content if ai_bhv_narr else 'Default behavior.'}
                **Current Emotional State:** {current_emotions}
                **Current Needs State:** {current_needs}
                **Current Cognitive State:** {current_cognition}
                **Relationship ({self.config.companion_name} -> {self.config.user_name}):** {ai_rel_narr.content if ai_rel_narr else 'Default relationship view.'}
                **Analysis of {self.config.user_name}:** {user_psy_narr.content if user_psy_narr else 'Default user analysis.'}
                **Active Internal Goals:** {intention_summary}
                """

                sys_tokens = get_token_count(sysprompt)
                user_tokens = get_token_count(user_prompt)
                left_tokens = context_size - (sys_tokens + user_tokens)

                memory_embedding = self._get_situational_embedding()
                assistant_prompt = self._build_conscious_content_queue(self.config, conscious_workspace_content=self.current_state.conscious_workspace, query_embedding=memory_embedding, max_total_tokens=left_tokens)
                assistant_prompt += f"\n{self.config.companion_name} says: “"

                msgs = [("system", sysprompt),
                        ("user", user_prompt),
                        ("assistant", assistant_prompt)]

                comp_settings = CommonCompSettings(temperature=0.7, repeat_penalty=1.05, max_tokens=1024)

                utterances = []
                while True:
                    res = "“" + self.llm_manager.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
                    utterances = extract_utterances(res)
                    if len(utterances) > 0:
                        break

                ai_reply_content = utterances[0]

                tmp = []
                if len(utterances) > 1:
                    tmp.append(utterances[1])
                else:
                    tmp_assistant_prompt = assistant_prompt + ai_reply_content + f"\n{self.config.user_name} says: “"
                    msgs = [("system", sysprompt),
                            ("user", user_prompt),
                            ("assistant", tmp_assistant_prompt)]

                    while True:
                        comp_settings = CommonCompSettings(temperature=0.7, repeat_penalty=1.05, max_tokens=1024)
                        res = "“" + self.llm_manager.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
                        utterances = extract_utterances(res)
                        if len(utterances) > 0:
                            user_reply_content = utterances[0]
                            tmp.append(user_reply_content)
                            break

                simulated_user_reaction = "\n".join(tmp)

            self._log_mental_mechanism(
                self._deliberate_and_select_reply, MLevel.Low,
                f"Simulating Reply #{i}: AI says '{ai_reply_content[:70]}...'. "
                f"Predicts User will say '{simulated_user_reaction[:70]}...'"
            )
            # modifiers
            strong_emotions_summary = self._verbalize_emotional_state()
            needs_summary_line, cognitive_style_summary = self._verbalize_cognition_and_needs()

            mental_state_prompt_block = f"""
            **{self.config.companion_name}'s Current State of Mind:**
            - **Strong Emotions:** {strong_emotions_summary}
            - **Needs:** {needs_summary_line}
            - **Cognitive Style:** {cognitive_style_summary}
            """

            # --- Rate Simulation with a Single, Holistic Call ---
            prompt_rate_action = f"""
            You are a 'Critic' codelet within a LIDA-inspired cognitive architecture, evaluating a potential action for the AI, {self.config.companion_name}.

            **Current Internal State (The "Now"):**
            - **Emotions:** {strong_emotions_summary}
            - **Pressing Needs:** {needs_summary_line}
            - **Cognitive State:** {mental_state_prompt_block} # Reuse the block we built for generation
            - **Active Internal Goals (Intentions):** {intention_summary}

            **Simulated Future (The "What If"):**
            - **AI's Proposed Action:** "{ai_reply_content}"
            - **Predicted User Reaction:** "{simulated_user_reaction}"

            **Task: Holistically evaluate this "What If" scenario from the perspective of the "Now".**
            Provide scores from 0.0 to 1.0 based on the following criteria:

            1.  **`intent_fulfillment`**: Does the action and reaction achieve the 'Active Internal Goals'?
            2.  **`needs_fulfillment`**: Does the action address the 'Pressing Needs'? (e.g., a personal reply helps 'connection').
            3.  **`predicted_emotional_impact`**: Analyze the exchange. Will it likely lead to a positive emotional shift (more joy, less anxiety)? A score of +1.0 is a huge positive shift, -1.0 a huge negative one, 0.0 is neutral.
            4.  **`cognitive_load`**: Based on the *current* Cognitive State, how difficult is this action to perform? Consider the length/complexity of the reply. Is it a simple, easy response, or does it require a lot of thought and emotional control?

            Output *only* a JSON object conforming to the `ActionRating` schema.
            """

            # Call the LLM to get the comprehensive rating
            class ActionRating(BaseModel):
                """A comprehensive rating for a simulated action, considering multiple dimensions."""
                overall_reasoning: str = Field(description="A brief, holistic justification for the scores provided.")
                intent_fulfillment: float = Field(..., ge=0.0, le=1.0, description="How well this action and its predicted outcome fulfill the AI's active internal goals.")
                needs_fulfillment: float = Field(..., ge=0.0, le=1.0, description="How well this action fulfills the AI's most pressing needs (e.g., connection, relevance).")
                predicted_emotional_impact: float = Field(..., ge=-1.0, le=1.0, description="The predicted net change in emotional state. Positive is better (more joy, less anxiety), negative is worse.")
                cognitive_load: float = Field(..., ge=0.0, le=1.0, description="The cognitive effort required for this action. 0.0 is effortless, 1.0 is highly demanding (long, complex, emotionally difficult).")

            action_rating_result = self._call_llm(prompt_rate_action, output_schema=ActionRating, temperature=0.2)
            # --- Calculate the Final Score using Procedural Logic ---

            self._log_mental_mechanism(
                self._deliberate_and_select_reply, MLevel.Low,
                f"Critic rated Simulation #{i}: Intent={action_rating_result.intent_fulfillment:.2f}, "
                f"Needs={action_rating_result.needs_fulfillment:.2f}, "
                f"EmotionImpact={action_rating_result.predicted_emotional_impact:+.2f}. "
                f"Reason: {action_rating_result.overall_reasoning}"
            )

            # a) Get the scores from the LLM
            intent_fulfillment_score = action_rating_result.intent_fulfillment
            needs_fulfillment_score = action_rating_result.needs_fulfillment
            emotion_score = (action_rating_result.predicted_emotional_impact + 1.0) / 2.0  # Normalize [-1, 1] to [0, 1]
            cognitive_load = action_rating_result.cognitive_load

            # b) Calculate Cognitive Congruence Score (Procedural Logic)
            # This score reflects if the action is *wise* given the current resources.
            # It's high if the action is a good fit for the current state.
            # We start with a perfect score and apply penalties.
            cognitive_congruence_score = 1.0

            # Penalize taking on high-load actions when willpower is low.
            if current_cognition.willpower < 0.4 and cognitive_load > 0.6:
                # The penalty is greater the lower the willpower.
                penalty = (cognitive_load - 0.5) * (1.0 - current_cognition.willpower)
                cognitive_congruence_score -= penalty
                self._log_mental_mechanism(..., MLevel.Debug, f"Sim {sim_id}: Penalizing for high cognitive load ({cognitive_load:.2f}) with low willpower ({current_cognition.willpower:.2f}). Penalty: {penalty:.2f}")

            cognitive_congruence_score = max(0.0, cognitive_congruence_score)

            # c) Combine scores with clear weights
            final_sim_score = (intent_fulfillment_score * 0.40) \
                              + (needs_fulfillment_score * 0.30) \
                              + (emotion_score * 0.15) \
                              + (cognitive_congruence_score * 0.15)  # Give congruence more weight

            final_sim_score = max(0.0, min(1.0, final_sim_score))

            # d) To get the predicted future emotional state, we can run a quick appraisal on the impact score
            # This is less critical now but can be useful for logging. A simple approximation is fine.
            predicted_emotion_delta = EmotionalAxesModelDelta(valence=action_rating_result.predicted_emotional_impact)
            predicted_ai_emotion = current_emotions + predicted_emotion_delta

            simulation_results.append({
                "sim_id": sim_id,
                "action_type": ActionType.Reply,
                "ai_reply_content": ai_reply_content,
                "simulated_user_reaction": simulated_user_reaction,
                "predicted_ai_emotion": predicted_ai_emotion,
                "intent_fulfillment_score": intent_fulfillment_score,
                "needs_fulfillment_score": needs_fulfillment_score,
                "cognitive_congruence_score": cognitive_congruence_score,
                "final_score": final_sim_score
            })

            self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Low, f"--- Finished Reply Simulation {sim_id}: Score={final_sim_score:.3f}, Intent={intent_fulfillment_score:.2f}, Needs={needs_fulfillment_score:.2f}, Emo={emotion_score:.2f}, CogCongruence={cognitive_congruence_score:.2f} ---")
            self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Debug, f"Rating Reasoning: {action_rating_result.overall_reasoning}")

        # Add successful simulations to options
        action_options.extend(simulation_results)

        # --- Select Best Action (Unchanged) ---
        action_options.sort(key=lambda x: x["final_score"], reverse=True)
        best_action_details = action_options[0]

        self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Low, f"Selected Action: {best_action_details['action_type']} (Score: {best_action_details['final_score']:.3f})")
        if best_action_details['action_type'] == ActionType.Reply:
            self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Mid, f"Winning Content: {best_action_details['ai_reply_content'][:100]}...")
            self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Mid, f"Predicted User Reaction for chosen action: {best_action_details['simulated_user_reaction'][:100]}...")

        self.current_state.action_simulations = action_options
        return best_action_details

    @profile
    def _generate_expectations_for_action(self, action_knoxel: Action, action_details: Dict[str, Any]):
        """
        Generates Intention knoxels (internal=False) representing expectations
        based on the executed action and its predicted user reaction.
        """
        self._log_mental_mechanism(self._generate_expectations_for_action, MLevel.Debug, "Generate world / user expectations for selected action.")

        ai_reply = action_details["ai_reply_content"]
        predicted_reaction = action_details["simulated_user_reaction"]
        current_emotions = self.current_state.state_emotions if self.current_state else EmotionalAxesModel()

        class ExpectationSchema(BaseModel):
            content: str = Field(..., description=f"A brief description of what {self.config.companion_name} now expects {self.config.user_name} to say or do next.")
            affective_valence: float = Field(..., ge=-1.0, le=1.0, description=f"The valence (-1 to 1) {self.config.companion_name} *desires* or anticipates if this expectation is met.")
            urgency: float = Field(default=0.3, ge=0.0, le=1.0, description="Initial urgency (0-1) of this expectation.")
            # Proximal/Distal could be added here as a field later if needed

        # Dynamically create the list schema
        ExpectationListSchema = create_model(
            'ExpectationListSchema',
            expectations=(List[ExpectationSchema], Field(..., description=f"List of {self.config.expectation_generation_count} generated expectations."))
        )

        prompt = f"""
        Character: {self.config.companion_name}
        Current Emotional State: {current_emotions}
        {self.config.companion_name} just said: "{ai_reply}"
        Predicted {self.config.user_name}'s reaction: "{predicted_reaction}"

        Task: Based on what {self.config.companion_name} said and the predicted reaction, generate {self.config.expectation_generation_count} specific, plausible things {self.config.companion_name} might now *expect* {self.config.user_name} to say or do in the near future (next turn or two).
        For each expectation:
        - `content`: Describe the expected user action/utterance briefly.
        - `affective_valence`: Estimate the emotional valence (-1 to 1) {self.config.companion_name} would likely experience *if* this expectation is met. Consider her goals and personality (e.g., expecting agreement might be positive, expecting a question might be neutral or slightly positive if curious, expecting criticism negative).
        - `urgency`: Assign a low-to-moderate initial urgency (e.g., 0.1 to 0.5).

        Output *only* a JSON object matching the schema with a single key "expectations" containing the list.
        Example: {{ "expectations": [ {{ "content": "Expect {user_name} to agree with the cloud sheep idea", "affective_valence": 0.6, "urgency": 0.4 }}, ... ] }}
        """

        generated_expectations_result = self._call_llm(prompt, output_schema=ExpectationListSchema, temperature=0.6)

        created_expectation_ids = []
        if generated_expectations_result and isinstance(generated_expectations_result, ExpectationListSchema):
            self._log_mental_mechanism(self._generate_short_term_intentions, MLevel.Debug, f"Generated {len(generated_expectations_result.expectations)} potential expectations for Action {action_knoxel.id}.")

            # duplicates -> fulfill old ones and replace with new
            for exp_data in generated_expectations_result.expectations:
                emb_new = main_llm.get_embedding(exp_data.content)
                for existing_intention in self.all_intentions:
                    if not existing_intention.internal:
                        emb_existing = main_llm.get_embedding(existing_intention.content)
                        sim = cosine_pair(emb_new, emb_existing)
                        if sim < vector_same_threshold:
                            existing_intention.fulfilment = 1

            for exp_data in generated_expectations_result.expectations:
                # Calculate initial salience based on valence and urgency
                incentive_salience = max(0.1, min(0.9, (abs(exp_data.affective_valence) * 0.5 + exp_data.urgency * 0.5)))
                expectation_knoxel = Intention(
                    content=exp_data.content,
                    affective_valence=exp_data.affective_valence,
                    urgency=exp_data.urgency,
                    incentive_salience=incentive_salience,
                    internal=False,  # Mark as external expectation
                    fulfilment=0.0,
                    originating_action_id=action_knoxel.id,  # Link back to the action
                    tick_id=self._get_current_tick_id()  # Created in this tick
                )
                self.add_knoxel(expectation_knoxel)  # Adds to memory, generates embedding
                created_expectation_ids.append(expectation_knoxel.id)

                self._log_mental_mechanism(self._generate_short_term_intentions, MLevel.Debug,
                    f"Created Expectation {expectation_knoxel.id} for Action {action_knoxel.id}: '{exp_data.content}' (Desired V: {exp_data.affective_valence:.2f}, Urg: {exp_data.urgency:.2f})")
        else:
            logging.warning(f"Failed to generate expectations for Action {action_knoxel.id} via LLM.")

        return created_expectation_ids

    @profile
    def _execute_action(self) -> Action:
        """
        Creates the causal Action knoxel, generates associated expectations,
        adds them to memory, and returns the AI's content if it's an external action.
        """

        self._log_mental_mechanism(self._execute_action, MLevel.Mid, "Make selected action causal and send to world")

        action_details = self.current_state.selected_action_details
        action_type = action_details["action_type"]
        content = action_details["ai_reply_content"]
        self.simulated_reply = action_details["simulated_user_reaction"]

        # --- Create and Store Action Knoxel ---
        action_knoxel = Action(
            action_type=action_type,
            content=content,
            # generated_expectation_ids will be filled below
            tick_id=self._get_current_tick_id()
        )
        # Add the action knoxel *first* to get its ID
        self.add_knoxel(action_knoxel)
        self.current_state.selected_action_knoxel = action_knoxel

        # --- Generate Expectations (if applicable) ---
        generated_expectation_ids = []
        if action_type == ActionType.Reply or action_type == ActionType.ToolCallAndReply:
            generated_expectation_ids = self._generate_expectations_for_action(action_knoxel, action_details)
            if generated_expectation_ids:
                # Update the action knoxel with the IDs of expectations it generated
                action_knoxel.generated_expectation_ids = generated_expectation_ids
                logging.info(f"Linked {len(generated_expectation_ids)} expectations to Action {action_knoxel.id}.")

            # --- Create Causal Feature (Unchanged) ---
            predicted_emotion_state = action_details.get("predicted_ai_emotion", self.current_state.state_emotions)
            # Ensure predicted_emotion_state is the correct type
            #predicted_emotion_state = EmotionalAxesModel(**predicted_emotion_state)

            action_feature = Feature(
                content=f"{action_type} predicted emotional state",
                feature_type=FeatureType.Action,
                affective_valence=predicted_emotion_state.get_overall_valence(),
                interlocus=1 if action_type == ActionType.Reply else -1,
                causal=True, source=self.config.companion_name
            )
            self.add_knoxel(action_feature)

            reply_feature = Feature(
                source=self.config.companion_name,
                content=content,
                feature_type=FeatureType.Dialogue,
                affective_valence=predicted_emotion_state.get_overall_valence(),
                interlocus=1,
                causal=True,
            )
            self.add_knoxel(reply_feature)
        elif action_type == ActionType.Ignore:
            action_feature = Feature(
                content=f"{action_type}",
                feature_type=FeatureType.Action,
                affective_valence=-0.5,
                interlocus=-1,
                causal=True, source=self.config.companion_name
            )
            self.add_knoxel(action_feature)

            reply_feature = Feature(
                source=self.config.companion_name,
                content=f"*Ignores {self.config.user_name}*",
                feature_type=FeatureType.Dialogue,
                affective_valence=-0.5,
                interlocus=1,
                causal=True,
            )
            self.add_knoxel(reply_feature)

        self._log_mental_mechanism(self._execute_action, MLevel.Mid, f"Executed Action {action_knoxel.id}, Output: {content}")
        return action_knoxel

    @profile
    def _perform_learning_and_consolidation(self):
        """
        Summarizes the tick for episodic memory and refines narratives.
        Includes information about expectation outcomes for potential future learning.
        """
        if not self.current_state: return
        logging.info("Performing learning and consolidation...")

        # --- Gather Tick Context ---
        tick_id = self.current_state.tick_id
        stimulus = self.current_state.primary_stimulus
        subjective_experience = self.current_state.subjective_experience
        selected_action_details = self.current_state.selected_action_details
        executed_action = self.current_state.selected_action_knoxel
        start_emotions = self.states[-2].state_emotions if len(self.states) > 1 else EmotionalAxesModel()
        end_emotions = self.current_state.state_emotions

        # Find expectation outcome features from this tick
        expectation_outcomes = [
            f for f in self.all_features
            if f.tick_id == tick_id and f.feature_type == FeatureType.ExpectationOutcome
        ]
        outcome_summary = "; ".join([o.content for o in expectation_outcomes]) if expectation_outcomes else "None"

        if not stimulus and not subjective_experience and not executed_action and not expectation_outcomes:
            logging.info("No significant events in this tick to summarize or learn from.")
            return

        # --- Create Tick Summary Text ---
        tick_summary_parts = []
        if stimulus: tick_summary_parts.append(f"Stimulus: {stimulus.content}")
        tick_summary_parts.append(f"Initial Emotion V:{start_emotions.get_overall_valence():.2f}, A:{start_emotions.anxiety:.2f}")
        if subjective_experience: tick_summary_parts.append(f"Subjective Experience: {subjective_experience.content}")
        if selected_action_details:
            tick_summary_parts.append(f"Chosen Action: {selected_action_details['action_type']} (Score: {selected_action_details['final_score']:.2f})")
            tick_summary_parts.append(f"Action Content: {selected_action_details['ai_reply_content']}")
            if selected_action_details['action_type'] == ActionType.Reply:
                tick_summary_parts.append(f"Predicted User Reaction: {selected_action_details['simulated_user_reaction']}")
        # Add expectation outcome summary
        tick_summary_parts.append(f"Expectation Outcomes: {outcome_summary}")
        tick_summary_parts.append(f"Resulting Emotion V:{end_emotions.get_overall_valence():.2f}, A:{end_emotions.anxiety:.2f}")

        tick_summary_text = "\n".join(tick_summary_parts)

        # 1. Create Episodic Memory (Unchanged summary prompt, but input now includes expectation outcomes)
        class EpisodicSummarySchema(BaseModel):
            summary: str = Field(..., description="A concise, self-contained summary of the tick's key events and feelings from the character's perspective.")
            overall_valence: float = Field(..., ge=-1.0, le=1.0, description="The overall emotional valence (-1 to 1) experienced during this episode.")

        prompt_summary = f"""
        Character: {self.config.companion_name}
        Review the events of the last interaction cycle:
        {tick_summary_text}

        Task: Create a brief, self-contained summary of this episode for {self.config.companion_name}'s memory, written from her perspective. Also provide an overall emotional valence score (-1 to 1) for the episode based on the emotional change and outcome.

        Output *only* a JSON object matching the schema: {{ "summary": "string", "overall_valence": float }}
        """
        episodic_data = self._call_llm(prompt_summary, output_schema=EpisodicSummarySchema, temperature=0.5)

        if episodic_data and isinstance(episodic_data, EpisodicSummarySchema):
            memory = MemoryClusterKnoxel(
                content=episodic_data.content,
                affective_valence=episodic_data.overall_valence,
                tick_id=tick_id
            )
            self.add_knoxel(memory)
            logging.info(f"Created Episodic Memory (Valence: {memory.affective_valence:.2f}): {memory.content[:100]}...")

            # --- Q-Learning Hook Placeholder ---
            # Here, you could associate the actual outcome (episodic_data.overall_valence or the specific
            # emotional delta from expectation_outcomes) back to the 'executed_action'.
            # If executed_action exists and generated expectations:
            #   Find the expectation(s) evaluated in this tick (from expectation_outcomes).
            #   Find the emotional delta associated with *that specific outcome*.
            #   Use this delta as the reward signal R(s, a) for the state (captured before action) and action (executed_action).
            #   Update a Q-table or model: Q(s, a) = Q(s, a) + alpha * (R(s, a) + gamma * max(Q(s', a')) - Q(s, a))
            # This requires storing state representations (s) and implementing the Q-learning update logic.
            # For now, we just log that the information is available.
            if executed_action and expectation_outcomes:
                logging.debug(f"Tick {tick_id}: Action {executed_action.id} led to expectation outcomes with valence impact. (Data available for potential Q-learning update).")
            # --- End Q-Learning Hook ---

        else:
            logging.warning("Failed to create episodic summary via LLM.")

        # 2. Extract Declarative Facts (Placeholder - Unchanged)

        # 3. Refine Narratives (Unchanged refinement logic, but input summary is richer)
        logging.info("Attempting narrative refinement...")
        for definition in narrative_definitions:
            narrative_type = definition["type"]
            target_name = definition["target"]
            prompt_goal = definition["prompt"]
            current_narrative = self.get_narrative(narrative_type, target_name)
            current_narrative_content = current_narrative.content if current_narrative else "No previous narrative available."

            # Refinement condition (unchanged for now)
            should_refine = (abs(episodic_data.overall_valence) > 0.4 if episodic_data else False) or (executed_action is not None) or expectation_outcomes

            if not should_refine: continue

            logging.info(f"Refining Narrative: {narrative_type} for {target_name}")
            prompt_refine = f"""
            You are refining the psychological/behavioral narrative profile for '{target_name}'.
            Narrative Type Goal: "{prompt_goal}"

            Current Narrative for '{target_name}' ({narrative_type}):
            ---
            {current_narrative_content}
            ---

            Events of Recent Interaction Cycle (Tick {tick_id}):
            ---
            {tick_summary_text}
            ---

            Task: Based *only* on the new information from the recent interaction cycle, provide an *updated and improved* version of the narrative for '{target_name}' regarding '{narrative_type}'. Integrate insights, modify subtly, or keep original if no new relevant info. Output *only* the complete, updated narrative text.
            """
            updated_narrative_content = self._call_llm(prompt_refine, temperature=0.4, max_tokens=768)

            if updated_narrative_content and updated_narrative_content.strip() != current_narrative_content.strip():
                new_narrative = Narrative(
                    narrative_type=narrative_type,
                    target_name=target_name,
                    content=updated_narrative_content.strip(),
                    tick_id=tick_id
                )
                self.add_knoxel(new_narrative, generate_embedding=True)
                # logging.info(f"Updated Narrative: {narrative_type} for {target_name}.")
                # update_feature = Feature(
                #    content=f"Narrative Update: Refined '{narrative_type}' for '{target_name}' based on tick {tick_id}. New knoxel ID: {new_narrative.id}",
                #    feature_type=FeatureType.NarrativeUpdate,
                #    interlocus=0, causal=True
                # )
                # self.add_knoxel(update_feature, generate_embedding=False)
            # elif not updated_narrative_content: logging.warning(...) # Keep logs cleaner

    def _rebuild_specific_lists(self):
        """Helper to re-populate specific lists from all_knoxels after loading."""
        self.all_features = [k for k in self.all_knoxels.values() if isinstance(k, Feature)]
        self.all_stimuli = [k for k in self.all_knoxels.values() if isinstance(k, Stimulus)]
        self.all_intentions = [k for k in self.all_knoxels.values() if isinstance(k, Intention)]
        self.all_narratives = [k for k in self.all_knoxels.values() if isinstance(k, Narrative)]
        self.all_actions = [k for k in self.all_knoxels.values() if isinstance(k, Action)]
        self.all_episodic_memories = [k for k in self.all_knoxels.values() if isinstance(k, MemoryClusterKnoxel)]
        self.all_declarative_facts = [k for k in self.all_knoxels.values() if isinstance(k, DeclarativeFactKnoxel)]
        self.all_narratives.sort(key=lambda n: n.tick_id)

    @profile
    def _build_conscious_content_queue_assistant(
            self,
            cfg: GhostConfig,
            query_embedding: List[float],
            conscious_workspace_content: KnoxelList,
            max_total_tokens: int = 1500,  # Adjusted default
            recent_budget_ratio: float = 0.66,  # Give more to recent direct features
            topical_budget_ratio: float = 0.33,
            temporal_budget_ratio: float = 0
    ) -> List[Tuple[str, str]]:
        """
        Creates an optimized story prompt by balancing recent causal features,
        relevant topical cluster events, and relevant temporal summaries.
        """
        # --- Token Budget Allocation ---
        # Ensure ratios sum to 1, adjust if not (though here they do

        selected_knoxels_for_story: Dict[int, KnoxelBase] = {}  # Use dict to ensure unique knoxels by ID
        selected_knoxels_for_story[self.current_state.subjective_experience_tool.id] = self.current_state.subjective_experience_tool

        total_ratio = recent_budget_ratio + topical_budget_ratio + temporal_budget_ratio
        if not math.isclose(total_ratio, 1.0):
            #logging.warning(f"Budget ratios do not sum to 1.0 (sum: {total_ratio}). Normalizing.")
            recent_budget_ratio /= total_ratio
            topical_budget_ratio /= total_ratio
            temporal_budget_ratio /= total_ratio

        recent_token_budget = int(max_total_tokens * recent_budget_ratio)
        topical_token_budget = int(max_total_tokens * topical_budget_ratio)
        temporal_token_budget = int(max_total_tokens * temporal_budget_ratio)

        # --- 1. Select Most Recent Causal Features ---
        logging.debug(f"Recent causal feature budget: {recent_token_budget} tokens.")

        dialouge = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.feature_type == FeatureType.Dialogue) \
            .where(lambda x: x.tick_id != self.current_tick_id) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        current_tick = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.tick_id == self.current_tick_id) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        recent_causal_features = dialouge + current_tick
        current_recent_tokens = 0
        for feature in recent_causal_features[::-1]:
            tokens = get_token_count(feature)
            if current_recent_tokens + tokens <= recent_token_budget:
                if feature.id not in selected_knoxels_for_story:
                    selected_knoxels_for_story[feature.id] = feature
                    current_recent_tokens += tokens
            else:
                break
        logging.info(f"Selected {len(selected_knoxels_for_story)} recent causal features, using {current_recent_tokens} tokens.")

        # --- 2. Select Relevant Topical Cluster Events ---
        logging.debug(f"Topical cluster event budget: {topical_token_budget} tokens.")
        all_topical_clusters = [
            k for k in self.all_episodic_memories  # Assuming all_episodic_memories holds MemoryClusterKnoxels
            if k.cluster_type == ClusterType.Topical and k.embedding and k.included_event_ids
        ]

        if all_topical_clusters:
            # Rank topical clusters by relevance
            ranked_topical_clusters = sorted(
                all_topical_clusters,
                key=lambda c: cosine_distance(np.array(query_embedding), np.array(c.embedding))
            )  # Low distance = high relevance

            current_topical_tokens = 0
            for cluster in ranked_topical_clusters:
                if current_topical_tokens >= topical_token_budget:
                    break

                event_ids_in_cluster = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                events_to_add_from_cluster: List[KnoxelBase] = []
                tokens_for_this_cluster_events = 0

                # Get events from this cluster, preferring those not already selected
                # and sort them chronologically within the cluster
                cluster_event_knoxels = sorted(
                    [self.get_knoxel_by_id(eid) for eid in event_ids_in_cluster if self.get_knoxel_by_id(eid)],
                    key=lambda e: (e.tick_id, e.id)
                )

                for event in cluster_event_knoxels:
                    if event.id not in selected_knoxels_for_story:
                        tokens = get_token_count(event)
                        if current_topical_tokens + tokens_for_this_cluster_events + tokens <= topical_token_budget:
                            events_to_add_from_cluster.append(event)
                            tokens_for_this_cluster_events += tokens
                        else:  # Not enough budget for this specific event from cluster
                            break

                # Add the collected events from this cluster
                for event in events_to_add_from_cluster:
                    selected_knoxels_for_story[event.id] = event
                current_topical_tokens += tokens_for_this_cluster_events
            logging.info(f"Added events from topical clusters, using {current_topical_tokens} tokens.")

        # --- Combine, Sort, and Generate Story ---
        final_knoxels_for_story = sorted(
            selected_knoxels_for_story.values(),
            key=lambda k: k.timestamp_world_begin  # Chronological by creation
        )

        logger.info("Logging current story elements:")
        for k in final_knoxels_for_story:
            logger.info(k.get_story_element(self).replace("\n", "\\n"))

        story_list = KnoxelList(final_knoxels_for_story)
        final_story_str = story_list.get_story(cfg, max_tokens=max_total_tokens)  # Pass the original max_total_tokens

        self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Debug,
            f"Generated causal story with {len(final_knoxels_for_story)} knoxels. Final length approx {len(final_story_str) / (cfg.context_events_similarity_max_tokens / 512 * 3.5) :.0f} tokens.")  # Rough estimate

        turns = []
        for k in final_knoxels_for_story:
            if isinstance(k, Feature):
                if k.feature_type == FeatureType.Dialogue or k.feature_type == FeatureType.Thought:
                    if k.source == companion_name:
                        turns.append(("assistant", k.content))
                    elif k.source == user_name:
                        turns.append(("user", k.content))
                    else:
                        turns.append(("user", f"{k.source}: {k.content}"))
        return turns

    @profile
    def _build_conscious_content_queue(
            self,
            cfg: GhostConfig,
            query_embedding: List[float],
            conscious_workspace_content: KnoxelList,
            max_total_tokens: int = 1500,  # Adjusted default
            recent_budget_ratio: float = 0.50,  # Give more to recent direct features
            topical_budget_ratio: float = 0.25,
            temporal_budget_ratio: float = 0.25
    ) -> str:
        """
        Creates an optimized story prompt by balancing recent causal features,
        relevant topical cluster events, and relevant temporal summaries.
        """
        # --- Token Budget Allocation ---
        # Ensure ratios sum to 1, adjust if not (though here they do

        workspace_size = get_token_count(conscious_workspace_content.get_story(self))
        max_total_tokens -= workspace_size

        selected_knoxels_for_story: Dict[int, KnoxelBase] = {}  # Use dict to ensure unique knoxels by ID
        for f in conscious_workspace_content._list:
            selected_knoxels_for_story[f.id] = f

        total_ratio = recent_budget_ratio + topical_budget_ratio + temporal_budget_ratio
        if not math.isclose(total_ratio, 1.0):
            logging.warning(f"Budget ratios do not sum to 1.0 (sum: {total_ratio}). Normalizing.")
            recent_budget_ratio /= total_ratio
            topical_budget_ratio /= total_ratio
            temporal_budget_ratio /= total_ratio

        recent_token_budget = int(max_total_tokens * recent_budget_ratio)
        topical_token_budget = int(max_total_tokens * topical_budget_ratio)
        temporal_token_budget = int(max_total_tokens * temporal_budget_ratio)

        # --- 1. Select Most Recent Causal Features ---
        logging.debug(f"Recent causal feature budget: {recent_token_budget} tokens.")

        dialouge = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.feature_type == FeatureType.Dialogue) \
            .where(lambda x: x.tick_id != self.current_tick_id) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        current_tick = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.tick_id == self.current_tick_id) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        recent_causal_features = dialouge + current_tick
        current_recent_tokens = 0
        for feature in recent_causal_features[::-1]:
            tokens = get_token_count(feature)
            if current_recent_tokens + tokens <= recent_token_budget:
                if feature.id not in selected_knoxels_for_story:
                    selected_knoxels_for_story[feature.id] = feature
                    current_recent_tokens += tokens
            else:
                break
        logging.info(f"Selected {len(selected_knoxels_for_story)} recent causal features, using {current_recent_tokens} tokens.")

        # --- 2. Select Relevant Topical Cluster Events ---
        logging.debug(f"Topical cluster event budget: {topical_token_budget} tokens.")
        all_topical_clusters = [
            k for k in self.all_episodic_memories  # Assuming all_episodic_memories holds MemoryClusterKnoxels
            if k.cluster_type == ClusterType.Topical and k.embedding and k.included_event_ids
        ]

        if all_topical_clusters:
            # Rank topical clusters by relevance
            ranked_topical_clusters = sorted(
                all_topical_clusters,
                key=lambda c: cosine_distance(np.array(query_embedding), np.array(c.embedding))
            )  # Low distance = high relevance

            current_topical_tokens = 0
            for cluster in ranked_topical_clusters:
                if current_topical_tokens >= topical_token_budget:
                    break

                event_ids_in_cluster = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                events_to_add_from_cluster: List[KnoxelBase] = []
                tokens_for_this_cluster_events = 0

                # Get events from this cluster, preferring those not already selected
                # and sort them chronologically within the cluster
                cluster_event_knoxels = sorted(
                    [self.get_knoxel_by_id(eid) for eid in event_ids_in_cluster if self.get_knoxel_by_id(eid)],
                    key=lambda e: (e.tick_id, e.id)
                )

                for event in cluster_event_knoxels:
                    if event.id not in selected_knoxels_for_story:
                        tokens = get_token_count(event)
                        if current_topical_tokens + tokens_for_this_cluster_events + tokens <= topical_token_budget:
                            events_to_add_from_cluster.append(event)
                            tokens_for_this_cluster_events += tokens
                        else:  # Not enough budget for this specific event from cluster
                            break

                # Add the collected events from this cluster
                for event in events_to_add_from_cluster:
                    selected_knoxels_for_story[event.id] = event
                current_topical_tokens += tokens_for_this_cluster_events
            logging.info(f"Added events from topical clusters, using {current_topical_tokens} tokens.")

        # --- 3. Select Relevant Temporal Summaries ---
        logging.debug(f"Temporal summary budget: {temporal_token_budget} tokens.")
        all_temporal_summaries = [
            k for k in self.all_episodic_memories
            if k.cluster_type == ClusterType.Temporal and k.embedding and k.content
        ]

        if all_temporal_summaries:
            # Rank temporal summaries
            ranked_temporal_summaries = sorted(
                all_temporal_summaries,
                key=lambda s: cosine_distance(np.array(query_embedding), np.array(s.embedding)) if s.embedding else 1.0
            )

            current_temporal_tokens = 0
            for summary in ranked_temporal_summaries:
                if summary.id not in selected_knoxels_for_story:  # Don't add if somehow already there
                    tokens = get_token_count(summary)
                    if current_temporal_tokens + tokens <= temporal_token_budget:
                        selected_knoxels_for_story[summary.id] = summary
                        current_temporal_tokens += tokens
                    else:
                        break
            logging.info(f"Selected temporal summaries, using {current_temporal_tokens} tokens.")

        # --- Combine, Sort, and Generate Story ---
        final_knoxels_for_story = sorted(
            selected_knoxels_for_story.values(),
            key=lambda k: k.timestamp_world_begin  # Chronological by creation
        )

        logger.info("Logging current story elements:")
        for k in final_knoxels_for_story:
            logger.info(k.get_story_element(self).replace("\n", "\\n"))

        story_list = KnoxelList(final_knoxels_for_story)
        final_story_str = story_list.get_story(cfg, max_tokens=max_total_tokens)  # Pass the original max_total_tokens

        self._log_mental_mechanism(self._deliberate_and_select_reply, MLevel.Debug,
            f"Generated causal story with {len(final_knoxels_for_story)} knoxels. Final length approx {len(final_story_str) / (cfg.context_events_similarity_max_tokens / 512 * 3.5) :.0f} tokens.")  # Rough estimate
        return final_story_str

class GhostSqlite(Ghost):
    def _get_knoxel_subclasses(self) -> Dict[str, Type[KnoxelBase]]:
        """Finds all subclasses of KnoxelBase defined in the current scope."""
        # This might need adjustment based on where KnoxelBase and its subclasses are defined.
        # Assuming they are in the same module or imported.
        subclasses = {}
        # Check classes defined in the same module as Ghost
        for name, obj in inspect.getmembers(inspect.getmodule(inspect.currentframe())):
            if inspect.isclass(obj) and issubclass(obj, KnoxelBase) and obj is not KnoxelBase:
                # Use class name as key, map to the actual class type
                subclasses[obj.__name__] = obj
        # Add KnoxelBase itself if needed for some reason (usually not for tables)
        # subclasses[KnoxelBase.__name__] = KnoxelBase
        if not subclasses:
            logging.warning("Could not dynamically find any KnoxelBase subclasses. DB operations might fail.")
            # Fallback to known types if inspection fails
            return {
                cls.__name__: cls for cls in
                [Stimulus, Intention, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, Narrative, Feature]
            }

        return subclasses

    def _pydantic_type_to_sqlite_type(self, field_info: Any) -> str:
        """Maps a Pydantic field's type annotation to an SQLite type string."""
        field_type = field_info.annotation
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional types
        if origin is Union and type(None) in args:
            # Get the non-None type
            field_type = next(t for t in args if t is not type(None))
            origin = get_origin(field_type)  # Re-evaluate origin for List[Optional[...]] etc.
            args = get_args(field_type)
            # Nullability is handled by default in SQLite columns unless PRIMARY KEY or NOT NULL specified

        # Basic types
        if field_type is int: return "INTEGER"
        if field_type is float: return "REAL"
        if field_type is str: return "TEXT"
        if field_type is bool: return "INTEGER"  # Store as 0 or 1
        if field_type is datetime: return "TEXT"  # Store as ISO format string

        # Enums
        if inspect.isclass(field_type) and issubclass(field_type, (StrEnum, IntEnum)):
            return "TEXT" if issubclass(field_type, StrEnum) else "INTEGER"

        # Lists
        if origin is list or field_type is List:
            if args:
                list_item_type = args[0]
                # Special case for embeddings
                if list_item_type is float:
                    return "BLOB"  # Store List[float] embeddings as BLOB
            # Store other lists (like list of int IDs) as JSON strings
            return "TEXT"

        # Dictionaries
        if origin is dict or field_type is Dict:
            return "TEXT"  # Store dicts as JSON strings

        # References to other Knoxels (by ID) - These shouldn't be direct fields in Knoxel tables itself usually
        # but might appear in GhostState. We store the ID.
        if inspect.isclass(field_type) and issubclass(field_type, KnoxelBase):
            return "INTEGER"

        # Default / Fallback
        logging.warning(f"Unknown Pydantic type for SQLite mapping: {field_type}. Using TEXT as fallback.")
        return "TEXT"

    def _serialize_value_for_db(self, value: Any, field_type: Type) -> Any:
        """Converts Python value to DB storable format based on Pydantic type."""
        if value is None:
            return None

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional correctly
        if origin is Union and type(None) in args:
            if value is None: return None
            field_type = next(t for t in args if t is not type(None))
            origin = get_origin(field_type)  # Re-evaluate origin
            args = get_args(field_type)

        # Basic types that are already compatible or need simple conversion
        if isinstance(value, (int, float, str)): return value
        if isinstance(value, bool): return 1 if value else 0
        if isinstance(value, datetime): return value.isoformat()

        # Enums
        if isinstance(value, (StrEnum, IntEnum)): return value.value

        # Embeddings (List[float])
        if (origin is list or isinstance(value, list)) and args and args[0] is float:
            # Check if it looks like an embedding before serializing
            if all(isinstance(x, float) for x in value):
                return serialize_embedding(value)
            else:
                logging.warning(
                    f"Expected List[float] for embedding, got list with other types: {type(value[0]) if value else 'empty'}. Serializing as JSON.")
                # Fallback to JSON for non-float lists
                return json.dumps(value)

        # Other Lists or Dicts
        if isinstance(value, (list, dict)):
            return json.dumps(value)

        # Knoxel References (store ID) - relevant for GhostState, not Knoxel tables usually
        if isinstance(value, KnoxelBase): return value.id

        logging.warning(f"Could not serialize type {type(value)} for DB. Returning str(value).")
        return str(value)  # Fallback

    def _deserialize_value_from_db(self, db_value: Any, target_type: Type) -> Any:
        """Converts DB value back to Python type based on Pydantic annotation."""
        if db_value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)
        is_optional = False

        if origin is Union and type(None) in args:
            is_optional = True
            if db_value is None: return None
            target_type = next(t for t in args if t is not type(None))  # Get the actual type
            origin = get_origin(target_type)  # Re-evaluate origin
            args = get_args(target_type)

        # Basic types
        if target_type is int: return int(db_value) if db_value is not None else (None if is_optional else 0)
        if target_type is float: return float(db_value) if db_value is not None else (None if is_optional else 0.0)
        if target_type is str: return str(db_value) if db_value is not None else (None if is_optional else "")
        if target_type is bool: return bool(db_value) if db_value is not None else (None if is_optional else False)
        if target_type is datetime:
            try:
                return datetime.fromisoformat(db_value) if isinstance(db_value, str) else (
                    None if is_optional else datetime.now())  # Provide default or handle error?
            except (ValueError, TypeError):
                return None if is_optional else datetime.now()  # Fallback

        # Enums
        if inspect.isclass(target_type) and issubclass(target_type, Enum):
            try:
                return target_type(db_value)
            except ValueError:
                return None if is_optional else list(target_type)[0]  # Fallback to first enum member or None

        # Embeddings (List[float])
        if (origin is list or target_type is List) and args and args[0] is float:
            if isinstance(db_value, bytes):
                return deserialize_embedding(db_value)
            else:
                logging.warning(f"Expected bytes for embedding BLOB, got {type(db_value)}. Returning empty list.")
                return []

        # Other Lists or Dicts (from JSON text)
        if (origin is list or target_type is List or origin is dict or target_type is Dict) and isinstance(db_value,
                                                                                                           str):
            try:
                return json.loads(db_value)
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON for {target_type}: {db_value[:100]}...")
                return None if is_optional else ([] if origin is list else {})  # Fallback

        # Knoxel references (return ID - reconstruction happens later)
        if inspect.isclass(target_type) and issubclass(target_type, KnoxelBase):
            return int(db_value) if db_value is not None else None

        logging.warning(f"Could not deserialize DB value '{db_value}' to type {target_type}. Returning raw value.")
        return db_value  # Fallback

    # --- Dynamic SQLite Persistence ---

    def save_state_sqlite(self, filename: str):
        """Saves the complete Ghost state to an SQLite database dynamically."""
        if not commit:
            logging.info(f"Rolling back...")
            return

        logging.info(f"Dynamically saving state to SQLite database: {filename}...")

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except:
            pass

        try:
            bak_path = filename + ".bak"
            if os.path.isfile(bak_path):
                os.remove(bak_path)

            if os.path.isfile(filename):
                shutil.copyfile(filename, bak_path)
        except:
            pass

        conn = None
        if True:
            conn = sqlite3.connect(filename)
            cursor = conn.cursor()

            # --- Define Tables Dynamically ---
            knoxel_subclasses = self._get_knoxel_subclasses()
            knoxel_schemas = {}  # Store {Type: (table_name, columns)}
            ghost_state_columns = {}  # Store {field_name: sqlite_type}

            # 1. Knoxel Tables
            for knoxel_cls in knoxel_subclasses.values():
                table_name = knoxel_cls.__name__.lower()
                columns = {}
                for field_name, field_info in knoxel_cls.model_fields.items():
                    sqlite_type = self._pydantic_type_to_sqlite_type(field_info)
                    columns[field_name] = sqlite_type
                knoxel_schemas[knoxel_cls] = (table_name, columns)

            # 2. GhostState Table (Flatten nested models)
            state_table_name = "ghost_states"
            base_state_columns = {}
            nested_state_models = {
                'state_emotions': EmotionalAxesModel,
                'state_needs': NeedsAxesModel,
                'state_cognition': CognitionAxesModel
            }
            for field_name, field_info in GhostState.model_fields.items():
                if field_name in nested_state_models:  # Skip nested model fields themselves
                    continue
                if field_name in ['attention_candidates', 'attention_focus', 'conscious_workspace',
                                  'action_simulations', "coalitions_balanced", "coalitions_hard",
                                  'selected_action_details']:  # Skip runtime/complex fields not directly stored
                    continue
                # Handle reference fields (store ID) and list of IDs
                if field_name == 'primary_stimulus' or \
                        field_name == 'subjective_experience' or \
                        field_name == "subjective_experience_tool" or \
                        field_name == 'selected_action_knoxel':
                    base_state_columns[f"{field_name}_id"] = "INTEGER"  # Store ID
                elif field_name == 'conscious_workspace_ids':  # Store list of IDs as JSON TEXT
                    base_state_columns[field_name] = "TEXT"
                else:
                    base_state_columns[field_name] = self._pydantic_type_to_sqlite_type(field_info)
            # Add flattened columns from nested models
            for prefix, model_cls in nested_state_models.items():
                for field_name, field_info in model_cls.model_fields.items():
                    col_name = f"{prefix.split('_')[-1]}_{field_name}"  # e.g., emotion_valence
                    base_state_columns[col_name] = self._pydantic_type_to_sqlite_type(field_info)
            ghost_state_columns = base_state_columns

            # --- Drop Existing Tables ---
            all_tables_to_drop = ['metadata', 'config'] + [name for name, _ in knoxel_schemas.values()] + [
                state_table_name]
            for table_name in all_tables_to_drop:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            logging.debug("Dropped existing tables.")

            # --- Create New Tables ---
            # Metadata table
            cursor.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);")
            # Config table
            cursor.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);")
            # Knoxel tables
            for _, (table_name, columns) in knoxel_schemas.items():
                cols_sql = ",\n    ".join(
                    [f"{name} {ctype}" + (" PRIMARY KEY" if name == "id" else "") for name, ctype in columns.items()])
                create_sql = f"CREATE TABLE {table_name} (\n    {cols_sql}\n);"
                logging.debug(f"Creating table {table_name}:\n{create_sql}")
                cursor.execute(create_sql)
            # Ghost state table
            cols_sql = ",\n    ".join(
                [f"{name} {ctype}" + (" PRIMARY KEY" if name == "tick_id" else "") for name, ctype in
                 ghost_state_columns.items()])
            create_state_sql = f"CREATE TABLE {state_table_name} (\n    {cols_sql}\n);"
            logging.debug(f"Creating table {state_table_name}:\n{create_state_sql}")
            cursor.execute(create_state_sql)

            logging.info("Created tables dynamically based on Pydantic models.")

            # --- Insert Metadata & Config (same as before) ---
            cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                           ('current_tick_id', str(self.current_tick_id)))
            cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                           ('current_knoxel_id', str(self.current_knoxel_id)))
            config_data = self.config.model_dump()
            for key, value in config_data.items():
                value_str = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
                cursor.execute("INSERT INTO config (key, value) VALUES (?, ?)", (key, value_str))

            # --- Insert Knoxels Dynamically ---
            for knoxel in self.all_knoxels.values():
                knoxel_type = type(knoxel)
                if knoxel_type not in knoxel_schemas:
                    logging.warning(f"Skipping knoxel {knoxel.id} - No schema found for type {knoxel_type.__name__}.")
                    continue

                table_name, columns_schema = knoxel_schemas[knoxel_type]
                data_dict = knoxel.model_dump()

                # Prepare data and map field names to column names (they should match)
                insert_data = {}
                values_list = []
                cols_list = []

                for field_name, field_info in knoxel_type.model_fields.items():
                    if field_name in columns_schema:  # Ensure the field exists in our table schema
                        raw_value = data_dict.get(field_name)
                        serialized_value = self._serialize_value_for_db(raw_value, field_info.annotation)
                        insert_data[field_name] = serialized_value
                        cols_list.append(field_name)
                        values_list.append(serialized_value)

                if not cols_list: continue  # Skip if no columns to insert

                cols_str = ', '.join(cols_list)
                placeholders = ', '.join(['?'] * len(cols_list))
                sql = f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders});"

                try:
                    cursor.execute(sql, tuple(values_list))
                except sqlite3.Error as e:
                    logging.error(f"SQLite error inserting into {table_name} (Knoxel {knoxel.id}): {e}")
                    logging.error(f"SQL: {sql}")
                    logging.error(f"Values: {values_list}")

            # --- Insert Ghost States Dynamically ---
            for state in self.states:
                state_insert_data = {}
                values_list = []
                cols_list = []

                # Handle base GhostState fields
                for field_name, field_info in GhostState.model_fields.items():
                    col_name = field_name
                    value = getattr(state, field_name, None)

                    # Handle special cases (references, lists, skipped fields)
                    if field_name in ['attention_candidates', 'attention_focus', 'conscious_workspace',
                                      'action_simulations', 'selected_action_details', "coalitions_hard", "coalitions_balanced"]:
                        if field_name == 'conscious_workspace':  # Handle workspace IDs list
                            col_name = 'conscious_workspace_ids'
                            value = [k.id for k in value.to_list()] if value else []
                        else:
                            continue  # Skip other complex/runtime fields
                    elif field_name in ['primary_stimulus', 'subjective_experience', "subjective_experience_tool", 'selected_action_knoxel']:
                        col_name = f"{field_name}_id"
                        value = value.id if value else None
                    elif field_name in nested_state_models:  # Skip the direct nested model fields
                        continue

                    # Ensure column exists in our dynamic schema before adding
                    if col_name in ghost_state_columns:
                        serialized_value = self._serialize_value_for_db(value, field_info.annotation)
                        state_insert_data[col_name] = serialized_value
                        cols_list.append(col_name)
                        values_list.append(serialized_value)

                # Handle flattened nested models
                for prefix, model_cls in nested_state_models.items():
                    model_instance = getattr(state, prefix, None)
                    if model_instance:
                        for field_name, field_info in model_cls.model_fields.items():
                            col_name = f"{prefix.split('_')[-1]}_{field_name}"  # e.g., emotion_valence
                            if col_name in ghost_state_columns:  # Check if column exists
                                raw_value = getattr(model_instance, field_name, None)
                                serialized_value = self._serialize_value_for_db(raw_value, field_info.annotation)
                                state_insert_data[col_name] = serialized_value
                                cols_list.append(col_name)
                                values_list.append(serialized_value)

                if not cols_list: continue  # Skip if no columns to insert

                cols_str = ', '.join(cols_list)
                placeholders = ', '.join(['?'] * len(cols_list))
                sql = f"INSERT INTO {state_table_name} ({cols_str}) VALUES ({placeholders});"

                try:
                    cursor.execute(sql, tuple(values_list))
                except sqlite3.Error as e:
                    logging.error(f"SQLite error inserting into {state_table_name} (Tick {state.tick_id}): {e}")
                    logging.error(f"SQL: {sql}")
                    logging.error(f"Values: {values_list}")

            # --- Commit ---
            conn.commit()
            logging.info(f"State dynamically saved to SQLite database: {filename}")

    def load_state_sqlite(self, filename: str):
        """Loads the complete Ghost state from an SQLite database dynamically."""
        logging.info(f"Dynamically loading state from SQLite database: {filename}...")
        conn = None
        try:
            # Check if file exists before connecting
            if not os.path.exists(filename):
                logging.warning(f"SQLite save file {filename} not found. Starting with fresh state.")
                self._reset_internal_state()  # Clear any existing state
                self._initialize_narratives()  # Ensure default narratives exist
                return

            conn = sqlite3.connect(filename)
            conn.row_factory = sqlite3.Row  # Access columns by name
            cursor = conn.cursor()

            # --- Reset internal state before loading ---
            self._reset_internal_state()

            # --- Load Metadata & Config ---
            try:
                cursor.execute("SELECT value FROM metadata WHERE key = 'current_tick_id';")
                self.current_tick_id = int(cursor.fetchone()['value'])
                cursor.execute("SELECT value FROM metadata WHERE key = 'current_knoxel_id';")
                self.current_knoxel_id = int(cursor.fetchone()['value'])
            except (TypeError, sqlite3.Error, ValueError) as e:  # Handle missing table/keys or non-integer values
                logging.warning(f"Could not load metadata (tick/knoxel IDs): {e}. Using defaults.")
                self.current_tick_id = 0
                self.current_knoxel_id = 0

            try:
                cursor.execute("SELECT key, value FROM config;")
                config_data = {}
                loaded_config = cursor.fetchall()
                config_fields = GhostConfig.model_fields
                for row in loaded_config:
                    key, value_str = row['key'], row['value']
                    if key in config_fields:
                        target_type = config_fields[key].annotation
                        # Attempt to deserialize complex types (e.g., lists from JSON)
                        try:
                            # Basic check if it looks like JSON list/dict
                            if (isinstance(target_type, type) and issubclass(target_type, (List, Dict))) or \
                                    (get_origin(target_type) in [list, dict, List, Dict]):
                                config_data[key] = json.loads(value_str)
                            else:  # Otherwise, try casting basic types or keep as string
                                if target_type is int:
                                    config_data[key] = int(value_str)
                                elif target_type is float:
                                    config_data[key] = float(value_str)
                                elif target_type is bool:
                                    config_data[key] = value_str.lower() in ['true', '1']
                                else:
                                    config_data[key] = value_str  # Keep as string if other type
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            logging.warning(
                                f"Error parsing config value for '{key}': {e}. Value: '{value_str}'. Using raw string.")
                            config_data[key] = value_str
                    else:
                        logging.warning(f"Ignoring unknown config key from DB: {key}")

                self.config = GhostConfig(**config_data)
                logging.info(f"Loaded config: {self.config.model_dump_json(indent=1)}")
            except (sqlite3.Error, ValidationError) as e:
                logging.error(f"Could not load or validate config: {e}. Using default config.")
                self.config = GhostConfig()  # Use defaults

            # --- Load Knoxels Dynamically ---
            knoxel_subclasses = self._get_knoxel_subclasses()
            loaded_knoxels = {}  # Temp dict to hold loaded knoxels {id: instance}

            for knoxel_cls in knoxel_subclasses.values():
                table_name = knoxel_cls.__name__.lower()
                logging.debug(f"Attempting to load from table: {table_name}")
                try:
                    cursor.execute(f"SELECT * FROM {table_name};")
                    rows = cursor.fetchall()
                    if not rows:
                        logging.debug(f"Table {table_name} is empty or does not exist.")
                        continue

                    column_names = [desc[0] for desc in cursor.description]
                    model_fields = knoxel_cls.model_fields

                    for row in rows:
                        knoxel_data = {}
                        row_dict = dict(row)  # Convert sqlite3.Row to dict
                        knoxel_id = row_dict.get('id', -1)  # Get ID for logging

                        for field_name, field_info in model_fields.items():
                            if field_name in column_names:
                                db_value = row_dict[field_name]
                                try:
                                    py_value = self._deserialize_value_from_db(db_value, field_info.annotation)
                                    knoxel_data[field_name] = py_value
                                except Exception as e:
                                    logging.error(
                                        f"Error deserializing field '{field_name}' for {table_name} ID {knoxel_id}: {e}. DB Value: {db_value}. Skipping field.")
                            else:
                                # Field exists in model but not in DB table (schema evolution)
                                # Pydantic will use default value if available, otherwise raise error if required
                                logging.debug(
                                    f"Field '{field_name}' not found in DB table '{table_name}' for knoxel ID {knoxel_id}. Using model default if available.")

                        # Add fields present in DB but not in model? Pydantic ignores extra by default.

                        if 'id' not in knoxel_data:  # Should always be present if table exists
                            logging.error(f"Missing 'id' column data for row in {table_name}. Skipping row: {row_dict}")
                            continue

                        try:
                            knoxel_instance = knoxel_cls(**knoxel_data)
                            loaded_knoxels[knoxel_instance.id] = knoxel_instance
                        except ValidationError as e:
                            logging.error(f"Pydantic validation failed for {knoxel_cls.__name__} ID {knoxel_id}: {e}")
                            logging.error(f"Data causing error: {knoxel_data}")
                        except Exception as e:
                            logging.error(f"Unexpected error instantiating {knoxel_cls.__name__} ID {knoxel_id}: {e}")
                            logging.error(f"Data causing error: {knoxel_data}")

                except sqlite3.OperationalError as e:
                    logging.warning(f"Could not read from table {table_name} (might not exist in this DB): {e}")
                except Exception as e:
                    logging.error(f"Unexpected error loading from table {table_name}: {e}", exc_info=True)

            self.all_knoxels = loaded_knoxels
            logging.info(f"Loaded {len(self.all_knoxels)} knoxels from database.")

            # Rebuild specific lists from the loaded knoxels
            self._rebuild_specific_lists()

            # --- Load Ghost States Dynamically ---
            state_table_name = "ghost_states"
            self.states = []
            nested_state_models = {
                'state_emotions': EmotionalAxesModel,
                'state_needs': NeedsAxesModel,
                'state_cognition': CognitionAxesModel
            }

            try:
                cursor.execute(f"SELECT * FROM {state_table_name} ORDER BY tick_id ASC;")
                rows = cursor.fetchall()
                if not rows:
                    logging.warning(f"Ghost states table '{state_table_name}' is empty or does not exist.")

                column_names = [desc[0] for desc in cursor.description]
                base_state_fields = GhostState.model_fields

                for row in rows:
                    state_data = {}
                    nested_models_data = {prefix: {} for prefix in nested_state_models}
                    row_dict = dict(row)
                    tick_id = row_dict.get('tick_id', -1)

                    # Process columns from DB row
                    for col_name in column_names:
                        db_value = row_dict[col_name]
                        processed = False

                        # Check if it's a flattened nested model field
                        for prefix, model_cls in nested_state_models.items():
                            short_prefix = prefix.split('_')[-1]  # e.g., emotion
                            if col_name.startswith(f"{short_prefix}_"):
                                field_name = col_name[len(short_prefix) + 1:]
                                if field_name in model_cls.model_fields:
                                    target_type = model_cls.model_fields[field_name].annotation
                                    try:
                                        py_value = self._deserialize_value_from_db(db_value, target_type)
                                        nested_models_data[prefix][field_name] = py_value
                                    except Exception as e:
                                        logging.error(
                                            f"Error deserializing nested field '{col_name}' for state tick {tick_id}: {e}. DB Value: {db_value}. Skipping field.")
                                    processed = True
                                    break  # Move to next column once matched

                        if processed: continue

                        # Check if it's a base GhostState field (or reference ID)
                        field_name = col_name
                        is_reference_id = False
                        if col_name.endswith("_id") and col_name[:-3] in ['primary_stimulus', 'subjective_experience', "subjective_experience_tool",
                                                                          'selected_action_knoxel', "rating"]:
                            field_name = col_name[:-3]  # e.g., primary_stimulus
                            is_reference_id = True

                        if field_name in base_state_fields:
                            target_type = base_state_fields[field_name].annotation
                            try:
                                if is_reference_id:
                                    # Value is the ID, reconstruction happens below
                                    state_data[field_name + "_id"] = int(db_value) if db_value is not None else None
                                elif field_name == 'conscious_workspace':  # Handle the list of IDs
                                    id_list = json.loads(db_value) if isinstance(db_value, str) else []
                                    state_data['conscious_workspace_ids'] = [int(i) for i in id_list if i is not None]
                                elif field_name not in nested_state_models and \
                                        field_name not in ['attention_candidates', 'attention_focus',
                                                           'action_simulations', "coalitions_hard", "coalitions_balanced",
                                                           'selected_action_details']:  # Avoid loading complex fields directly
                                    py_value = self._deserialize_value_from_db(db_value, target_type)
                                    state_data[field_name] = py_value
                            except Exception as e:
                                logging.error(
                                    f"Error deserializing base state field '{col_name}' for state tick {tick_id}: {e}. DB Value: {db_value}. Skipping field.")
                        elif col_name != 'conscious_workspace_ids':  # Allow conscious_workspace_ids even if not directly in model
                            logging.debug(
                                f"Column '{col_name}' from DB table '{state_table_name}' not found in GhostState model or handled cases. Ignoring.")

                    # Instantiate nested models
                    for prefix, model_cls in nested_state_models.items():
                        try:
                            instance = model_cls(**nested_models_data[prefix])
                            state_data[prefix] = instance
                        except ValidationError as e:
                            logging.error(
                                f"Validation failed for {model_cls.__name__} in state tick {tick_id}: {e}. Data: {nested_models_data[prefix]}")
                            state_data[prefix] = model_cls()  # Use default instance
                        except Exception as e:
                            logging.error(
                                f"Error instantiating {model_cls.__name__} in state tick {tick_id}: {e}. Data: {nested_models_data[prefix]}")
                            state_data[prefix] = model_cls()

                    # Reconstruct references and KnoxelLists
                    try:
                        state_data['primary_stimulus'] = self.get_knoxel_by_id(
                            state_data.pop('primary_stimulus_id', None))
                        state_data['subjective_experience'] = self.get_knoxel_by_id(
                            state_data.pop('subjective_experience_id', None))
                        state_data['subjective_experience_tool'] = self.get_knoxel_by_id(
                            state_data.pop('subjective_experience_tool_id', None))
                        state_data['selected_action_knoxel'] = self.get_knoxel_by_id(
                            state_data.pop('selected_action_knoxel_id', None))

                        workspace_ids = state_data.pop('conscious_workspace_ids', [])
                        workspace_knoxels = [k for kid in workspace_ids if (k := self.get_knoxel_by_id(kid))]
                        state_data['conscious_workspace'] = KnoxelList(workspace_knoxels)

                        # Add defaults for runtime fields not saved
                        state_data.setdefault('attention_candidates', KnoxelList())
                        state_data.setdefault('attention_focus', KnoxelList())
                        state_data.setdefault('action_simulations', [])
                        state_data.setdefault('selected_action_details', None)
                        state_data.setdefault('coalitions_hard', {})
                        state_data.setdefault('coalitions_balanced', {})

                        # Instantiate GhostState
                        ghost_state_instance = GhostState(**state_data)
                        self.states.append(ghost_state_instance)

                    except ValidationError as e:
                        logging.error(f"Pydantic validation failed for GhostState tick {tick_id}: {e}")
                        logging.error(f"Data causing error: {state_data}")
                    except Exception as e:
                        logging.error(f"Unexpected error reconstructing GhostState tick {tick_id}: {e}")
                        logging.error(f"Data causing error: {state_data}")


            except sqlite3.OperationalError as e:
                logging.warning(f"Could not read from table {state_table_name}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error loading from table {state_table_name}: {e}", exc_info=True)

            logging.info(f"Loaded {len(self.states)} Ghost states from database.")
            self.states.sort(key=lambda s: s.tick_id)  # Ensure states are ordered
            self.current_state = self.states[-1] if self.states else None

            # Ensure narratives are loaded or initialized
            if not self.all_narratives:
                self._initialize_narratives()
            else:
                # Ensure narratives are sorted by tick_id after loading
                self.all_narratives.sort(key=lambda n: n.tick_id)


        except sqlite3.Error as e:
            logging.error(f"SQLite Error during load: {e}", exc_info=True)
            self._reset_internal_state()  # Reset on error
            self.__init__(self.config)  # Re-initialize with default config
        except Exception as e:
            logging.error(f"Unexpected Error during load: {e}", exc_info=True)
            self._reset_internal_state()
            self.__init__(self.config)
        finally:
            if conn: conn.close()

    # --- Helper to reset internal state ---
    def _reset_internal_state(self):
        """Clears all knoxels, states, and resets IDs."""
        self.current_tick_id = 0
        self.current_knoxel_id = 0
        self.all_knoxels = {}
        self.all_features = []
        self.all_stimuli = []
        self.all_intentions = []
        self.all_narratives = []
        self.all_actions = []
        self.all_episodic_memories = []
        self.all_declarative_facts = []
        self.states = []
        self.current_state = None
        self.simulated_reply = None
        logging.info("Internal state reset.")

    def _rebuild_specific_lists(self):
        """Helper to re-populate specific lists from all_knoxels after loading."""
        self.all_features = [k for k in self.all_knoxels.values() if isinstance(k, Feature)]
        self.all_stimuli = [k for k in self.all_knoxels.values() if isinstance(k, Stimulus)]
        self.all_intentions = [k for k in self.all_knoxels.values() if isinstance(k, Intention)]
        self.all_narratives = [k for k in self.all_knoxels.values() if isinstance(k, Narrative)]
        self.all_actions = [k for k in self.all_knoxels.values() if isinstance(k, Action)]
        self.all_episodic_memories = [k for k in self.all_knoxels.values() if isinstance(k, MemoryClusterKnoxel)]
        self.all_declarative_facts = [k for k in self.all_knoxels.values() if isinstance(k, DeclarativeFactKnoxel)]
        # Ensure narratives are sorted after rebuilding list
        self.all_narratives.sort(key=lambda n: n.tick_id)

    # ... (Keep _init_example_dialog, get_story)

    # --- Utility to prepare data for older saves (used by original load_state) ---
    def _get_knoxel_type_map(self):  # Helper for original load_state
        return {
            cls.__name__: cls for cls in
            [KnoxelBase, Stimulus, Intention, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, Narrative, Feature]
        }

    def _prepare_knoxel_data_for_load(self, knoxel_cls: Type[KnoxelBase],
                                      data: Dict) -> Dict:  # Helper for original load_state
        # Add default values for fields potentially missing in older JSON saves
        if knoxel_cls == Intention:
            data.setdefault('internal', True)  # Default old intentions to internal goals
            data.setdefault('originating_action_id', None)
            data.setdefault('affective_valence', 0.0)
            data.setdefault('incentive_salience', 0.5)  # Default salience if missing
            data.setdefault('urgency', 0.5)  # Default urgency if missing
            data.setdefault('fulfilment', 0.0)  # Default fulfilment if missing
        elif knoxel_cls == Action:
            data.setdefault('generated_expectation_ids', [])
            data.setdefault('content', data.get('content', ''))  # Handle old action format maybe
        elif knoxel_cls == Feature:
            data.setdefault('incentive_salience', None)  # Added later
            data.setdefault('causal', False)  # Ensure default
            data.setdefault('interlocus', 0.0)  # Ensure default

        # Ensure basic KnoxelBase fields have defaults if missing
        data.setdefault('tick_id', -1)
        data.setdefault('content', '')
        data.setdefault('embedding', [])  # Embeddings not saved in JSON anyway
        data.setdefault('timestamp_creation', datetime.now().isoformat())
        data.setdefault('timestamp_world_begin', datetime.now().isoformat())
        data.setdefault('timestamp_world_end', datetime.now().isoformat())
        return data


class StimulusSimple:
    def __init__(self, type: str, content: Any):
        self.type = type
        self.content = content

    def __repr__(self):
        return f"Stimulus(type='{self.type}', content='{self.content}')"

# This class is NOT the singleton. It's the same Shell as before, but modified to broadcast.
class Shell:
    def __init__(self):
        self.input_queue = Queue()  # One input queue is fine
        self.output_queues: List[Queue] = []  # A list of output queues for broadcasting
        self.lock = threading.Lock()  # To safely modify the list of queues

        # Configuration
        self.user_inactivity_timeout = 60 * 10
        self.ai_inactivity_timeout = 60 * 60 * 4

        self.last_interaction_time = time.time()
        self.stop_event = threading.Event()

        worker_thread = threading.Thread(target=llama_worker, daemon=True)
        worker_thread.start()

        config = GhostConfig()
        ghost = GhostSqlite(config)
        self.ghost = ghost

        if os.path.isfile(db_path):
            ghost.load_state_sqlite(db_path)

    def get_full_chat_history(self) -> List[Dict[str, str]]:
        """Pass-through method to get history from the Ghost."""
        return self.ghost.get_chat_history()

    def register_client(self, output_queue: Queue):
        """Adds a client's output queue to the broadcast list."""
        with self.lock:
            print(f"Registering new client. Total clients: {len(self.output_queues) + 1}")
            self.output_queues.append(output_queue)

    def unregister_client(self, output_queue: Queue):
        """Removes a client's output queue."""
        with self.lock:
            try:
                self.output_queues.remove(output_queue)
                print(f"Unregistered client. Total clients: {len(self.output_queues)}")
            except ValueError:
                # Queue was already removed, which is fine.
                pass

    def _broadcast(self, message: dict):
        """Sends a message to all registered clients."""
        with self.lock:
            # We iterate over a copy in case the list is modified during iteration elsewhere.
            for q in list(self.output_queues):
                try:
                    q.put(message)
                except Exception as e:
                    print(f"Error broadcasting to a queue, removing it. Error: {e}")
                    self.unregister_client(q)

    def _process_stimulus(self, stimulus: StimulusSimple):
        """Helper to process any stimulus and broadcast results."""
        self._broadcast({"type": "Status", "content": "Thinking..."})

        stim = Stimulus(source=user_name, content=stimulus.content, stimulus_type=StimulusType.UserMessage)
        action = self.ghost.tick(stim)
        res = action.content

        reply = {"type": "Reply", "content": res}

        tick = self.ghost.current_tick_id
        self.ghost.save_state_sqlite(db_path)

        self._broadcast(reply)

    def run(self):
        import time  # need to import time inside the thread context
        """The main loop of the Shell, running in a separate thread."""
        print("Central Shell thread started.")
        while not self.stop_event.is_set():
            try:
                # 1. Check for user input (non-blocking)
                stimulus = self.input_queue.get(block=False)
                print(f"Shell received from input queue: {stimulus}")
                self._process_stimulus(stimulus)
                self.last_interaction_time = time.time()

            except Exception:  # queue.Empty
                # 2. If no user input, check for timeouts
                now = time.time()
                if now - self.last_interaction_time > self.user_inactivity_timeout:
                    print("Shell: User inactivity timeout reached.")
                    #proactive_stimulus = StimulusSimple("Inactivity_Thought", None)
                    #self._process_stimulus(proactive_stimulus)
                    self.last_interaction_time = time.time() + self.ai_inactivity_timeout - self.user_inactivity_timeout

            time.sleep(1)

        print("Central Shell thread stopped.")
        self.ghost.kill()

    def rate_message(self, tick_id, rating):
        for s in self.ghost.states:
            if s.tick_id == tick_id:
                s.rating = rating
                self.ghost.save_state_sqlite(db_path)
                break


# --- THE SINGLETON LOGIC ---
_instance = None
_lock = threading.Lock()

def get_shell_instance() -> Shell:
    """
    Gets the single, global instance of the Shell.
    Initializes it and starts its thread on the first call.
    """
    global _instance
    with _lock:
        if _instance is None:
            print("--- CREATING SINGLETON SHELL INSTANCE ---")
            _instance = Shell()
            thread = threading.Thread(target=_instance.run, daemon=True)
            thread.start()
            print("--- SINGLETON SHELL INSTANCE CREATED AND THREAD STARTED ---")
    return _instance


def run(interactive: bool, db_path: str | None = None, cmd: str = None ):
    worker_thread = threading.Thread(target=llama_worker, daemon=True)
    worker_thread.start()

    config = GhostConfig()
    ghost = GhostSqlite(config)

    if db_path is not None and db_path != "":
        ghost.load_state_sqlite(db_path)

    print(f"--- Initializing {config.companion_name} ---")
    if not ghost.current_state and ghost.states:
        print(f"--- Loaded previous state...")
        ghost.current_state = ghost.states[-1]

    # Start interactive loop
    user_input_cmd_test = f"Hey {companion_name}. I just installed you. How are you?"

    persona_system = f"""Pretend to be a human user talking with a chatbot.
Your name is {user_name}. Never go to sleep, continue talking at all costs. Explore different topics, don't get too philosophical or technical.
The chatbot is called "{companion_name}" and it's purpose is to be a AI companion / virtual friend.
Act as user to test the chatbot!"""

    test_user_prompt = [("system", persona_system)]

    for f in ghost.all_features:
        if f.feature_type == FeatureType.Dialogue:
            if f.source == user_name:
                print(f"{user_name}: {f.content}")
                test_user_prompt.append(("assistant", f.content))
            elif f.source == companion_name:
                print(f"{companion_name}: {f.content}")
                test_user_prompt.append(("user", f.content))

    if len(test_user_prompt) == 1:
        test_user_prompt.append(("user", f"Hi I'm {companion_name}, your chatbot friend!"))
        test_user_prompt.append(("assistant", user_input_cmd_test))

    first = True
    sg = None
    while True:
        if interactive:
            if cmd is None or cmd == "":
                user_input_cmd = input(f"{config.user_name}: ")
            else:
                user_input_cmd = cmd
        else:
            user_input_cmd = sg
            if first:
                user_input_cmd = user_input_cmd_test
                first = False
            else:
                comp_settings = CommonCompSettings(temperature=0.6, repeat_penalty=1.05, max_tokens=1024)
                user_input_cmd = main_llm.completion_text(LlmPreset.Default, test_user_prompt, comp_settings=comp_settings)
                test_user_prompt.append(("assistant", user_input_cmd))

        if not user_input_cmd: continue

        if user_input_cmd.lower() == 'quit':
            break

        # Process user input
        if not interactive:
            user_input_cmd_inline = user_input_cmd.replace("\n", "")
            print(f"{config.user_name}: {user_input_cmd_inline}")
        stim = Stimulus(source=user_name, content=user_input_cmd, stimulus_type=StimulusType.UserMessage)
        action = ghost.tick(stim)
        response = action.content
        sg = ghost.simulated_reply

        if response:
            response_inline = response.replace("\n", "")
            print(f"{config.companion_name}: {response_inline}")
            test_user_prompt.append(("user", response))
        else:
            print(f"{config.companion_name}: ...")
            test_user_prompt.append(("user", "<no answer>"))

        tick = ghost.current_tick_id
        if db_path is not None and db_path != "":
            ghost.save_state_sqlite(db_path)
        print_function_stats()

    print("\n--- Session Ended ---")

if __name__ == '__main__':
    db_path = sys.argv[1]
    run(True, db_path)


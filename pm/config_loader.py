import logging
import os
import sys
from pathlib import Path

import yaml

from pm.tools.tools_common import get_toolset_from_url
from pm.utils.log_utils import setup_logger

logger = logging.getLogger(__name__)


companion_name: str = ""
user_name: str = ""
local_files_only = False
embedding_model = ""
embedding_dim = 768
vector_same_threshold = 0.08
character_card_story = ""
timestamp_format = "%A, %d.%m.%y %H:%M"
log_path = "./logs"
db_path = ""
commit = True
mcp_server_url: str = ""
available_tools = []
enable_tool_calling: bool = False
shell_system_name = "System-Agent"

# Get absolute path to the config file
CONFIG_PATH = "config.yaml"

def _find_pm_config_in_argv() -> str | None:
    """Return a user-provided config path if -pm_config or --pm_config is present."""
    try:
        argv = sys.argv  # don’t copy if not available
    except AttributeError:
        return None  # e.g. frozen app, REPL, etc.

    for i, arg in enumerate(argv):
        if arg in ("-pm_config", "--pm_config", "pm_config", "/pm_config"):
            # next argument exists and isn't another flag
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                return argv[i + 1]
    return None

env_path = os.environ.get("PM_CONFIG")
if env_path:
    CONFIG_PATH = Path(env_path).expanduser().resolve().as_posix()
if _find_pm_config_in_argv() is not None:
    CONFIG_PATH = _find_pm_config_in_argv()

if not os.path.exists(CONFIG_PATH):
    raise Exception("config.yaml doesn't exist can't be found. Make sure you're calling pm from the directory where config.yaml is, or set env var PM_CONFIG, or pass --pm_config <PATH>")

# Load YAML file
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

# Update global variables dynamically
global_map = {
    "db_path": config_data.get("db_path", ""),
    "companion_name": config_data.get("companion_name", ""),
    "user_name": config_data.get("user_name", ""),
    "embedding_model": config_data.get("embedding_model", ""),
    "embedding_dim": config_data.get("embedding_dim", 0),
    "timestamp_format": config_data.get("timestamp_format", ""),
    "character_card_story": config_data.get("character_card_story", ""),
    "commit": config_data.get("commit", True),
    "mcp_server_url": config_data.get("mcp_server_url", ""),
    "enable_tool_calling": config_data.get("enable_tool_calling", False)
}
globals().update(global_map)

setup_logger(log_path, "main.log")

wsl_pref = "/mnt/c/"
wsl_rep = "C:/"
if os.name == "nt":
    db_path = db_path.replace(wsl_pref, wsl_rep)
    embedding_model = embedding_model.replace(wsl_pref, wsl_rep)

if mcp_server_url == "http://127.0.0.1:8000/mcp":
    from mcp_demo.mcp_server import start_server
    start_server()
    logger.info("Started Demo MCP server...")

# Extract model configurations
models = config_data.get("models", {})
model_mapping = config_data.get("model_mapping", {})

# Generate model_map dictionary
context_size = -1
model_map = {
    "embedding_dim": embedding_dim,
    "embedding_model": embedding_model,
}
for model_class, model_key in model_mapping.items():
    if model_key in models:
        context_size = max(context_size, models[model_key]["context"])
        path = models[model_key]["path"]
        if os.name == "nt":
            path = path.replace(wsl_pref, wsl_rep)
        model_map[model_class] = {
            "model_key": model_key,
            "path": path,
            "layers": models[model_key]["layers"],
            "context": models[model_key]["context"],
            "last_n_tokens_size": models[model_key]["last_n_tokens_size"],
            "temperature": models[model_key]["temperature"],
            "top_k": models[model_key]["top_k"],
            "top_p": models[model_key]["top_p"],
            "min_p": models[model_key]["min_p"],
            "repeat_penalty": models[model_key]["repeat_penalty"],
            "frequency_penalty": models[model_key]["frequency_penalty"],
            "presence_penalty": models[model_key]["presence_penalty"],
            "mmproj_file": models.get("mmproj_file", ""),
            "thinking_mode": models.get("thinking_mode", False)
        }
    else:
        raise KeyError(f"Model key '{model_key}' in model_mapping not found in models section.")

QUOTE_START = '"'
QUOTE_END = '"'

toolset = get_toolset_from_url(mcp_server_url)
if toolset:
    for tool_name, tool_type in toolset.tool_router_model.model_fields.items():
        available_tools.append(tool_name)
tools_list_str = ", ".join(available_tools)

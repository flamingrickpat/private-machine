import os

import yaml
from pydantic import BaseModel, PrivateAttr

from pm.utils.log_utils import setup_logger

import os
pid = os.getpid()
print(pid)

class PmConfig(BaseModel):
    companion_name: str = ""
    user_name: str = ""
    local_files_only: bool = False
    embedding_model: str = ""
    embedding_dim: int = 768
    vector_same_threshold: float = 0.08
    character_card_story: str = ""
    timestamp_format: str = "%A, %d.%m.%y %H:%M"
    log_path: str = "./logs"
    db_path: str = ""
    commit: bool = True
    shell_system_name: str = "System-Agent"
    worker_context_limit_tokens: int = 0
    agent_context_weights: dict = {}
    agent_context_min_section_tokens: int = 96
    agent_context_latest_messages: int = 40
    agent_context_workspace_items: int = 48
    agent_context_target_ratio: float = 0.72
    agent_context_safety_margin_tokens: int = 220
    supported_capabilities: list[str] = []
    unsupported_capabilities: list = []
    capability_notes: list = []
    _model_map: dict | None = PrivateAttr(default=None)

def load_config(path: str) -> PmConfig:
    with open(path, "r", encoding="utf-8") as file:
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
        "enable_tool_calling": config_data.get("enable_tool_calling", False),
        "worker_context_limit_tokens": int(config_data.get("worker_context_limit_tokens", 0) or 0),
        "agent_context_weights": config_data.get("agent_context_weights", {}) or {},
        "agent_context_min_section_tokens": int(config_data.get("agent_context_min_section_tokens", 96) or 96),
        "agent_context_latest_messages": int(config_data.get("agent_context_latest_messages", 40) or 40),
        "agent_context_workspace_items": int(config_data.get("agent_context_workspace_items", 48) or 48),
        "agent_context_target_ratio": float(config_data.get("agent_context_target_ratio", 0.72) or 0.72),
        "agent_context_safety_margin_tokens": int(config_data.get("agent_context_safety_margin_tokens", 220) or 220),
        "supported_capabilities": config_data.get("supported_capabilities", []) or [],
        "unsupported_capabilities": config_data.get("unsupported_capabilities", []) or [],
        "capability_notes": config_data.get("capability_notes", []) or [],
    }
    cfg = PmConfig.model_validate(global_map)
    setup_logger(cfg.log_path, "main.log")

    # Extract model configurations
    models = config_data.get("models", {})
    model_mapping = config_data.get("model_mapping", {})

    # Generate model_map dictionary
    context_size = -1
    model_map = {
        "embedding_dim": cfg.embedding_dim,
        "embedding_model": cfg.embedding_model,
    }
    for model_class, model_key in model_mapping.items():
        if model_key in models:
            context_size = max(context_size, models[model_key]["context"])
            path = models[model_key]["path"]
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
                "mmproj_file": models[model_key]["mmproj_file"],
                "reasoning_model": models[model_key]["reasoning_model"],
                "user_name": cfg.user_name,
                "companion_name": cfg.companion_name
            }
        else:
            raise KeyError(f"Model key '{model_key}' in model_mapping not found in models section.")

    cfg._model_map = model_map

    return cfg
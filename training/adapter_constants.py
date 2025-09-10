from pathlib import Path

DEFAULT_MODEL = "/mnt/c/workspace/AI/gemma-3-12b-it-qat-q4_0-unquantized"#!/usr/bin/env python
ADAPTER_DIR   = Path("./adapters")
KV_DIR        = Path("./kv_store")
HISTORY_FILE  = Path("./chat_history.jsonl")
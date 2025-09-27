from enum import Enum
from queue import Queue
from types import FunctionType
from typing import Dict, Any
from typing import List
from typing import Type
from typing import (
    Union,
)

from pydantic import BaseModel

class LlmPreset(Enum):
    Internal = "Internal"
    Default = "Default"
    CurrentOne = "current"
    Fast = "Fast"

class CommonCompSettings:
    def __init__(self,
                 max_tokens: int = 1024,
                 temperature: float = None,
                 top_k: float = None,
                 top_p: float = None,
                 min_p: float = None,
                 repeat_penalty: float = None,
                 frequency_penalty: float = None,
                 presence_penalty: float = None,
                 stop_words: List[str] = None,
                 tools_json: List[Type[BaseModel]] = None, # Keep this for potential future use, but we primarily use the 'tools' parameter
                 completion_callback: Union[None, FunctionType] = None,
                 eval_only: bool = False,
                 seed: int = None,
                 duplex: bool = False,
                 queue_to_user: Queue = None,
                 queue_from_user: Queue = None,
                 disable_eos: bool = False,
                 allow_interrupts: bool = False,
                 wait_for_start_signal: bool = False,
                 caller_id: str = None
                 ):
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.stop_words = stop_words if stop_words is not None else []
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tools_json = tools_json if tools_json is not None else []  # Legacy?
        self.completion_callback = completion_callback
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.eval_only = eval_only
        self.seed = seed
        self.duplex = duplex
        self.queue_to_user = queue_to_user
        self.queue_from_user = queue_from_user
        self.disable_eos = disable_eos
        self.allow_interrupts = allow_interrupts
        self.wait_for_start_signal = wait_for_start_signal
        self.caller_id = caller_id

    def fill_defaults(self, model_map: Dict[str, Any], preset: LlmPreset):
        if preset.value in model_map.keys():
            if self.temperature is None:
                self.temperature = model_map[preset.value]["temperature"]
            if self.top_k is None:
                self.top_k = model_map[preset.value]["top_k"]
            if self.top_p is None:
                self.top_p = model_map[preset.value]["top_p"]
            if self.min_p is None:
                self.min_p = model_map[preset.value]["min_p"]
            if self.repeat_penalty is None:
                self.repeat_penalty = model_map[preset.value]["repeat_penalty"]
            if self.frequency_penalty is None:
                self.frequency_penalty = model_map[preset.value]["frequency_penalty"]
            if self.presence_penalty is None:
                self.presence_penalty = model_map[preset.value]["presence_penalty"]
        if self.seed is None:
            self.seed = 1337

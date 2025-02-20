from abc import ABC,abstractmethod
from pydantic import BaseModel, Field, validator
from typing import List, Union, Dict, Any, Optional, Callable
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

from typing import Dict, List, Any, Union
import enum

from pydantic import BaseModel, Field


class ToolResultStatus(int, enum.Enum):
    Error = -1,
    Success = 1,
    SuccessAndExit = 2

class ToolResult(BaseModel):
    status: ToolResultStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class ToolResultList(BaseModel):
    results: List[ToolResult] = Field(default_factory=list)


class LlmType(int, Enum):
    LlamaCpp = 1
    OpenAI = 2
    Classifier = 3
    Embedding = 4


class PromptKeyword(str, Enum):
    StopCache = "<|stop_cache|>"


class LlmPreset(str, Enum):
    CurrentOne = "CurrentOne"
    Conscious = "Conscious"
    CauseEffect = "CauseEffect"
    Auxiliary = "Auxiliary"
    Emotion = "Emotion"
    Summarize = "Summarize"
    ExtractFacts = "ExtractFacts"
    Default = "Default"
    ConvertToInternalThought = "ConvertToInternalThought"
    GenerateTot = "GenerateTot"

class LlmModel(BaseModel):
    type: LlmType = Field(default=LlmType.LlamaCpp)
    identifier: str = Field(default_factory=str)
    path: str = Field(default_factory=str)
    bos_token: str = Field(default_factory=str)
    eos_token: str = Field(default_factory=str)
    eot_token: str = Field(default_factory=str)
    eom_token: str = Field(default_factory=str)
    turn_system_token: str = Field(default_factory=str)
    turn_assistant_token: str = Field(default_factory=str)
    turn_user_token: str = Field(default_factory=str)
    turn_script_result_token: str = Field(default_factory=str)
    call_script_token: str = Field(default_factory=str)
    context_size: int = 4096
    default_temperature: float = Field(default=0.6)
    default_top_k: int = Field(default=50)
    default_top_p: float = Field(default=0.9)
    default_repeat_penalty: float = Field(default=1.1)
    n_gpu_layers: int = Field(default=-1)

    def get_tokens_for_tempalte(self):
        res = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "eot_token": self.eot_token,
            "eom_token": self.eom_token,
            "turn_system_token": self.turn_system_token,
            "turn_assistant_token": self.turn_assistant_token,
            "turn_user_token": self.turn_user_token,
            "turn_script_result_token": self.turn_script_result_token,
            "call_script_token": self.call_script_token,
        }
        return res

    def insert_newlines_after_special_tokens(self, text: str) -> str:
        tokens = self.get_tokens_for_tempalte()
        for token_name, token in tokens.items():
            if token:  # Only if the token is set
                # Use regex to replace the token with itself followed by a newline
                text = re.sub(re.escape(token), f"{token}\n", text)
        return text

class CommonCompSettings(BaseModel):
    max_tokens: int = Field(default=1024)
    stop_words: List[str] = Field(default_factory=list)
    seed: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=0.7)
    top_k: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    repeat_penalty: Optional[float] = Field(default=1.1)
    break_on_special_token: bool = Field(default=True)
    tools_func: List[Any] = Field(default_factory=list)
    tools_json: List[Any] = Field(default_factory=list)
    tools_json_optional: List[Any] = Field(default_factory=list)
    tools_func_llama: List[Any] = Field(default_factory=list)
    stop_on_eos: bool = Field(default=True)
    stop_on_eot: bool = Field(default=True)
    stop_on_eom: bool = Field(default=False)
    use_model_cache: bool = Field(default=False, description="Add <|stop_cache|> to prompt to mark the end of the cached area.")
    enforced_structure_regex: str = Field(default=None)
    enforced_structure_gbnf: str = Field(default=None)
    force_include_string: str = Field(default=None)
    frequency_penalty: float = Field(default=0)
    presence_penalty: float = Field(default=0)

    @validator('tools_func', pre=True)
    def val_tools_func(cls, value):
        if not isinstance(value, list):
            return [value]
        return value

    @validator('tools_func_llama', pre=True)
    def val_tools_func2(cls, value):
        if not isinstance(value, list):
            return [value]
        return value

    @validator('tools_json', pre=True)
    def val_tools_json(cls, value):
        if not isinstance(value, list):
            return [value]
        return value


class CompletionStopReason(int, Enum):
    Error = 0
    TokenLimit = 1
    StopToken = 2
    StopWord = 3
    FunctionCalled = 4
    MaxTokensGenerated = 5


class CompletionResult(BaseModel):
    user_prompt_raw: Optional[str] = Field(default=None)
    output_raw: Optional[str] = Field(default=None)
    output_sanitized: Optional[str] = Field(default=None)
    output_sanitized_inline: Optional[str] = Field(default=None)
    full_text_raw: Optional[str] = Field(default=None)
    function_calls: List[str] = Field(default_factory=list)
    function_output: List[str] = Field(default_factory=list)
    grammar: Optional[str] = Field(default=None)
    stop_reason: CompletionStopReason = Field(default=CompletionStopReason.Error)

    def __str__(self):
        return self.output_raw

    def parse_structure(self):
        pass


class PromptTemplate(BaseModel):
    prompt_template: str

    def __str__(self):
        return self.prompt_template


class Llm(ABC):
    @abstractmethod
    def set_model(self, settings: LlmModel):
        pass

    @abstractmethod
    def set_prompt_template(self, prompt: str):
        pass

    @abstractmethod
    def get_token_length(self, prompt: str):
        pass

    @abstractmethod
    def completion(self,
                   prompt: str,
                   comp_settings: Union[CommonCompSettings, None,] = None,
                   use_prompt_template: bool = False,
                   script_delegate: Callable[[str], List[ToolResult]] = None,
                   progress_delegte: Callable[[int, str], None] = None
                   ) -> CompletionResult:
        pass

    @abstractmethod
    def reset(self):
        pass

import json
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import InvalidToolCall, ToolCall, ToolCallChunk
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from typing_extensions import Self

from pm.llm.base_llm import CommonCompSettings


class ChatLlamaCppCustom(ChatLlamaCpp):
    llamacpp_llm: Any = Field(default=None)
    tools: Any = Field(default=None)
    functions: Any = Field(default=None)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        return self

    def set_client(self, llamacppllm):
        from pm.llm.llamacpp import LlamaCppLlm
        self.llamacpp_llm: LlamaCppLlm = llamacppllm
        self.client = llamacppllm.llama
        self.tools = []

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
            functions: List[Callable] = None
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        self.tools = tools
        if functions is None:
            functions = []
        self.functions = functions
        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling create_chat_completion."""
        params: Dict = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "logprobs": self.logprobs,
            "stop_sequences": self.stop,  # key here is convention among LLM classes
            "repeat_penalty": self.repeat_penalty,
        }
        if self.grammar:
            params["grammar"] = self.grammar

        if len(self.tools) > 0:
            comp_settings = CommonCompSettings(
                tools_json=self.tools
            )
            logits_processor = self.llamacpp_llm._get_logit_processor(comp_settings)
            params["logits_processor"] = logits_processor

        return params

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        comp_settings = CommonCompSettings(
            tools_json=self.tools,
            tools_func_llama=self.functions,
            max_tokens=1024,
            stop_words=stop[0] if stop is not None and len(stop) > 0 else "teststringnevergonnahappen"
        )
        response = self.llamacpp_llm.completion(prompt=self.llamacpp_llm.convert_langchain_to_raw_string(input),
                                                comp_settings=comp_settings)
        return AIMessage(content=response.output_sanitized)

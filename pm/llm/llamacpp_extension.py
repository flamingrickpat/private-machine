import json
import logging
import random
from datetime import datetime
from typing import Union, Optional, List, Callable, Tuple, Dict
import re
import gc
import ctypes
import contextlib
import os

import numpy as np
from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation

try:
    from llama_cpp import Llama, llama_cpp, LlamaGrammar
except Exception as e:
    print("Please download and install llama-cpp-python. I got mine from here: https://github.com/abetlen/llama-cpp-python/releases/tag/v0.2.90-cu124")
    exit(1)

from llama_cpp import LogitsProcessorList
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, \
    build_token_enforcer_tokenizer_data
from lmformatenforcer import JsonSchemaParser, RegexParser
import tqdm

from pm.llm.base_llm import (Llm, LlmModel, CommonCompSettings, CompletionResult, CompletionStopReason, ToolResult,
                             ToolResultStatus, ToolResultList)
from pm.utils.string_utils import list_ends_with, clip_last_unfinished_sentence, find_python_functions_in_text

logger = logging.getLogger(__name__)

default_settings = {
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "type_k": 2,
    "type_v": 2,
    "flash_attn": True,
    "n_batch": 512
}


def call_python_tools(funcs: str) -> List[Dict]:
    # Extract function name and arguments
    results: List[Dict] = []
    functions = re.findall(r'(\w+\(.*?\))', funcs)
    for function_string in functions:
        logger.info(function_string)

        func_name, args_str = function_string.strip().split('(', 1)
        args_str = args_str.rstrip(')')

        # Parse arguments
        args = {}
        if args_str:
            for arg in args_str.split(','):
                if arg.strip() == "item":
                    continue

                parts = arg.split('=')
                if len(parts) == 1 and parts[0] == "item":
                    continue
                elif len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"')  # Remove quotes if present
                    args[key] = value

        # Get the function from the registered functions dict
        from pm.controller import controller
        if func_name in controller.global_python_tools:
            func = controller.global_python_tools[func_name]
            # Call the function with unpacked arguments
            if "item" in func.__code__.co_varnames:
                args["item"] = None

            logger.info(f"Calling {func.__name__} with args {args}")
            try:
                res = func(**args)
            except Exception as e:
                res = {"error": f"Function '{func_name}' raised '{e}'!"}

            results.append(res)
        else:
            res = {"error": f"Function '{func_name}' is not registered"}
            results.append(res)
    return results


def silent_decode(b: bytes) -> str:
    return_val = ""
    with contextlib.suppress(UnicodeDecodeError):
        return_val = b.decode('utf-8')
    return return_val


def load_state_fast(self, state) -> None:
    """
    Don't copy anything since it doesn't change the output when on VRAM
    """
    assert self._ctx.ctx is not None
    # Only filling in up to `n_tokens` and then zero-ing out the rest
    self.scores[: state.n_tokens, :] = state.scores #.copy()
    self.scores[state.n_tokens:, :] = 0.0
    self.input_ids = state.input_ids #.copy()
    self.n_tokens = state.n_tokens
    state_size = state.llama_state_size
    LLamaStateArrayType = ctypes.c_uint8 * state_size
    llama_state = LLamaStateArrayType.from_buffer_copy(state.llama_state)

    if llama_cpp.llama_set_state_data(self._ctx.ctx, llama_state) != state_size:
        raise RuntimeError("Failed to set llama state data")

class LlamaCppLlm(Llm):

    def __init__(self, verbose: bool = False):
        self.temporary_settings = None
        self.tool_comp_settings = None
        self.llama: Llama = None
        self.identifier = ""
        self.model_settings: LlmModel | None = None
        self.state = None
        self.settings = {}
        self.use_gpu = True
        self.termination_tokens = []
        self.termination_tokens_string = []

        self.bos_token: List[int] = []
        self.eos_token: List[int] = []
        self.eot_token: List[int] = []
        self.eom_token: List[int] = []
        self.turn_system_token: List[int] = []
        self.turn_assistant_token: List[int] = []
        self.turn_user_token: List[int] = []
        self.turn_script_result_token: List[int] = []
        self.call_script_token: List[int] = []
        self.new_line_token: List[int] = []
        self.verbose: bool = verbose

    def convert_langchain_to_raw_string(self, prompt: List[Tuple[str, str]], comp_settings: CommonCompSettings, join_same_role: bool = True) -> str:
        self._internal_set_model()
        context_size = self.model_settings.context_size
        max_generation_tokens = comp_settings.max_tokens

        raw_string = self._detokenize(self.bos_token, special=True)

        # Add the system prompt without modification
        last_role = ""
        accumulated_message = ""
        effective_prompt = []
        total_token_count = 0

        # Process each part of the prompt
        for i, (role, message) in enumerate(prompt):
            if role == "system":
                # Add system token directly to the raw_string
                raw_string += self._detokenize(self.turn_system_token, special=True) + "\n" + message.strip() + self._detokenize(self.eot_token, special=True) + "\n"
            else:
                # Tokenize the message and calculate its token count
                message_tokens = self._tokenize(message, special=False)
                effective_prompt.append((role, message_tokens))
                total_token_count += len(message_tokens)

        # Calculate the max allowed tokens for the effective prompt (excluding system prompt)
        max_allowed_tokens = context_size - max_generation_tokens - len(self._tokenize(raw_string, special=True))

        # Trim messages from the start if the effective prompt exceeds the max allowed tokens
        while total_token_count > max_allowed_tokens and effective_prompt:
            removed_role, removed_tokens = effective_prompt.pop(0)
            total_token_count -= len(removed_tokens)

        # Reassemble the prompt from the trimmed effective prompt list
        for role, message_tokens in effective_prompt:
            message = self._detokenize(message_tokens, special=False)

            if role != last_role or not join_same_role:
                if accumulated_message:
                    # Append accumulated message for the previous role if switching roles
                    raw_string += accumulated_message.strip() + self._detokenize(self.eot_token, special=True) + "\n"
                    accumulated_message = ""

                # Add the appropriate role token
                if role == "user":
                    raw_string += self._detokenize(self.turn_user_token, special=True) + "\n"
                elif role == "assistant":
                    raw_string += self._detokenize(self.turn_assistant_token, special=True) + "\n"

            # Accumulate messages if joining the same role's turns
            accumulated_message += message + "\n" if join_same_role else message
            last_role = role

        # Append any remaining message from the last role
        if accumulated_message:
            raw_string += accumulated_message.strip()
            if last_role != "assistant":
                raw_string += self._detokenize(self.eot_token, special=True)

        # Add assistant token at the end if the last role wasn't assistant
        if last_role != "assistant":
            raw_string += self._detokenize(self.turn_assistant_token, special=True)

        return raw_string.strip()

    def set_model(self, settings: LlmModel):
        self.temporary_settings = settings

    def _internal_set_model(self):
        settings = self.temporary_settings
        if settings.identifier != self.identifier:
            self.identifier = settings.identifier
            self.model_settings = settings

            self.state = None
            self.settings = default_settings.copy()

            if settings.context_size is not None:
                self.settings["n_ctx"] = settings.context_size

            self.settings["n_gpu_layers"] = self.model_settings.n_gpu_layers
            self.use_gpu = self.settings["n_gpu_layers"] != 0

            # probably shouldn't make a copy when it lives on vram
            if self.use_gpu:
                Llama.load_state = load_state_fast

            self.llama = Llama(settings.path, verbose=self.verbose, **self.settings)

            tmp = [settings.eos_token]
            for st in tmp:
                if st == "":
                    continue
                tokens = self._tokenize(st, special=True)
                if len(tokens) != 1:
                    raise Exception(f"Invalid termination token! Check specification for {st}")
                self.termination_tokens.append(tokens[0])
                self.termination_tokens_string.append(st)

            self.new_line_token = self._tokenize("\n")
            self._initialize_tokens()

    def _initialize_tokens(self):
        token_fields = [
            'bos_token', 'eos_token', 'eot_token', 'eom_token',
            'turn_system_token', 'turn_assistant_token', 'turn_user_token',
            'turn_script_result_token', 'call_script_token'
        ]

        for field in token_fields:
            setting_value = getattr(self.model_settings, field)
            if setting_value is not None:
                tokens = self.llama.tokenize(setting_value.encode(encoding="utf-8"), add_bos=False, special=True)
                setattr(self, field, tokens)

    def get_token_length(self, prompt: str):
        tokens_long_prompt = self._tokenize(prompt, special=True)
        return len(tokens_long_prompt)

    def set_prompt_template(self, prompt: str):
        self.reset()
        tokens = self._tokenize(prompt, special=True)
        self.llama.eval(tokens)
        self.state = self.llama.save_state()

    def reset(self):
        self.llama.reset()
        self.state = None
        gc.collect()

    def _tokenize(self, text: str, special: bool = True, add_bos: bool = False) -> List[int]:
        return self.llama.tokenize(text.encode(encoding="utf-8"), special=special, add_bos=add_bos)

    def _detokenize(self, tokens: Union[int, List[int]], special: bool = False) -> str:
        if not isinstance(tokens, list):
            tokens = [tokens]
        return silent_decode(self.llama.tokenizer()._model.detokenize(tokens, special=special))

    def _get_full_grammar(self, comp_settings: Union[CommonCompSettings, None,]) -> Optional[LlamaGrammar]:
        if comp_settings is not None and comp_settings.enforced_structure_gbnf:
            from pm.llm.gbnf_grammar import generate_gbnf_structure_grammar
            tmp = generate_gbnf_structure_grammar(comp_settings.enforced_structure_gbnf)
            grammar = LlamaGrammar.from_string(tmp.gbnf, verbose=self.verbose)
            if grammar is None:
                raise Exception(f"Bad grammar: {tmp.gbnf}")
            return grammar

        if comp_settings is not None and comp_settings.tools_func is not None and len(comp_settings.tools_func) > 0:
            from pm.llm.gbnf_grammar import generate_gbnf_grammar
            tmp = generate_gbnf_grammar(comp_settings.tools_func)
            grammar = LlamaGrammar.from_string(tmp.gbnf, verbose=self.verbose)
            if grammar is None:
                raise Exception(f"Bad grammar: {tmp.gbnf}")
            return grammar

    def _get_func_block_grammar(self, comp_settings: Union[CommonCompSettings, None,]) -> Optional[LlamaGrammar]:
        if comp_settings is not None and comp_settings.tools_func_llama is not None and len(comp_settings.tools_func_llama) > 0:
            from pm.llm.gbnf_grammar import generate_gbnf_grammar
            tmp = generate_gbnf_grammar(comp_settings.tools_func_llama)
            grammar = LlamaGrammar.from_string(tmp.gbnf, verbose=self.verbose)
            if grammar is None:
                raise Exception(f"Bad grammar: {tmp.gbnf}")
            return grammar

    def _get_logit_processor(self, comp_settings: Union[CommonCompSettings, None,]) -> Optional[LogitsProcessorList]:
        if comp_settings is not None and comp_settings.enforced_structure_regex is not None:
            tokenizer_data = build_token_enforcer_tokenizer_data(self.llama)
            character_level_parser = RegexParser(comp_settings.enforced_structure_regex)
            if character_level_parser:
                logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(tokenizer_data, character_level_parser)])
                return logits_processors

        if comp_settings is not None and comp_settings.tools_json is not None and len(comp_settings.tools_json) > 0:
            if len(comp_settings.tools_json) > 1:
                raise Exception("not supported yet")

            os.environ["LMFE_STRICT_JSON_FIELD_ORDER"] = "1"
            tokenizer_data = build_token_enforcer_tokenizer_data(self.llama)
            schema = comp_settings.tools_json[0].schema()
            logger.info(f"Using schema: {schema}")
            character_level_parser = JsonSchemaParser(json_schema=schema)
            character_level_parser.config.force_json_field_order = True

            if character_level_parser:
                logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(tokenizer_data, character_level_parser)])
                return logits_processors
        return None

    def _get_logit_processor_optional(self, comp_settings: Union[CommonCompSettings, None,]) -> Optional[LogitsProcessorList]:
        if comp_settings is not None and comp_settings.tools_json_optional is not None and len(comp_settings.tools_json_optional) > 0:
            if len(comp_settings.tools_json_optional) > 1:
                raise Exception("not supported yet")

            os.environ["LMFE_STRICT_JSON_FIELD_ORDER"] = "1"
            tokenizer_data = build_token_enforcer_tokenizer_data(self.llama)
            schema = comp_settings.tools_json_optional[0].schema()
            logger.info(f"Using schema: {schema}")
            character_level_parser = JsonSchemaParser(json_schema=schema)
            character_level_parser.config.force_json_field_order = True

            if character_level_parser:
                logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(tokenizer_data, character_level_parser)])
                return logits_processors
        return None


    def completion_simple(self, prompt: str):
        output = self.llama.create_completion(prompt, max_tokens=4096, temperature=self.model_settings.default_temperature,
                                     repeat_penalty=self.model_settings.default_repeat_penalty,
                                     top_k=self.model_settings.default_top_k,
                                     top_p=self.model_settings.default_top_p, seed=1337)
        return output["choices"][0]["text"]



    def completion(self,
                   prompt: str,
                   comp_settings: Union[CommonCompSettings, None,] = None,
                   use_prompt_template: bool = False,
                   script_delegate: Callable[[str], List[ToolResult]] = call_python_tools,
                   progress_delegate: Callable[[int, str], None] = None,
                   show_progress: bool = False
                   ) -> CompletionResult:
        # only load model when actually using completion
        self._internal_set_model()

        if use_prompt_template and self.state is None:
            raise Exception("No saved state!")

        result = CompletionResult()
        result.user_prompt_raw = prompt

        base_settings = {
            "top_k": self.model_settings.default_top_k,
            "top_p": self.model_settings.default_top_p,
            "temp": self.model_settings.default_temperature,
            "repeat_penalty": self.model_settings.default_repeat_penalty
        }
        max_tokens = 128
        random.seed(str(datetime.utcnow()))
        seed = random.getrandbits(32)
        stop_words = []
        if comp_settings is None:
            comp_settings = CommonCompSettings()

        if comp_settings.top_k is not None:
            base_settings["top_k"] = comp_settings.top_k
        if comp_settings.top_p is not None:
            base_settings["top_p"] = comp_settings.top_p
        if comp_settings.temperature is not None:
            base_settings["temp"] = comp_settings.temperature
        if comp_settings.repeat_penalty is not None:
            base_settings["repeat_penalty"] = comp_settings.repeat_penalty
        if comp_settings.seed is not None:
            seed = comp_settings.seed
        if comp_settings.stop_words:
            for sw in comp_settings.stop_words:
                if sw != "":
                    stop_words.append(sw)
        if comp_settings.max_tokens is not None:
            max_tokens = comp_settings.max_tokens

        force_include_string = comp_settings.force_include_string

        context_size = self.model_settings.context_size

        grammar = None
        if comp_settings.tools_json and len(comp_settings.tools_json) > 0:
            gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(comp_settings.tools_json)
            grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=self.verbose)

        # ban that pesky eos token
        eos_token = self._tokenize(self.model_settings.eos_token, special=True)[0]
        logits_processor = None #self._get_logit_processor(comp_settings)
        logit_bias = {
            str(eos_token): 0 - float("inf")
        }
        logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}

        def logit_bias_processor(
            input_ids,
            scores,
        ):
            new_scores = np.copy(
                scores
            )  # Does it make sense to copy the whole array or can we just overwrite the original one?
            for input_id, score in logit_bias_map.items():
                new_scores[input_id] = score + scores[input_id]
            return new_scores

        _logit_bias_processor = LogitsProcessorList([logit_bias_processor])
        if logits_processor is None:
            logits_processor = _logit_bias_processor
        else:
            logits_processor = logits_processor.extend(_logit_bias_processor)

        # set seed
        self.llama.set_seed(seed)
        prompt_tokens = self._tokenize(prompt)
        reset = True
        if use_prompt_template:
            self.llama.load_state(self.state)
            reset = False
        else:
            self.llama.reset()
            self.llama.eval(prompt_tokens)

        completion_tokens = []
        script_tokens = []
        cnt = len(prompt_tokens) + 1
        new_cnt = 0
        last_autodetect_func_position = 0

        is_writing_code = False
        temp_grammar = None
        is_writing_tool_call = False
        tool_bracket_count = 0

        possible_func_names = []
        if comp_settings is not None and comp_settings.tools_func_llama is not None and len(comp_settings.tools_func_llama) > 0:
            possible_func_names.extend([x.__name__ for x in comp_settings.tools_func_llama])
        if comp_settings is not None and comp_settings.tools_func is not None and len(comp_settings.tools_func) > 0:
            possible_func_names.extend([x.__name__ for x in comp_settings.tools_func])

        pbar = None
        if show_progress:
            pbar = tqdm.tqdm(total=max_tokens)
            pbar.update(cnt)

        n_tokens = 0
        sample_idx = n_tokens + len(prompt_tokens) - 1

        while True:
            if cnt > context_size:
                result.stop_reason = CompletionStopReason.TokenLimit
                break

            if new_cnt >= max_tokens:
                result.stop_reason = CompletionStopReason.MaxTokensGenerated
                break

            # make sure its iterable
            def eval_append(tl):
                nonlocal cnt
                nonlocal pbar
                nonlocal progress_delegate
                nonlocal completion_tokens
                nonlocal new_cnt

                if not isinstance(tl, list):
                    tl = [tl]

                for ts in tl:
                    cnt += 1
                    new_cnt += 1
                    if pbar is not None:
                        pbar.update(1)
                    self.llama.eval([ts])
                    completion_tokens.append(ts)

                    if progress_delegate is not None:
                        progress_delegate(cnt, self._detokenize(ts, special=True))

            # execute
            def execute_script_block(script_block: str) -> bool:
                nonlocal script_tokens
                nonlocal is_writing_code
                nonlocal last_autodetect_func_position
                nonlocal cnt

                if script_delegate is None:
                    script_res_str = "Error: The scripting delegate is not set up!"
                else:
                    res = script_delegate(script_block)
                    script_res_str = json.dumps(res, indent=2)

                res_tokens = self._tokenize(script_res_str, special=False)
                eval_append(token)
                eval_append(self.new_line_token)
                eval_append(self.turn_script_result_token)
                eval_append(self.new_line_token)
                eval_append(res_tokens)
                eval_append(self.eot_token)

                if result.stop_reason == CompletionStopReason.FunctionCalled:
                    return True

                eval_append(self.new_line_token)
                eval_append(self.turn_assistant_token)

                result.function_calls.append(script_block)
                result.function_output.append(script_res_str)

                script_tokens = []
                is_writing_code = False
                last_autodetect_func_position = cnt
                return False

            token = self.llama.sample(logits_processor=logits_processor,
                                      grammar=grammar if grammar is not None else temp_grammar,
                                      idx=sample_idx, **base_settings)

            detok_token = self._detokenize(token, special=False)
            if not is_writing_tool_call and "{" in detok_token and len(comp_settings.tools_json_optional) > 0:
                is_writing_tool_call = True
                logits_processor = self._get_logit_processor_optional(comp_settings)
                token = self.llama.sample(logits_processor=logits_processor,
                                          grammar=grammar if grammar is not None else temp_grammar,
                                          idx=sample_idx, **base_settings)

            if is_writing_tool_call:
                tool_bracket_count += detok_token.count("{")
                tool_bracket_count -= detok_token.count("}")
                if tool_bracket_count == 0:
                    is_writing_tool_call = False
                    logits_processor = None

            sample_idx += 1

            if token in self.eot_token and force_include_string is not None:
                tmp = self._detokenize(completion_tokens, special=True)
                if force_include_string not in tmp:
                    force_tokens = self._tokenize(comp_settings.force_include_string, special=False)
                    eval_append(force_tokens)
                    force_include_string = None
                    continue

            if is_writing_code:
                script_tokens.append(token)

            if (token in self.eom_token or token in self.eot_token) and is_writing_code:
                token = self.eom_token
                temp_grammar = None

                all_script = self._detokenize(script_tokens, special=False)
                should_break = execute_script_block(all_script)
                if should_break:
                    break
            else:
                all_text = self._detokenize(completion_tokens, special=True)

                if (comp_settings.stop_on_eom and list_ends_with(completion_tokens, self.eom_token)) or\
                    (comp_settings.stop_on_eos and list_ends_with(completion_tokens, self.eos_token)) or\
                    (comp_settings.stop_on_eot and list_ends_with(completion_tokens, self.eot_token)):
                    result.stop_reason = CompletionStopReason.StopToken
                    break

                sw_break = False
                for sw in stop_words:
                    if sw in all_text:
                        result.stop_reason = CompletionStopReason.StopWord
                        sw_break = True
                        break

                if sw_break:
                    break

                if token in self.call_script_token:
                    is_writing_code = True
                    temp_grammar = self._get_func_block_grammar(comp_settings)

                eval_append(token)

                if not is_writing_code and len(possible_func_names) > 0:
                    new_slice = self._detokenize(completion_tokens[last_autodetect_func_position:], special=True)
                    funcs = find_python_functions_in_text(new_slice, possible_func_names)
                    if len(funcs) > 0:
                        last_autodetect_func_position = cnt
                        call_block = "\n".join(funcs)
                        should_break = execute_script_block(call_block)
                        if should_break:
                            break

        combined_tokens = prompt_tokens + completion_tokens
        result.full_text_raw = self._detokenize(combined_tokens, special=True)
        result.output_raw = self._detokenize(completion_tokens, special=True)
        result.output_sanitized = self._detokenize(completion_tokens)

        if result.stop_reason == CompletionStopReason.TokenLimit or result.stop_reason == CompletionStopReason.MaxTokensGenerated:
            result.output_sanitized = clip_last_unfinished_sentence(result.output_sanitized)

        for sw in stop_words:
            result.output_sanitized = result.output_sanitized.replace(sw, "")
        for sw in self.termination_tokens_string:
            result.output_sanitized = result.output_sanitized.replace(sw, "")
        result.output_sanitized = result.output_sanitized.strip()
        result.output_sanitized_inline = result.output_sanitized.replace("\n", "\\n")

        logger.info(self.model_settings.insert_newlines_after_special_tokens(result.full_text_raw))
        result.parse_structure()

        return result

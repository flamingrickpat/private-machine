import asyncio
import contextlib
import copy
import datetime
import difflib
import gc
import json
import logging
import os
import queue
import random
import string
import sys
import textwrap
import base64
import ctypes
from argparse import ArgumentError
from dataclasses import dataclass, field
from typing import (
    Union,
    Tuple, Sequence,
)
import re
from typing import Type
from typing import Dict, Any
import time
from typing import List

import llama_cpp.llama_cpp as llama_cpp
import numpy as np
from llama_cpp import Llama, LlamaGrammar, suppress_stdout_stderr, LlamaSamplingContext, LlamaSamplingParams
import psutil

from pm.utils.datetime_utils import get_causal_datetime
from pm.utils.threading_utils import crash_hard

try:
    from fastmcp import Client
except Exception:
    Client = None
from json_repair import repair_json
from lmformatenforcer import JsonSchemaParser, TokenEnforcerTokenizerData
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor
from pydantic import BaseModel

from pm.system.llm.llm_common import LlmPreset, CommonCompSettings
from pm.system.llm.llm_manager import LlmManager
from pm.utils.tool_utils import create_tool_router_model
from pm.utils.pydantic_utils import generate_pydantic_json_schema_str, generate_pydantic_markdown_str, dataclass_from_dict
from pm.utils.string_utils import remove_n_words, parse_json_from_response, between, replace_tagged_blocks, remove_strings
from pm.utils.duplex_utils import DuplexSignalFinish, DuplexSignalInterrupt, DuplexStartGenerationText, DuplexStartGenerationTool, DuplexSignalEog, DuplexSignalTerminate, DuplexSignalFinished, DuplexJsonBegin, DuplexAssistantInjectEnd, DuplexAssistantInjectBegin
from pm.utils.gbnf_utils import better_generate_gbnf_grammar_and_documentation, fix_gbnf_grammar_generator
fix_gbnf_grammar_generator()

logger = logging.getLogger(__name__)

LOG_INITIAL_PROMPT = True
CTX_LEN_SAFE_LENGTH = 32
REASON_NO_THINK = "<think>\n\n</think>\n\n"
REASON_THINK = "<think>"
ALLOWED_GRAMMAR_CHARACTERS = re.compile(r"^[\x20-\x7EÄÖÜäöüß€£¥¢°§±µ²³¼½¾¼ß\u00A0-\u024F\u1F600-\u1F64F]*$")
UNIVERSAL_IMAGE_BEGIN = "<begin_image>"
UNIVERSAL_IMAGE_END = "<end_image>"
MAGIC_IMAGE_TOKEN = -1
PHYSICAL_CORES = psutil.cpu_count(logical=False) - 1

def _strftime_now(f):
    return get_causal_datetime().strftime(f)

def _log_conversation(conversation: list[tuple[str, str]] | str, file_path: str, max_width: int = 80, addendum: List[str] = None):
    if file_path is None or file_path == "":
        return

    crash = False

    with open(file_path, 'w', encoding='utf-8') as file:
        if isinstance(conversation, str):
            if "{user_name}" in conversation or "{companion_name}" in conversation:
                crash = True
            file.write(conversation)
        else:
            for role, message in conversation:
                # Add a clear indicator for the role
                file.write(f"{'=' * 10} {role.upper()} {'=' * 10}\n\n")

                if "{user_name}" in message or "{companion_name}" in message:
                    crash = True

                # Wrap the message for readability
                wrapped_message = textwrap.fill(message.replace("\n", "µ"), width=max_width).replace("µ", "\n")
                file.write(wrapped_message + "\n\n")

                # Add a separator between turns
                file.write(f"{'-' * max_width}\n\n")

        if addendum:
            file.write(f"\n")
            file.write("\n".join(addendum))

    if crash:
        crash_hard("Unformatted names still in prompt!")

def _sample_from_logits(
    x: np.ndarray,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 40,
    fast: bool = False,
    rng: np.random.Generator | None = None,
) -> int:
    if fast or temperature == 0:
        return int(np.argmax(x))

    if rng is None:
        rng = np.random.default_rng(seed=time.time_ns())

    n = x.shape[0]
    k = n if not top_k or top_k <= 0 or top_k >= n else top_k

    # Work only on top-k logits
    idx = np.argpartition(x, -k)[-k:]
    vals = np.array(x[idx], copy=True)  # small copy: only k elements

    # sanitize
    vals[~np.isfinite(vals)] = -1e10

    # temperature
    if temperature != 1.0:
        vals /= max(temperature, 1e-6)

    # stable softmax in-place on the small buffer
    vals -= np.max(vals)
    np.exp(vals, out=vals)
    s = vals.sum()
    if s <= 0 or not np.isfinite(s):
        return int(idx[np.argmax(vals)])
    vals /= s

    # top-p on the top-k slice only
    if 0 < top_p < 1.0 and len(vals) > 1:
        order = np.argsort(vals)[::-1]
        sorted_vals = vals[order]
        sorted_idx = idx[order]

        cum = np.cumsum(sorted_vals)
        keep_count = np.searchsorted(cum, top_p, side="left") + 1
        sorted_vals = sorted_vals[:keep_count]
        sorted_idx = sorted_idx[:keep_count]

        s = sorted_vals.sum()
        if s <= 0 or not np.isfinite(s):
            return int(sorted_idx[0])
        sorted_vals /= s

        return int(rng.choice(sorted_idx, p=sorted_vals))

    return int(rng.choice(idx, p=vals))

def _split_by_value(lst, sep):
    """Split into runs of non-sep values; each sep becomes its own [sep]."""
    out, run = [], []
    for x in lst:
        if x == sep:
            if run:
                out.append(run)
                run = []
            out.append([x])  # the separator as its own group
        else:
            run.append(x)
    if run:
        out.append(run)
    return out

def _build_regular_tokens_list_fast(llm) -> list[tuple[int, str, bool]]:
    token_0 = llm.tokenize(b"0")[-1]
    regular_tokens = []
    special_tokens = {llm.token_bos(), llm.token_eos()}
    n_vocab = llm.n_vocab()

    for token_idx in range(n_vocab):
        if token_idx in special_tokens:
            continue

        try:
            decoded = llm.detokenize([token_idx]).decode("utf-8", errors="ignore")

            # Skip empty or bad tokens immediately
            if not decoded:
                continue

            # Skip tokens with non-allowed characters (non-ascii / non-german / no emoji)
            if not ALLOWED_GRAMMAR_CHARACTERS.match(decoded):
                continue

            # Compute word start flag cheaply
            decoded_after_0 = llm.detokenize([token_0, token_idx]).decode("utf-8", errors="ignore")
            is_word_start_token = len(decoded_after_0) > len(decoded)

            regular_tokens.append((token_idx, decoded, is_word_start_token))

        except Exception:
            continue

    return regular_tokens


def _build_token_enforcer_tokenizer_data_fast(llm: Llama) -> TokenEnforcerTokenizerData:
    regular_tokens = _build_regular_tokens_list_fast(llm)

    def decoder_fn(sent: List[int]) -> str:
        try:
            return llm.detokenize(sent).decode('utf-8')
        except:
            return decoder_fn(sent[:-1])

    return TokenEnforcerTokenizerData(regular_tokens, decoder_fn, llm.token_eos(), False, llm.n_vocab())

@dataclass
class _SchemaState:
    current_target_bm: Type[BaseModel] | None = None
    character_level_parser: Any | None = None
    apply_bias_func: Any | None = None
    constraint_start_idx: int | None = None

    def clear(self) -> None:
        self.current_target_bm = None
        self.character_level_parser = None
        self.apply_bias_func = None
        self.constraint_start_idx = None

@dataclass
class _GenerationState:
    comp_settings: CommonCompSettings
    sampler_args: dict
    static_prefix_n: int
    prefix_hit: bool
    user_to_ass_string: str
    user_to_ass_string_reason: str

    schema_state: _SchemaState = field(default_factory=_SchemaState)

    completion_tokens: List[int] = field(default_factory=list)
    stream_prev_text: str = ""
    finish_reason: str = "length"

    duplex_user_buffer: str = ""
    is_waiting_for_user_input: bool = False
    duplex_generation_mode_is_set: bool = False
    halt_signal: bool = False

    sample_time: float = 0.0
    sample_count: int = 0
    start_time: float = 0.0

    max_tokens: int = 4096

    def __post_init__(self) -> None:
        self.max_tokens = self.comp_settings.max_tokens or 4096
        self.start_time = time.time()

class LlmManagerLLama(LlmManager):
    def __init__(self, model_map: Dict[str, Any], test_mode: bool=False):
        super().__init__(model_map, test_mode)
        self.orig_is_hybrid: bool = True
        self.tokenizer_data = None
        self.clip_model_path = None
        self.chat_template = None
        self.current_ctx = None
        self.model_path = None
        self.llm: Llama | None = None
        self.eos_token_id = -1
        self.state = None
        self.state_tokens = None
        self.state_input_ids = None
        self.state_scores = None
        self.mtmd_ctx = None
        self.user_to_ass_string_raw = None
        self.ass_to_user_string = None
        self._mtmd_cpp = None
        self.reasoning_model = False
        self.use_prefix_caching: bool = True

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

    def _init_mtmd_context(self, llama_model: Llama):
        """Initialize mtmd context with the llama model."""
        if self.mtmd_ctx is not None:
            return  # Already initialized

        import llama_cpp.mtmd_cpp as mtmd_cpp

        self.verbose = False
        self._mtmd_cpp = mtmd_cpp

        with suppress_stdout_stderr(disable=self.verbose):
            # Get default parameters
            ctx_params = self._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = True
            ctx_params.print_timings = self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.verbosity = 2 if self.verbose else 0  # GGML_LOG_LEVEL_INFO = 2

            # Initialize mtmd context
            self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                self.clip_model_path.encode(),
                llama_model.model,
                ctx_params
            )

            if self.mtmd_ctx is None:
                raise ValueError(f"Failed to load mtmd context from: {self.clip_model_path}")

            # Check if vision is supported
            if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")

            def mtmd_free():
                with suppress_stdout_stderr(disable=self.verbose):
                    if self.mtmd_ctx is not None:
                        self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
                        self.mtmd_ctx = None

    def load_model(self, preset: LlmPreset):
        if preset == LlmPreset.CurrentOne:
            return

        if preset.value in self.model_map.keys():
            ctx = int(self.model_map[preset.value]["context"])
            last_n_tokens_size = int(self.model_map[preset.value]["last_n_tokens_size"])
            model_path = self.model_map[preset.value]["path"]
            layers = self.model_map[preset.value]["layers"]
            mmproj_file = self.model_map[preset.value]["mmproj_file"]
            self.reasoning_model = self.model_map[preset.value]["reasoning_model"]
        else:
            raise Exception("Unknown model!")

        if self.model_path != model_path:
            time.sleep(1)
            gc.collect()
            time.sleep(1)

            with suppress_stdout_stderr(disable=False):
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=layers,
                    n_ctx=ctx,
                    verbose=False,
                    last_n_tokens_size=last_n_tokens_size,
                    n_threads=PHYSICAL_CORES,
                    flash_attn_type=llama_cpp.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO if layers != 0 else llama_cpp.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED,
                )
                self.eos_token_id = self.llm.metadata["tokenizer.ggml.eos_token_id"]

                self.current_ctx = ctx
                self.model_path = model_path
                from jinja2 import Template, StrictUndefined
                self.chat_template = Template(self.llm.metadata["tokenizer.chat_template"], undefined=StrictUndefined)
                self.tokenizer_data = _build_token_enforcer_tokenizer_data_fast(self.llm)
                self.orig_is_hybrid = self.llm.is_hybrid

            if mmproj_file is not None and mmproj_file != "":
                self.clip_model_path = mmproj_file
                self._init_mtmd_context(self.llm)

            self._update_special_strings()

    def _replace_names_input(self, text: str):
        return (text.replace("{{companion_name}}", self.model_map[LlmPreset.Internal.value]["companion_name"])
                .replace("{companion_name}", self.model_map[LlmPreset.Internal.value]["companion_name"])
                .replace("{{user_name}}", self.model_map[LlmPreset.Internal.value]["user_name"])
                .replace("{user_name}", self.model_map[LlmPreset.Internal.value]["user_name"]))

    def _load_image(self, image_url: str) -> bytes:
        try:
            return base64.b64decode(image_url)
        except Exception:
            if image_url.startswith("data:"):
                image_bytes = base64.b64decode(image_url.split(",")[1])
                return image_bytes
            elif os.path.exists(image_url):
                with open(image_url, 'rb') as binary_file:
                    binary_file_data = binary_file.read()
                    return binary_file_data
            else:
                import urllib.request
                with urllib.request.urlopen(image_url) as f:
                    image_bytes = f.read()
                    return image_bytes

    def _create_bitmap_from_bytes(self, image_bytes: bytes):
        """Create mtmd_bitmap from image bytes."""
        if self.mtmd_ctx is None:
            raise ValueError("mtmd context not initialized")

        with suppress_stdout_stderr(disable=self.verbose):
            # Create bitmap from buffer using helper function
            n = len(image_bytes)
            buf = (ctypes.c_ubyte * n).from_buffer_copy(image_bytes)  # makes a copy

            bitmap = self._mtmd_cpp.mtmd_helper_bitmap_init_from_buf(self.mtmd_ctx, buf, n)
            if not bitmap:
                raise ValueError("Failed to create bitmap from image bytes")
            return bitmap

    def _render(self, messages, functions = None, function_call = None, tools = None, tool_choice = None, enable_thinking: bool = False, add_generation_prompt: bool = False):
        def raise_exception(_msg: str):
            print(self.llm.metadata["tokenizer.chat_template"])
            print()
            print(_msg)

        text = self.chat_template.render(
            messages=messages,
            add_generation_prompt=True,
            eos_token=self.llm.detokenize([self.llm.token_eos()]),
            bos_token=self.llm.detokenize([self.llm.token_bos()]),
            strftime_now=_strftime_now,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            raise_exception=raise_exception,
            enable_thinking=enable_thinking
        )
        return text

    def _reconstruct_turns_from_tokens(self, input_ids: List[int]):
        chunks = []
        cur = []

        for t in input_ids:
            if t == self.llm.token_eot():
                if cur:
                    chunks.append(cur)
                    cur = []
            else:
                cur.append(t)

        if cur:
            chunks.append(cur)

        texts = [self._detokenize(chunk, True).strip() for chunk in chunks]

        if not texts:
            return []

        # if the first chunk looks like system prompt
        first = texts[0].lower()

        has_system_turn = False
        if "system" in first[:50]:
            has_system_turn = True

        roles = []
        for i in range(len(texts)):
            if has_system_turn:
                if i == 0:
                    role = "system"
                elif i % 2 == 1:
                    role = "user"
                else:
                    role = "assistant"
            else:
                if i % 2 == 0:
                    role = "user"
                else:
                    role = "assistant"

            roles.append(role)

        # ---- merge small chunks (optional cleanup) ----
        turns = []

        for role, text in zip(roles, texts):
            if not text:
                continue

            if turns and turns[-1][0] == role:
                turns[-1] = (role, turns[-1][1] + "\n" + text)
            else:
                turns.append((role, text))

        return turns

    def _update_special_strings(self):
        # we sometimes seed the assistant response to get rid of crap like "ok, lets tackle this, here is a bunch of useless words and now here is what you wanted :) <result>"
        if self.clip_model_path is None or self.clip_model_path == "":
            messages = [
                {"role": "system", "content": "test1", "tool_calls": None},
                {"role": "user", "content": "test2@test3@test4", "tool_calls": None},
                {"role": "assistant", "content": "test5", "tool_calls": None},
                {"role": "user", "content": "test6", "tool_calls": None},
                {"role": "assistant", "content": "test7", "tool_calls": None}
            ]
        else:
            messages = [
                {"role": "system", "content": "test1", "tool_calls": None},
                {
                    "role": "user",
                    "tool_calls": None,
                    "content": [
                        {"type": "text", "text": "test2"},
                        {"type": "type", "image_url": {"url": "test3", "image_url": "test3", "data_url": "test3"}, "image": {"url": "test3", "image_url": "test3", "data_url": "test3"}},
                        {"type": "text", "text": "test4"},
                    ]
                },
                {"role": "assistant", "content": "test5", "tool_calls": None},
                {"role": "user", "content": "test6", "tool_calls": None},
                {"role": "assistant", "content": "test7", "tool_calls": None}
            ]

        try:
            text = self._render(messages)
        except Exception as e:
            messages = messages[1:]  # remove sys prompt in case its not allowed for model
            text = self._render(messages)

        self.remove_ass_string = text.split("test7")[1]
        self.user_to_ass_string_raw = between(text, "test6", "test7")
        self.user_to_ass_string_raw = self.user_to_ass_string_raw.replace("<think>", "").replace("</think>", "").strip()
        self.ass_to_user_string = between(text, "test5", "test6")
        if "test3" in text:
            self.text_to_image_string = between(text, "test2", "test3")
            self.image_to_text_string = between(text, "test3", "test4")
        else:
            self.text_to_image_string = between(text, "test2", "test4")
            self.image_to_text_string = ""

        self.media_marker_string = "<IMG>"
        if self._mtmd_cpp:
            self.media_marker_string = self._mtmd_cpp.mtmd_default_marker().decode('utf-8')


    def _tokenize_prompt(self,
                         preset: LlmPreset,
                         inp: List[Tuple[str, str]],
                         comp_settings: CommonCompSettings,
                         sysprompt_addendum: str | None = None
                        ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], List[int], List[Any]]:
        # change format
        inp_formatted = []
        for msg in inp:
            inp_formatted.append((msg[0].strip(), msg[1].strip()))

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

        # replace names and stuff
        inp_formatted_replaced = []
        for msg in inp_formatted:
            inp_formatted_replaced.append((msg[0], self._replace_names_input(msg[1])))
        inp_formatted = inp_formatted_replaced

        # convert format
        openai_inp = []
        for msg in inp_formatted:
            openai_inp.append({"role": msg[0], "content": msg[1], "tool_calls": None})

        cached_image_chunks = []
        begin_image_token = -1
        removed_words = []

        add_generation_prompt = openai_inp[-1]["role"] != "assistant"

        # remove messages until it fits in context
        rendered_data = None
        trimmed = False
        while True:
            try:
                rendered_data = self._render(openai_inp, enable_thinking=comp_settings.enable_thinking, add_generation_prompt=add_generation_prompt)
            except Exception as e:
                print(f"Error when trimming prompt: {e}\n{openai_inp}")

            # strip each line to save tokens; lots of unnecessary whitespaces from indented block strings
            #rendered_data = "\n".join([x.strip() for x in rendered_data.splitlines()])

            # let the assistant continue if they talked last, remove last eot
            if not add_generation_prompt:
                i = rendered_data.rfind(self._detokenize(self.llm.token_eot(), special=True))
                if i != -1:
                    rendered_data = rendered_data[:i].strip()

            # remove thinking tags
            if not self.reasoning_model:
                rendered_data = remove_strings(rendered_data, "</think>", "<think>")

            # replace visual input
            rendered_data, images = replace_tagged_blocks(rendered_data,
                                                          in_begin=UNIVERSAL_IMAGE_BEGIN,
                                                          in_end=UNIVERSAL_IMAGE_END,
                                                          out_start=self.text_to_image_string,
                                                          out_media=self.media_marker_string,
                                                          out_end=self.image_to_text_string)

            for image in images:
                img_bytes = self._load_image(image)
                bitmap_llama = self._create_bitmap_from_bytes(img_bytes)
                bitmaps = [bitmap_llama]

                image_prompt_text = f"test1{self.image_to_text_string}{self.media_marker_string}{self.text_to_image_string}test2"

                input_text = self._mtmd_cpp.mtmd_input_text()
                input_text.text = image_prompt_text.encode('utf-8')
                input_text.add_special = True
                input_text.parse_special = True

                # Create input chunks
                chunks = self._mtmd_cpp.mtmd_input_chunks_init()
                if chunks is None:
                    raise ValueError("Failed to create input chunks")

                bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                result = self._mtmd_cpp.mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(bitmaps)
                )

                if result != 0:
                    raise ValueError(f"Failed to tokenize input: error code {result}")

                n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)
                if n_chunks != 3:
                    raise ValueError(f"Chunking failed with len: {n_chunks}")

                # extract image chunks
                real_text_to_image_string = ""
                real_image_to_text_string = ""
                for i in range(n_chunks):
                    chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None:
                        continue

                    chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                    if chunk_type == self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        # Handle text chunk
                        n_tokens_out = ctypes.c_size_t()
                        tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                            chunk, ctypes.byref(n_tokens_out)
                        )

                        if tokens_ptr and n_tokens_out.value > 0:
                            # Convert ctypes array to Python list
                            tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]
                            if i == 0:
                                begin_image_token = tokens[-1]
                                real_text_to_image_string = self.llm.detokenize([tokens[-1]], special=True)
                            elif i == 2:
                                real_image_to_text_string = self.llm.detokenize([tokens[0]], special=True)
                    elif chunk_type in [self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_IMAGE, self._mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_AUDIO]:
                        cached_image_chunks.append((chunks, chunk))

                # free llama bitmap
                self._mtmd_cpp.mtmd_bitmap_free(bitmap_llama)

                # replace errors from template with real special tokens to avoid double tokens (mtmd already adds them)"
                rendered_data = rendered_data.replace(
                    f"{self.text_to_image_string}{self.media_marker_string}{self.image_to_text_string}",
                    f"{real_text_to_image_string}{real_image_to_text_string}",
                )

            tokenized_prompt = self.llm.tokenize(
                rendered_data.encode("utf-8"),
                add_bos=True,
                special=True,
            )

            # insert magic token so we know when to eval chunk
            if begin_image_token != -1:
                tokenized_prompt = [y for x in tokenized_prompt for y in ([x, MAGIC_IMAGE_TOKEN] if x == begin_image_token else [x])]

            # start deleting in middle
            if len(tokenized_prompt) > ((self.current_ctx - comp_settings.max_tokens) * 1.5):
                crash_hard(f"Too many tokens: 50% more than allowed, check prompt generation logic: {len(tokenized_prompt)} with only {self.current_ctx} allowed!")

            if (self.current_ctx - comp_settings.max_tokens) - 32 < 1:
                crash_hard("Negative available tokens, check context and max_tokens!")
            if len(tokenized_prompt) > (self.current_ctx - comp_settings.max_tokens) - 32:
                for i in range(1, len(openai_inp)):
                    cur = openai_inp[i]["content"]
                    trimmed = remove_n_words(cur, 32, removed_words)
                    openai_inp[i]["content"] = trimmed
                    trimmed = True
            else:
                break

        if trimmed:
            logger.warning("Had to remove some words: " + "\n".join(removed_words))

        return inp_formatted, openai_inp, tokenized_prompt, cached_image_chunks

    def _get_sampler_completion_args(self, comp_settings: CommonCompSettings):
        sampler_args = {
            "temperature": comp_settings.temperature,
            "top_k": comp_settings.top_k,
            "top_p": comp_settings.top_p,
            "min_p": comp_settings.min_p,
            "repeat_penalty": comp_settings.repeat_penalty,
            "frequency_penalty": comp_settings.frequency_penalty,
            "present_penalty": comp_settings.presence_penalty,
        }

        comp_args = {
            "max_tokens": comp_settings.max_tokens,
            "stop": comp_settings.stop_words,
            "seed": comp_settings.seed
        }

        completion_args = sampler_args | comp_args

        self.llm.set_seed(-1)
        if comp_settings.seed is not None:
            self.llm.set_seed(comp_settings.seed)

        sampler_args["temp"] = sampler_args["temperature"]
        # sampler has temp instead of temperature
        del sampler_args["temperature"]

        logit_bias_map = {}
        if comp_settings.disable_eos:
            logit_bias_map[self.llm.token_eos()] =  float('-inf')
        if comp_settings.disable_eot:
            logit_bias_map[self.llm.token_eot()] = float('-inf')
        if len(logit_bias_map) > 0:
            sampler_args["logit_bias"] = logit_bias_map

        return sampler_args, completion_args

    def _eval_helper(self, t):
        """
        Don't create checkpoints in local evals, useless overhead in dynamic generation.
        """
        self.llm.is_hybrid = False
        res = self.llm.eval(t)
        self.llm.is_hybrid = self.orig_is_hybrid
        return res

    def _prefix_helper_hybrid(self, tokens: List[int], min_prefix_len: int = 1024) -> Tuple[List[int], bool]:
        original_tokens = list(tokens)
        hit = False
        # Check for kv cache prefix match
        if self.use_prefix_caching and self.llm.n_tokens > 0:
            longest_prefix = self.llm.longest_token_prefix(self.llm._input_ids, tokens[:-1])
            if longest_prefix > min_prefix_len:
                reset = False

                if longest_prefix == len(tokens):
                    if self.llm.verbose:
                        print(f"Llama.generate: Full match. Forcing prefix-- to evaluate 1 token.", file=sys.stderr)
                    longest_prefix -= 1

                # Physically erase trailing "ghost" tokens from the C++ KV cache
                # to prevent attention misalignment in multi-round chats.
                if longest_prefix < self.llm.n_tokens:
                    if self.llm.is_hybrid and self.llm._hybrid_cache_mgr is not None:
                        if self.llm.verbose:
                            print(f"Llama.generate: Hybrid model rollback triggered.", file=sys.stderr)

                        best_ckpt = self.llm._hybrid_cache_mgr.find_best_checkpoint(original_tokens, 0)
                        if best_ckpt is not None and self.llm._hybrid_cache_mgr.restore_checkpoint(best_ckpt, seq_id=0):
                            actual_prefix = best_ckpt.pos
                            hit = True
                        else:
                            actual_prefix = 0
                            self.llm._hybrid_cache_mgr.clear()
                            self.llm._ctx.memory_clear(True)

                        self.llm.n_tokens = actual_prefix
                        tokens = original_tokens[actual_prefix:]
                        if self.llm.verbose:
                            print(
                                f"Llama.generate: {actual_prefix} prefix-match hit, "
                                f"remaining {len(tokens)} prompt tokens to eval",
                                file=sys.stderr,
                            )
                    else:
                        hit = True
                        if self.llm.verbose:
                            print(f"Llama.generate: Truncating KV cache size from {self.llm.n_tokens} to {longest_prefix}", file=sys.stderr)
                        self.llm._ctx.memory_seq_rm(0, longest_prefix, -1)

                        # Adjust the tokens array and cursor to reuse the matched cache
                        self.llm.n_tokens = longest_prefix
                        tokens = tokens[longest_prefix:]

                        if self.llm.verbose:
                            print(
                                f"Llama.generate: {longest_prefix} prefix-match hit, "
                                f"remaining {len(tokens)} prompt tokens to eval",
                                file=sys.stderr,
                            )
        if not hit:
            # No prefix matched. Completely clear the KV cache to prevent context poisoning.
            self.llm.n_tokens = 0
            self.llm._ctx.memory_clear(True)
            if self.llm.is_hybrid and self.llm._hybrid_cache_mgr is not None:
                self.llm._hybrid_cache_mgr.clear()

        return tokens, hit

    def _init_sampling_ctx_from_dict(self, args: Dict[str, Any]) -> LlamaSamplingContext:
        self._close_sampling_ctx()
        params = dataclass_from_dict(LlamaSamplingParams, args)
        return LlamaSamplingContext(params, self.llm._model)

    def _close_sampling_ctx(self):
        if self.llm._sampling_ctx is not None:
            self.llm._sampling_ctx.close()
            self.llm._sampling_ctx = None

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

        if discard_thinks:
            content = content.split("</think>")[-1]

        return content.strip()

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None,
                        discard_thinks: bool = True,
                        log_file_path: str | None = None
                        ) -> (str, List[BaseModel]):
        log_dict = {}
        try:
            self.load_model(preset)
            self.llm.is_hybrid = self.orig_is_hybrid
            # disable hybrid cache logic
            if not self.use_prefix_caching:
                self.llm._hybrid_cache_mgr = None

            if comp_settings is None:
                comp_settings = CommonCompSettings()
            comp_settings.fill_defaults(self.model_map, preset)

            if tools is None:
                tools = []

            addendum = ""
            if len(tools) > 0:
                comp_settings.tools_json += tools
                tool_calls = "\n".join([generate_pydantic_markdown_str(t) for t in tools])
                addendum = "\nSimplified JSON Schema:\n" + tool_calls

            log_dict["inp"] = inp

            # get tokens and formatted version of raw prompt
            inp_formatted, openai_inp, tokenized_prompt, image_chunks = self._tokenize_prompt(preset, inp, comp_settings, addendum)

            log_dict["inp_formatted"] = inp_formatted
            log_dict["openai_inp"] = openai_inp
            log_dict["tokenized_prompt"] = tokenized_prompt

            sampler_args, completion_args = self._get_sampler_completion_args(comp_settings)

            calls = []
            addendum = []

            # close sampler
            self._close_sampling_ctx()

            # preare ass to user string for reasoning / non reasoning
            user_to_ass_string = self.user_to_ass_string_raw
            wait_for_thinking = False
            if self.reasoning_model:
                if comp_settings.enable_thinking:
                    user_to_ass_string += REASON_THINK
                    wait_for_thinking = True
                else:
                    user_to_ass_string += REASON_NO_THINK

            content = None
            tools = comp_settings.tools_json
            if len(tools) == 0:
                if comp_settings.completion_callback is None and comp_settings.eval_only == False and len(image_chunks) == 0 and not comp_settings.duplex and not comp_settings.cache_prompt:
                    res = self.llm.create_completion(tokenized_prompt, **completion_args)
                    finish_reason = res["choices"][0]["finish_reason"]
                    content = res["choices"][0]["text"]
                else:
                    start_eval = time.time()
                    eval_cnt = 0
                    tokenized_prompt, prefix_hit = self._prefix_helper_hybrid(tokenized_prompt)

                    if len(tokenized_prompt) == 0:
                        raise ValueError("len(tokenized_prompt) == 0 after prefix handling; must leave 1 token for re-eval")

                    body_tokens = tokenized_prompt[:-1]
                    last_token = tokenized_prompt[-1:]

                    # replay prompt tail except last token, without checkpointing
                    if body_tokens:
                        self.llm.is_hybrid = False
                        self.llm.eval(body_tokens)

                    # create checkpoint at N-1 only for fresh prompt paths
                    if not prefix_hit and self.use_prefix_caching and self.llm._hybrid_cache_mgr is not None:
                        self.llm.is_hybrid = self.orig_is_hybrid
                        self.llm._hybrid_cache_mgr.save_checkpoint(
                            current_pos=self.llm.n_tokens,
                            tokens=self.llm.input_ids[:self.llm.n_tokens].tolist(),
                            seq_id=0
                        )

                    # evaluate final prompt token so logits are correct for first sample
                    self.llm.is_hybrid = False
                    self.llm.eval(last_token)

                    # log final prompt
                    log_dict["final_prompt"] = self._detokenize(self.llm.input_ids[:self.llm.n_tokens].tolist(), special=True)

                    self.llm.is_hybrid = self.orig_is_hybrid

                    eval_time = time.time() - start_eval
                    self.llm._sampling_ctx = self._init_sampling_ctx_from_dict(sampler_args)

                    # all tokens that are generated and not sent from the user queue
                    completion_tokens = []
                    stream_prev_text = ""

                    is_in_think_block = False
                    is_generating_tool_call = False
                    max_tokens = comp_settings.max_tokens or 4096

                    n_tokens = 0
                    sample_idx = n_tokens + len(tokenized_prompt) - 1

                    finish_reason = "length"

                    def eval_local(_tokens):
                        if len(_tokens) + self.llm.n_tokens > (self.current_ctx - CTX_LEN_SAFE_LENGTH):
                            raise ValueError(f"Out of context space: {self.llm.n_tokens} + {len(_tokens)} > {self.current_ctx}")

                        nonlocal sample_idx

                        sample_idx += len(_tokens)
                        for t in _tokens:
                            completion_tokens.append(t)
                        self._eval_helper(_tokens)

                    cur_functin_call_buffer = []
                    function_called = False
                    duplex_user_buffer = ""
                    is_waiting_for_user_input = False
                    in_duplex_mode_has_generation_mode_set = False

                    start_time = time.time()
                    sample_cnt = 0
                    sample_time = 0

                    class SchemaHelper:
                        def __init__(self):
                            self.current_target_bm = None
                            self.character_level_parser = None
                            self.apply_bias_func = None
                            self.constraint_start_idx = None

                        def clear(self):
                            self.current_target_bm = None
                            self.character_level_parser = None
                            self.apply_bias_func = None
                            self.constraint_start_idx = None

                    sh = SchemaHelper()
                    static_prefix_n = self.llm.n_tokens
                    while (not comp_settings.duplex and (self.llm.n_tokens - static_prefix_n) < max_tokens) or (comp_settings.duplex and self.llm.n_tokens < self.current_ctx - CTX_LEN_SAFE_LENGTH):
                        halt_signal = False
                        def switch_to_listening():
                            nonlocal is_waiting_for_user_input
                            nonlocal duplex_user_buffer

                            if not is_waiting_for_user_input:
                                duplex_user_buffer = ""
                                eval_local(self._tokenize(self.ass_to_user_string, special=True))

                            is_waiting_for_user_input = True

                        def switch_to_generating():
                            nonlocal is_waiting_for_user_input
                            nonlocal duplex_user_buffer
                            nonlocal completion_tokens
                            nonlocal stream_prev_text

                            if is_waiting_for_user_input:
                                duplex_user_buffer = ""
                                eval_local(self._tokenize(user_to_ass_string, special=True))

                            stream_prev_text = ""
                            completion_tokens.clear()
                            is_waiting_for_user_input = False

                        msg = None
                        if comp_settings.duplex:
                            try:
                                msg = comp_settings.queue_from_user.get_nowait()
                                if isinstance(msg, str):
                                    duplex_user_buffer += self._replace_names_input(msg)
                                elif isinstance(msg, DuplexAssistantInjectBegin):
                                    duplex_user_buffer += user_to_ass_string
                                elif isinstance(msg, DuplexAssistantInjectEnd):
                                    duplex_user_buffer += self.ass_to_user_string
                                elif isinstance(msg, DuplexSignalFinish):
                                    # eval everything in buffer
                                    tmp = self._tokenize(duplex_user_buffer + user_to_ass_string, special=True)
                                    eval_local(tmp)

                                    switch_to_generating()
                                    pass
                                elif isinstance(msg, DuplexSignalInterrupt):
                                    # switch to listening mode
                                    tmp = self._tokenize("-" + self.ass_to_user_string, special=True)
                                    eval_local(tmp)

                                    switch_to_listening()
                                elif isinstance(msg, DuplexStartGenerationText):
                                    in_duplex_mode_has_generation_mode_set = True

                                    if comp_settings.use_lm_format_enforcer:
                                        sh.clear()
                                    else:
                                        self.llm._sampling_ctx = self._init_sampling_ctx_from_dict(sampler_args)

                                    switch_to_generating()
                                elif isinstance(msg, DuplexStartGenerationTool):
                                    in_duplex_mode_has_generation_mode_set = True
                                    sampler_args_with_grammar = copy.copy(sampler_args)
                                    max_tokens += 4000

                                    switch_to_generating()
                                    if wait_for_thinking:
                                        thinking_tokens = []
                                        while True:
                                            token = self.llm.sample(idx=None)
                                            eval_local([token])
                                            thinking_tokens.append(token)
                                            self.llm._sampling_ctx.accept(token, False if self.llm._sampling_ctx.params.grammar == "" else True)
                                            thinking_output = self._detokenize(thinking_tokens, special=True)
                                            if "</think>" in thinking_output:
                                                break

                                    if comp_settings.use_lm_format_enforcer:
                                        sh.clear()
                                        sh.current_target_bm = msg.bm
                                        sh.constraint_start_idx = self.llm.n_tokens
                                    else:
                                        bm: Type[BaseModel] = msg.bm
                                        gbnf_grammar, _ = better_generate_gbnf_grammar_and_documentation([bm])
                                        grammar = LlamaGrammar(_grammar=gbnf_grammar)
                                        sampler_args_with_grammar["grammar"] = grammar
                                        self.llm._sampling_ctx = self._init_sampling_ctx_from_dict(sampler_args_with_grammar)

                                    comp_settings.queue_to_user.put(DuplexJsonBegin())
                                elif isinstance(msg, DuplexSignalTerminate):
                                    halt_signal = True
                            except queue.Empty:
                                pass

                        if halt_signal:
                            break

                        if is_waiting_for_user_input:
                            if duplex_user_buffer is None or duplex_user_buffer == "":
                                time.sleep(0.1)
                                continue

                            tokens = self._tokenize(duplex_user_buffer, special=False)
                            eval_local(tokens)

                            m = re.search(r'[.?!]', duplex_user_buffer)
                            if m and comp_settings.allow_interrupts:
                                # simple single token, check if llm wants to switch turn and use that as interrupt
                                token = self.llm.sample(idx=None)
                                self.llm._sampling_ctx.accept(token, False)
                                sample_cnt += 1
                                as_non_special = self._detokenize([token], special=False).strip()
                                if as_non_special == "":
                                    switch_to_generating()

                            duplex_user_buffer = ""
                            continue

                        if comp_settings.duplex and comp_settings.wait_for_start_signal and not in_duplex_mode_has_generation_mode_set:
                            time.sleep(0.1)
                            continue

                        """
                        SAMPLE
                        """
                        start_sample = time.time()
                        if comp_settings.use_lm_format_enforcer and sh.current_target_bm:
                            # with constrained output
                            if not sh.character_level_parser:
                                sh.character_level_parser = JsonSchemaParser(json.loads(generate_pydantic_json_schema_str(sh.current_target_bm)))

                            if not sh.apply_bias_func:
                                sh.apply_bias_func = build_llamacpp_logits_processor(self.tokenizer_data, sh.character_level_parser, analyze=False)

                            logits_tmp = np.ctypeslib.as_array(self.llm._ctx.get_logits(), (self.llm.n_vocab(),))
                            logits = logits_tmp.copy()
                            del logits_tmp

                            constrained_prefix = self.llm.input_ids[sh.constraint_start_idx:self.llm.n_tokens]
                            sh.apply_bias_func(constrained_prefix, logits)

                            token = _sample_from_logits(logits, temperature=comp_settings.temperature, top_k=comp_settings.top_k, top_p=comp_settings.top_p, fast=False)
                            self.llm._sampling_ctx.accept(token, False if self.llm._sampling_ctx.params.grammar == "" else True)
                        else:
                            # set sampler idx (if we use it, default is None -> -1 internally)
                            idx = None
                            token = self.llm.sample(idx=idx)
                            self.llm._sampling_ctx.accept(token, False if self.llm._sampling_ctx.params.grammar == "" else True )

                        sample_time += time.time() - start_sample
                        sample_cnt += 1
                        sampled_tokens = [token]

                        """
                        STOP REASON CHECK
                        """
                        if (llama_cpp.llama_token_is_eog(self.llm._model.vocab, token) and comp_settings.stop_on_eot) and (not prefix_hit or (prefix_hit and sample_cnt > 8)):
                            if comp_settings.duplex:
                                comp_settings.queue_to_user.put(DuplexSignalEog())
                                switch_to_listening()
                                continue
                            else:
                                finish_reason = "stop"
                                break
                        if comp_settings.stop_words and len(comp_settings.stop_words) > 0:
                            if isinstance(comp_settings.stop_words, list):
                                stop_words = comp_settings.stop_words
                            else:
                                stop_words = [comp_settings.stop_words]

                            cur_completion_as_text = self._detokenize(completion_tokens, special=False)
                            found = False
                            for sw in stop_words:
                                if sw in cur_completion_as_text:
                                    found = True
                                    break
                            if found:
                                finish_reason = "sw"
                                break

                        """
                        CALLBACK REACTOR
                        """
                        # get the full text from actually generated tokens
                        stream_full_text = self._detokenize(completion_tokens + sampled_tokens, special=False)

                        # get delta from last completion tokens string state
                        # this is so that we capture the correct string representation of multi-token characters
                        if stream_full_text.startswith(stream_prev_text):
                            stream_delta = stream_full_text[len(stream_prev_text):]
                        else:
                            stream_delta = stream_full_text

                        # set prev state
                        stream_prev_text = stream_full_text

                        # check if there is a completion_callback delegate
                        # this calls a python function from somehwere the either returns the generated text 1:1 or modifies it
                        if comp_settings.completion_callback is not None:
                            new_delta = comp_settings.completion_callback(stream_delta)
                            if new_delta is None:
                                finish_reason = "sw"
                                break
                        else:
                            new_delta = stream_delta

                        # check if the callback data is different, replace current token with injected content
                        if new_delta != stream_delta:
                            sampled_tokens = self._tokenize(new_delta, special=False)

                        """
                        FINAL EVAL
                        """
                        eval_local(sampled_tokens)
                        if comp_settings.queue_to_user is not None:
                            if comp_settings.completion_callback is not None:
                                raise ValueError("CommonCompletionSettings: completion_callback and queue_to_user not allowed at the same time!")
                            comp_settings.queue_to_user.put(new_delta)

                    duration = time.time() - start_time
                    if duration < 0.05:
                        duration = 0.05
                    t_s = sample_cnt / duration
                    addendum.append(f"PREFILL TIME: {eval_time}")
                    addendum.append(f"PREFILL T/S : {eval_time / (eval_cnt + 1)}")
                    addendum.append(f"GEN DURATION: {duration}")
                    addendum.append(f"GEN SAMPLE S: {sample_time}")
                    addendum.append(f"GEN TOKEN/S : {t_s}")

                    prefix_delta = self.llm.input_ids[static_prefix_n:self.llm.n_tokens].tolist()
                    full_output = self._detokenize(prefix_delta, special=True)
                    content = self._detokenize(prefix_delta, special=False)
            else:
                gbnf_grammar, _ = better_generate_gbnf_grammar_and_documentation(tools)
                grammar = LlamaGrammar(_grammar=gbnf_grammar)
                cnt = 0
                while True:
                    try:
                        res = self.llm.create_chat_completion_openai_v1(openai_inp, grammar=grammar, **completion_args)
                        content = res.choices[0].message.content
                        finish_reason = res.choices[0].finish_reason
                        good_json_string = repair_json(content)
                        calls.append(tools[0].model_validate_json(good_json_string))
                        inp_formatted.append(("assistant", content))
                        break
                    except Exception as e:
                        msg = f"Error when creating basemodel {e}\nBM: {tools[0].__class__.__name__}\nJSON: {content}"
                        logger.info(msg)
                        cnt += 1
                        if cnt > 10:
                            crash_hard(msg)

            if comp_settings.duplex:
                comp_settings.queue_to_user.put(DuplexSignalEog())
                comp_settings.queue_to_user.put(DuplexSignalFinished())

            if comp_settings.stop_words and len(comp_settings.stop_words) > 0:
                if isinstance(comp_settings.stop_words, list):
                    stop_words = comp_settings.stop_words
                else:
                    stop_words = [comp_settings.stop_words]
                for sw in stop_words:
                    content = content.replace(sw, "")

            content = content.strip()
            log_dict["content"] = content
            return content, calls
        finally:
            try:
                log_dict["everything"] = self._detokenize(self.llm.input_ids[:self.llm.n_tokens].tolist(), special=True)
            except:
                pass
            try:
                file_path = log_file_path + f".completion.log"
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(json.dumps(log_dict, indent=4))
            except:
                pass


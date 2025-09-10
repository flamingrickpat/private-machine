import asyncio
import contextlib
import copy
import difflib
import gc
import json
import logging
import os
import queue
import random
import re
import string
import textwrap
import base64
import ctypes
from typing import (
    Union,
    Tuple,
)
from typing import Type
from typing import Dict, Any
import time
from typing import List

import numpy.typing as npt
import llama_cpp.llama_cpp as llama_cpp
import numpy as np
import psutil
from fastmcp import Client
from json_repair import repair_json
from llama_cpp import Llama, LlamaGrammar, suppress_stdout_stderr
#from lmformatenforcer import JsonSchemaParser
#from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from pydantic import BaseModel

from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_manager import LlmManager
from pm.tools.tools_common import create_tool_router_model
from pm.utils.string_utils import remove_n_words, parse_json_from_response
from pm.utils.duplex_utils import DuplexSignalFinish, DuplexSignalInterrupt
from pm.utils.gbnf_utils import better_generate_gbnf_grammar_and_documentation, fix_gbnf_grammar_generator
fix_gbnf_grammar_generator()

logger = logging.getLogger(__name__)

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
                wrapped_message = textwrap.fill(message.replace("\n", "µ"), width=max_width).replace("µ", "\n")
                file.write(wrapped_message + "\n\n")

                # Add a separator between turns
                file.write(f"{'-' * max_width}\n\n")


universal_image_begin = "<begin_image>"
universal_image_end = "<end_image>"
magic_image_token = -1
physical_cores = psutil.cpu_count(logical=False)

class LlmManagerLLama(LlmManager):
    def __init__(self, model_map: Dict[str, Any], test_mode: bool=False):
        super().__init__(model_map, test_mode)
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
        self.user_to_ass_string = None
        self.ass_to_user_string = None

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
        else:
            raise Exception("Unknown model!")

        if self.model_path != model_path:
            time.sleep(1)
            gc.collect()
            time.sleep(1)

            with suppress_stdout_stderr(disable=False):
                self.llm = Llama(model_path=model_path, n_gpu_layers=layers, n_ctx=ctx, verbose=False, last_n_tokens_size=last_n_tokens_size, flash_attn=True, n_threads=physical_cores)
                self.eos_token_id = self.llm.metadata["tokenizer.ggml.eos_token_id"]

                # self.llm.cache = LlamaDiskCache(capacity_bytes=int(42126278829))
                self.current_ctx = ctx
                self.model_path = model_path
                from jinja2 import Template, StrictUndefined
                self.chat_template = Template(self.llm.metadata["tokenizer.chat_template"], undefined=StrictUndefined) #Environment(loader=BaseLoader).from_string(self.llm.metadata["tokenizer.chat_template"])

            if mmproj_file is not None and mmproj_file != "":
                self.clip_model_path = mmproj_file
                self._init_mtmd_context(self.llm)


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

    def _tokenize_prompt(self,
                         preset: LlmPreset,
                         inp: List[Tuple[str, str]],
                         comp_settings: CommonCompSettings,
                         sysprompt_addendum: str | None = None
                        ) -> (str, List[BaseModel]):
        # we sometimes seed the assistant response to get rid of crap like "ok, lets tackle this, here is a bunch of useless words and now here is what you wanted :) <result>"
        messages = [
            {"role": "system", "content": "test1"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test2"},
                    {"type": "type", "image_url": {"url": "test3", "image_url": "test3", "data_url": "test3"}, "image": {"url": "test3", "image_url": "test3", "data_url": "test3"}},
                    {"type": "text", "text": "test4"},
                ]
            },
            {"role": "assistant", "content": "test5"},
            {"role": "user", "content": "test6"},
            {"role": "assistant", "content": "test7"}
        ]

        def remove_strings(text, str1, str2):
            pattern = re.escape(str1) + r"\s*" + re.escape(str2)
            return re.sub(pattern, "", text)

        def between(s: str, a: str, b: str) -> str:
            """Return the substring strictly between the first (and only) a and b."""
            start = s.index(a) + len(a)
            end = s.index(b, start)
            return s[start:end]

        def replace_tagged_blocks(
                text: str,
                in_begin: str,
                in_end: str,
                out_start: str,
                out_media: str,
                out_end: str = ""
        ) -> Tuple[str, List[str]]:
            """
            Find every block between in_begin and in_end, collect the inner text,
            and replace each block with out_start + out_media + out_end.

            Returns (new_text, extracted_list).
            """
            pattern = re.compile(re.escape(in_begin) + r"(.*?)" + re.escape(in_end), re.DOTALL)
            extracted: List[str] = []

            def _repl(m: re.Match) -> str:
                extracted.append(m.group(1))
                return f"{out_start}{out_media}{out_end}"

            new_text = pattern.sub(_repl, text)
            return new_text, extracted

        try:
            text = self.chat_template.render(
                messages=messages,
                add_generation_prompt=True,
                eos_token=self.llm.detokenize([self.llm.token_eos()]),
                bos_token=self.llm.detokenize([self.llm.token_bos()]),
            )
        except:
            messages = messages[1:] # remove sys prompt in case its not allowed for model
            text = self.chat_template.render(
                messages=messages,
                add_generation_prompt=True,
                eos_token=self.llm.detokenize([self.llm.token_eos()]),
                bos_token=self.llm.detokenize([self.llm.token_bos()]),
            )

        remove_ass_string = text.split("test7")[1]
        self.user_to_ass_string = between(text, "test6", "test7")
        self.ass_to_user_string = between(text, "test5", "test6")
        if "test3" in text:
            text_to_image_string = between(text, "test2", "test3")
            image_to_text_string = between(text, "test3", "test4")
        else:
            text_to_image_string = between(text, "test2", "test4")
            image_to_text_string = ""

        media_marker_string = self._mtmd_cpp.mtmd_default_marker().decode('utf-8')

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

        # convert format
        openai_inp = []
        for msg in inp_formatted:
            openai_inp.append({"role": msg[0], "content": msg[1]})

        cached_image_chunks = []
        begin_image_token = -1

        # remove messages until it fits in context
        rendered_data = None
        trimmed = False
        while True:
            try:
                def raise_exception(msg: str):
                    print(msg)

                rendered_data = self.chat_template.render(
                    messages=openai_inp,
                    add_generation_prompt=True,
                    eos_token=self.llm.detokenize([self.llm.token_eos()]),
                    bos_token=self.llm.detokenize([self.llm.token_bos()]),
                    raise_exception=raise_exception
                )
            except Exception as e:
                print(f"Error when trimming prompt: {e}\n{openai_inp}")

            # strip each line to save tokens; lots of unnecessary whitespaces from indented block strings
            rendered_data = "\n".join([x.strip() for x in rendered_data.splitlines()])

            # let the assistant continue if they talked last
            if openai_inp[-1]["role"] == "assistant":
                rendered_data = rendered_data.strip().removesuffix(remove_ass_string.strip())

            # remove thinkign tags
            rendered_data = remove_strings(rendered_data, "</think>", "<think>")

            # replace visual input
            rendered_data, images = replace_tagged_blocks(rendered_data,
                                              in_begin=universal_image_begin,
                                              in_end=universal_image_end,
                                              out_start=text_to_image_string,
                                              out_media=media_marker_string,
                                              out_end=image_to_text_string)

            for image in images:
                img_bytes = self._load_image(image)
                bitmap_llama = self._create_bitmap_from_bytes(img_bytes)
                bitmaps = [bitmap_llama]

                image_prompt_text = f"test1{image_to_text_string}{media_marker_string}{text_to_image_string}test2"

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
                    f"{text_to_image_string}{media_marker_string}{image_to_text_string}",
                    f"{real_text_to_image_string}{real_image_to_text_string}",
                )

            tokenized_prompt = self.llm.tokenize(
                rendered_data.encode("utf-8"),
                add_bos=True,
                special=True,
            )

            # insert magic token so we know when to eval chunk
            if begin_image_token != -1:
                tokenized_prompt = [y for x in tokenized_prompt for y in ([x, magic_image_token] if x == begin_image_token else [x])]

            # start deleting in middle
            if (self.current_ctx - comp_settings.max_tokens) - 32 < 1:
                raise Exception("Negative available tokens, check context and max_tokens!")
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

        return inp_formatted, openai_inp, tokenized_prompt, cached_image_chunks

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
        comp_settings.fill_defaults(preset)

        if tools is None:
            tools = []

        addendum = ""
        if len(tools) > 0:
            comp_settings.tools_json += tools
            tool_calls = "\n".join([json.dumps(t.model_json_schema()) for t in tools])
            addendum = "\nTool Documentation JSON Schema:\n" + tool_calls

        # get tokens and formatted version of raw prompt
        inp_formatted, openai_inp, tokenized_prompt, image_chunks = self._tokenize_prompt(preset, inp, comp_settings, addendum)

        # log final prompt
        if log_file_path is not None:
            log_conversation(inp_formatted, log_file_path, 200)

        sampler_args = {
            "temperature": comp_settings.temperature,
            "top_k": comp_settings.top_k,
            "top_p": comp_settings.top_p,
            "min_p": comp_settings.min_p,
            "repeat_penalty": comp_settings.repeat_penalty,
            "frequency_penalty": comp_settings.frequency_penalty,
            "presence_penalty": comp_settings.presence_penalty,
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

        if comp_settings.disable_eos:
            logit_bias_map = {self.llm.token_eos(): -1000000.0}
            sampler_args["logit_bias"] = logit_bias_map

        calls = []

        content = None
        tools = comp_settings.tools_json
        if len(tools) == 0:
            if comp_settings.completion_callback is None and comp_settings.eval_only == False and len(image_chunks) == 0 and not comp_settings.duplex:
                res = self.llm.create_completion(tokenized_prompt, **completion_args)
                finish_reason = res["choices"][0]["finish_reason"]
                content = res["choices"][0]["text"]
                if discard_thinks:
                    content = content.split("</think>")[-1]
            else:
                evaluated = False
                if self.state is not None and not comp_settings.eval_only:
                    def check_diff(a, b):
                        if len(a) > len(b):
                            return None
                        if b[:len(a)] == a:
                            return b[len(a):]
                        return None

                    diff = check_diff(self.state_tokens, tokenized_prompt)
                    if diff is not None:
                        self.llm.load_state(self.state)
                        if len(diff) == 0:
                            self.llm.eval([tokenized_prompt[-1]])
                        self.llm.eval(diff)
                        self.llm.input_ids = copy.deepcopy(self.state_input_ids)
                        self.llm.scores = copy.deepcopy(self.state_scores)

                        evaluated = True

                def split_by_value(lst, sep):
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

                use_sample_index = False
                if not evaluated:
                    self.state = None
                    self.state_tokens = None

                    self.llm.reset()

                    sep_by_image = split_by_value(tokenized_prompt, magic_image_token)
                    for sublist in sep_by_image:
                        if len(sublist) == 1 and sublist[0] == magic_image_token:
                            use_sample_index = False
                            chunks, chunk = image_chunks[0]
                            del image_chunks[0]

                            chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)

                            if self.llm.n_tokens + chunk_n_tokens > self.llm.n_ctx():
                                raise ValueError(
                                    f"Prompt exceeds n_ctx: {self.llm.n_tokens + chunk_n_tokens} > {self.llm.n_ctx()}"
                                )

                            new_n_past = llama_cpp.llama_pos(0)
                            result = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                                self.mtmd_ctx,
                                self.llm._ctx.ctx,
                                chunk,
                                llama_cpp.llama_pos(self.llm.n_tokens),
                                llama_cpp.llama_seq_id(0),
                                self.llm.n_batch,
                                False,  # logits_last
                                ctypes.byref(new_n_past)
                            )

                            if result != 0:
                                raise ValueError(f"Failed to evaluate chunk: error code {result}")

                            # Update llama's token count
                            self.llm.n_tokens = new_n_past.value

                            # free
                            self._mtmd_cpp.mtmd_input_chunks_free(chunks)
                        else:
                            self.llm.eval(sublist)

                if magic_image_token in tokenized_prompt:
                    tokenized_prompt.remove(magic_image_token)

                if comp_settings.eval_only:
                    self.state = self.llm.save_state()
                    self.state_tokens = tokenized_prompt
                    self.state_input_ids = copy.deepcopy(self.llm.input_ids)
                    self.state_scores = copy.deepcopy(self.llm.scores)
                    return "", []

                self.llm._sampler = self.llm._init_sampler(**sampler_args)

                completion_tokens = []
                all_tokens = []
                for t in tokenized_prompt:
                    all_tokens.append(t)

                is_in_think_block = False
                is_generating_tool_call = False
                max_tokens = comp_settings.max_tokens or 4096

                n_tokens = 0
                sample_idx = n_tokens + len(tokenized_prompt) - 1
                finish_reason = "length"

                def eval_local(tokens):
                    nonlocal sample_idx
                    sample_idx += len(tokens)
                    for t in tokens:
                        completion_tokens.append(t)
                        all_tokens.append(t)
                    self.llm.eval(tokens)

                bad_tokens = []
                cur_functin_call_buffer = []
                function_called = False
                duplex_user_buffer = ""
                is_waiting_for_user_input = False

                while len(completion_tokens) < max_tokens:
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

                        if is_waiting_for_user_input:
                            duplex_user_buffer = ""
                            eval_local(self._tokenize(self.user_to_ass_string, special=True))

                        is_waiting_for_user_input = False

                    msg = None
                    if comp_settings.duplex:
                        try:
                            msg = comp_settings.queue_from_user.get_nowait()
                            if isinstance(msg, str):
                                duplex_user_buffer += msg
                            elif isinstance(msg, DuplexSignalFinish):
                                # eval everything in buffer
                                tmp = self._tokenize(duplex_user_buffer + self.user_to_ass_string, special=True)
                                eval_local(tmp)

                                switch_to_generating()
                                pass
                            elif isinstance(msg, DuplexSignalInterrupt):
                                # switch to listening mode
                                tmp = self._tokenize("-" + self.ass_to_user_string, special=True)
                                eval_local(tmp)

                                switch_to_listening()
                        except queue.Empty:
                            pass

                    def longest_common_substring(a: str, b: str) -> str:
                        matcher = difflib.SequenceMatcher(None, a, b)
                        match = matcher.find_longest_match(0, len(a), 0, len(b))
                        return a[match.a: match.a + match.size]

                    if is_waiting_for_user_input:
                        tokens = self._tokenize(duplex_user_buffer, special=False)
                        eval_local(tokens)

                        m = re.search(r'[.?!]', duplex_user_buffer)
                        if m:
                            # simple single token, check if llm wants to switch turn and use that as interrupt
                            token = self.llm.sample(idx=None, **sampler_args)
                            as_non_special = self._detokenize([token], special=False).strip()
                            if as_non_special == "":
                                switch_to_generating()

                        duplex_user_buffer = ""
                        continue

                    # set sampler idx (if we use it, default is None -> -1 internally)
                    idx = None
                    if use_sample_index:
                        idx = sample_idx

                    token = self.llm.sample(idx=idx, **sampler_args)
                    cur_tokens_to_eval = [token]
                    cur_token_as_text = self._detokenize([token], special=False)
                    cur_completion_as_text = self._detokenize(completion_tokens, special=False)

                    # check for exit
                    if llama_cpp.llama_token_is_eog(self.llm._model.vocab, token):
                        if comp_settings.duplex:
                            switch_to_listening()
                            continue
                        else:
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

                    if comp_settings.completion_callback is not None:
                        callback_eval_text = comp_settings.completion_callback(cur_token_as_text)
                        if callback_eval_text is None:
                            finish_reason = "sw"
                            break
                    else:
                        callback_eval_text = cur_token_as_text

                    if comp_settings.queue_to_user is not None:
                        comp_settings.queue_to_user.put(callback_eval_text)

                    if cur_token_as_text == callback_eval_text:
                        eval_local(cur_tokens_to_eval)
                    else:
                        callback_tokens = self._tokenize(callback_eval_text, special=False)
                        eval_local(callback_tokens)

                full_output = self._detokenize(completion_tokens, special=True)
                content = self._detokenize(completion_tokens, special=False)
        else:
            gbnf_grammar, documentation = better_generate_gbnf_grammar_and_documentation(tools)
            grammar = LlamaGrammar(_grammar=gbnf_grammar)
            while True:
                try:
                    res = self.llm.create_chat_completion_openai_v1(openai_inp, grammar=grammar, **comp_args)
                    content = res.choices[0].message.content
                    finish_reason = res.choices[0].finish_reason
                    good_json_string = repair_json(content)
                    calls.append(tools[0].model_validate_json(good_json_string))
                    break
                except Exception as e:
                    logger.info(f"Error when creating basemodel {e}\nBM: {tools[0].__class__.__name__}\nJSON: {content}")

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

        comp_settings.fill_defaults(self.model_map, preset)

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
                        if len(bad_tokens) > 0 and False:
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
                                try:
                                    result = loop.run_until_complete(client.call_tool(tool_name, tool_args))
                                    if len(result) > 0:
                                        result_text = dict(result[0])
                                    else:
                                        result_text = "Successful; No items returned."
                                except Exception as e:
                                    result_text = f"Error when calling MCP-Server: {e}"

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

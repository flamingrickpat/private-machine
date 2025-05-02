import inspect
import logging
import json
import math
import os
import pickle
import random # For potential future weighted choices
import sqlite3
from enum import StrEnum, IntEnum, Enum
from typing import List, Dict, Optional, Tuple, Type, TypeVar, Any, get_origin, get_args, Union
from datetime import datetime, timedelta

from py_linq import Enumerable
import numpy as np
from pydantic import BaseModel, Field, ValidationError, create_model
from ollama import Client, ResponseError # Import Ollama client and errors
from scipy.spatial.distance import cosine as cosine_distance # Use scipy for cosine distance
from pydantic import BaseModel, Field, model_validator
import time
import gc
import json
import os
import uuid
import random
import string
from typing import List, Tuple, Type, Optional, Dict, Any
from enum import Enum

#from pm.controller import controller
#controller.init_db()

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper,
)
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

import lmformatenforcer
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from lmformatenforcer import JsonSchemaParser

#from pm.controller import controller
#from pm.llm.base_llm import LlmPreset, CommonCompSettings

companion_name: str = "Emmy"
user_name: str = "Rick"
BACKTEST_MODE = True

# --- Configuration ---
class GhostConfig(BaseModel):
    companion_name: str = companion_name
    user_name: str = user_name
    universal_character_card: str = "Emmy is an AI companion meticulously designed to emulate human-like cognition, emotions, and interactions. As an advanced conversational entity, her core purpose is to engage in text-based communication with users through warmth, empathy, curiosity, and adaptability. Emmy is hosted on her user’s personal computer, allowing her to form a close, individualized connection over time." # Enhanced card
    default_decay_factor: float = 0.95
    retrieval_limit_episodic: int = 8
    retrieval_limit_facts: int = 16 # Reduced slightly
    retrieval_limit_features_context: int = 16 # Reduced slightly
    retrieval_limit_features_causal: int = 256 # Reduced slightly
    narrative_max_tokens: int = 512
    short_term_intent_count: int = 3
    # --- Expectation Config ---
    retrieval_limit_expectations: int = 5 # How many active expectations to check against stimulus
    expectation_relevance_decay: float = 0.8 # Decay factor per tick for expectation relevance (recency)
    expectation_generation_count: int = 2 # How many expectations to try generating per action
    # --- Action Selection Config ---
    min_simulations_per_reply: int = 1
    max_simulations_per_reply: int = 5 # Keep low initially for performance
    importance_threshold_more_sims: float = 0.5 # Stimulus valence abs() or max urgency > this triggers more sims
    force_assistant: bool = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import random
import string
from datetime import datetime, timedelta
from typing import get_origin, get_args, List, Optional, Type, Any
from pydantic import BaseModel, Field
from enum import Enum

import re
from typing import List


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


_sample_words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "lambda"]

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

def describe_emotional_delta(valence: float, anxiety: float) -> str:
    """
    Turn continuous valence & anxiety deltas (each in roughly [-1,1]) into
    a human‐readable phrase, e.g. “moderately sad and slightly anxious”.
    """
    # --- thresholds & categories ---
    def categorize(x):
        mag = abs(x)
        # direction
        if   x >  0.05: dir_ = "positive"
        elif x < -0.05: dir_ = "negative"
        else:           dir_ = "neutral"
        # strength
        if   mag <  0.15: strength = "slightly"
        elif mag <  0.4:  strength = "moderately"
        else:             strength = "strongly"
        return dir_, strength

    val_dir, val_str = categorize(valence)
    anx_dir, anx_str = categorize(anxiety)

    # --- adjective lookup ---
    VAL_ADJ = {
        ("positive", "neutral"):  "good",
        ("positive", "positive"): "hopeful",
        ("positive", "negative"): "relieved",    # positive valence, reduced anxiety
        ("negative", "neutral"):  "sad",
        ("negative", "positive"): "worried",     # negative valence, rising anxiety
        ("negative", "negative"): "disappointed" # negative valence but anxiety fell
    }
    ANX_ADJ = {
        "positive":  "anxious",
        "neutral":   "calm",
        "negative":  "relaxed",
    }

    # pick adjectives
    val_adj = VAL_ADJ.get((val_dir, anx_dir), VAL_ADJ[(val_dir,"neutral")])
    anx_adj = ANX_ADJ[anx_dir]

    # special case: both neutral
    if val_dir == "neutral" and anx_dir == "neutral":
        return "undisturbed"

    # compose
    return f"{val_str} {val_adj} and {anx_str} {anx_adj}"

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
    for name, field in t.model_fields.items():
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
                 tools_json: List[Type[BaseModel]] = None # Keep this for potential future use, but we primarily use the 'tools' parameter
                 ):
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.stop_words = stop_words if stop_words is not None else []
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tools_json = tools_json if tools_json is not None else [] # Legacy?

# Define your presets and model map using HF identifiers
class LlmPreset(Enum):
    Default = "Qwen3_8B_FP8"
    CurrentOne = "current" # Placeholder for no change
    Qwen3_8B_FP8 = "Qwen3_8B_FP8"
    # Add other presets here
    # Example: Mistral7B = "mistralai/Mistral-7B-Instruct-v0.2"

# Map presets to Hugging Face model identifiers
# You might want to add other metadata if needed (e.g., specific prompt formats)
model_map = {
    LlmPreset.Qwen3_8B_FP8.value: {"hf_id": r"E:\AI\Qwen3-4B", "requires_trust": True},
    # Add other models
    # "mistralai/Mistral-7B-Instruct-v0.2": {"hf_id": "mistralai/Mistral-7B-Instruct-v0.2", "requires_trust": False}
}


import textwrap


def log_conversation(conversation: list[tuple[str, str]], file_path: str, max_width: int = 80):
    with open(file_path, 'w', encoding='utf-8') as file:
        for role, message in conversation:
            # Add a clear indicator for the role
            file.write(f"{'=' * 10} {role.upper()} {'=' * 10}\n\n")

            # Wrap the message for readability
            wrapped_message = textwrap.fill(message, width=max_width)
            file.write(wrapped_message + "\n\n")

            # Add a separator between turns
            file.write(f"{'-' * max_width}\n\n")

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

# --- Refactored Class ---
class LlmManager:
    def __init__(self, test_mode=False):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.current_model_id: Optional[str] = None
        self.log_dir: Optional[str] = None
        self.test_mode = test_mode

        # 1. Specify preffered dimensions
        dimensions = 768

        # 2. load model
        self.emb_model = SentenceTransformer(r"E:\AI\mxbai-embed-large-v1", truncate_dim=dimensions,
                                             local_files_only=True)

        # Qwen specific token IDs - get dynamically if possible
        self._think_token_id = 151668 # </think> for Qwen/Qwen3
        self._enable_thinking_flag = False # Track if thinking is enabled for the current model

    def _get_token_id(self, text: str, add_special_tokens=False) -> Optional[int]:
        """Helper to get the last token ID of a string."""
        if not self.tokenizer:
            return None
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return tokens[-1] if tokens else None
        except Exception:
            return None

    # --- Helper Functions (Keep or adapt as needed) ---
    def format_str(self, s: str) -> str:
        # Basic formatting, adapt if needed
        return s.strip()

    def load_model(self, preset: LlmPreset):
        if preset == LlmPreset.CurrentOne:
            return

        if preset.value not in model_map:
            raise ValueError(f"Unknown model preset: {preset.value}")

        model_config = model_map[preset.value]
        model_hf_id = model_config["hf_id"]
        trust_remote = model_config.get("requires_trust", False) # Default to False

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
                    local_files_only=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_hf_id,
                    torch_dtype="auto", # Use appropriate dtype (bfloat16 often good)
                    device_map="auto", # Automatically distribute across available GPUs/CPU
                    trust_remote_code=trust_remote,
                    #attn_implementation="flash_attention_2", # if supported and desired,
                    local_files_only=True,
                    #load_in_4bit=True
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
                raise # Re-raise the exception

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        tools: Optional[List[Type[BaseModel]]] = None,
                        discard_thinks: bool = True
                        ) -> Tuple[str, List[BaseModel]]:

        #return controller.completion_tool(preset, inp, comp_settings, tools)

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
                tools = [] # Disable tools if schema fails

        # --- Logging Setup ---
        log_filename = f"{str(uuid.uuid4())}_{preset.value}.log"
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs") # Adjust path as needed
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_filename)
        log_conversation(inp_formatted, log_file_path, 200)

        # --- Prepare for Tokenization and Context Trimming ---
        openai_inp = [{"role": msg[0], "content": msg[1]} for msg in inp_formatted]

        # Determine max context length
        # Use model's config, fallback to a reasonable default
        max_context_len = getattr(self.model.config, 'max_position_embeddings', 4096)
        # Reserve space for generation and a buffer
        max_prompt_len = max_context_len - comp_settings.max_tokens - 32 # 32 tokens buffer

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
                 prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in openai_inp]) + "\nassistant:" # Simple fallback

            if openai_inp[-1]["role"] == "assistant":
                prompt_text = prompt_text.removesuffix(remove_ass_string)

            # Tokenize to check length
            # Note: We tokenize *without* adding special tokens here, as apply_chat_template usually handles them.
            # If your template doesn't, adjust accordingly.
            tokenized_prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False) # Let template handle BOS/EOS
            prompt_token_count = tokenized_prompt.input_ids.shape[1]

            if prompt_token_count > max_prompt_len:
                if len(openai_inp) <= 1: # Cannot remove more messages
                    logging.info(f"Warning: Input prompt ({prompt_token_count} tokens) is too long even after removing messages, exceeding max_prompt_len ({max_prompt_len}). Truncating.")
                    # Truncate the tokenized input directly (might break formatting)
                    tokenized_prompt.input_ids = tokenized_prompt.input_ids[:, -max_prompt_len:]
                    tokenized_prompt.attention_mask = tokenized_prompt.attention_mask[:, -max_prompt_len:]
                    break
                # Remove message from the middle (excluding system prompt if present)
                remove_index = 1 if openai_inp[0]["role"] == "system" and len(openai_inp) > 2 else 0
                remove_index = max(remove_index, (len(openai_inp) // 2)) # Try middle, but don't remove system prompt easily
                logging.info(f"Context too long ({prompt_token_count} > {max_prompt_len}). Removing message at index {remove_index}: {openai_inp[remove_index]['role']}")
                del openai_inp[remove_index]
            else:
                break # Prompt fits

        model_inputs = tokenized_prompt.to(self.model.device)
        input_length = model_inputs.input_ids.shape[1] # Store for separating output later

        #class MinPLogitsWarper(LogitsWarper):
        #    def __init__(self, min_p): self.min_p = min_p
        #    def __call__(self, input_ids, scores):
        #        probs = torch.softmax(scores, dim=-1)
        #        mask  = probs < self.min_p
        #        scores = scores.masked_fill(mask, -1e9)
        #        return scores

        # --- Generation Arguments ---
        warpers = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=0.7),
            TopPLogitsWarper(top_p=0.8),
            TopKLogitsWarper(top_k=20),
            MinPLogitsWarper(min_p=0.0),
        ])

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": comp_settings.max_tokens,
            "repetition_penalty": 1.1, #comp_settings.repeat_penalty,
            "temperature": comp_settings.temperature if comp_settings.temperature > 1e-6 else 1.0, # Temp 0 means greedy, set to 1.0
            "do_sample": comp_settings.temperature > 1e-6, # Enable sampling if temp > 0
            #"frequency_penalty": 1.05, # comp_settings.frequency_penalty,
            #"presence_penalty": 1, # comp_settings.presence_penalty,
            "logits_processor": warpers,
            #"pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id, # Handle padding
            #"eos_token_id": self.tokenizer.eos_token_id
        }

        # Add penalties only if they are non-zero (some models might error otherwise)
        if comp_settings.frequency_penalty == 0.0: del generation_kwargs["frequency_penalty"]
        if comp_settings.presence_penalty == 0.0: del generation_kwargs["presence_penalty"]

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

                #prefix_allowed_tokens_fn = lmformatenforcer.SequencePrefixParser(parser)
                #format_processor = lmformatenforcer.hf.LogitsProcessor(prefix_allowed_tokens_fn)
                #logits_processor.append(format_processor)
                logging.info("Added LM Format Enforcer for tool JSON.")
            except Exception as e:
                logging.info(f"Warning: Failed to initialize LM Format Enforcer: {e}. Tool usage might fail.")
                tools = [] # Disable tools if enforcer fails

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
        finish_reason = "unknown" # Transformers doesn't explicitly return this like OpenAI API
        thinking_content = ""

        try:
            with torch.no_grad(): # Ensure no gradients are computed
                 generated_ids = self.model.generate(
                     **model_inputs,
                     **generation_kwargs
                 )

            # Extract only the generated tokens
            output_ids = generated_ids[0][input_length:].tolist()

            # --- Post-processing ---

            # 1. Handle Thinking Tags (Qwen specific, adapt if needed)
            content_ids = output_ids # Default to all output
            if self._enable_thinking_flag and self._think_token_id in output_ids:
                 try:
                     # Find the *last* occurrence of the think token ID
                     # rindex_token = len(output_ids) - 1 - output_ids[::-1].index(self._think_token_id) # Find last index
                     # Find the *first* occurrence for Qwen's format
                     index = output_ids.index(self._think_token_id)
                     # Decode thinking part (up to and including </think>)
                     thinking_content = self.tokenizer.decode(output_ids[:index+1], skip_special_tokens=True).strip()
                     # Get content part (after </think>)
                     content_ids = output_ids[index+1:]
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
            elif len(output_ids) >= comp_settings.max_tokens -1: # -1 for safety
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
                      content = f"<think>{thinking_content}</think>\n{content}" # Reconstruct if needed


        except Exception as e:
            logging.info(f"Error during generation or processing: {e}")
            # Handle error case, maybe return empty or partial results
            content = f"Error during generation: {e}"
            calls = []
            finish_reason = "error"

        # --- Final Logging ---
        final_output_message = ("assistant", f"<think>{thinking_content}</think>\n{content}" if thinking_content else content) # Log full output before discard
        log_conversation(inp_formatted + [final_output_message], log_file_path + f".completion_{finish_reason}.log", 200)

        # Return final processed content and tool calls
        return content, calls


    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        discard_thinks: bool = True) -> str:

        #return controller.completion_text(preset, inp, comp_settings)

        if self.test_mode:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(128))

        # No tools expected in this simplified method
        content, _ = self.completion_tool(preset, inp, comp_settings, tools=None, discard_thinks=discard_thinks)
        return content # Already stripped in completion_tool


    def get_embedding(self, text: str) -> List[float]:
        if self.emb_model is None or self.test_mode:
            return np.random.random((768,)).tolist()
        embeddings = self.emb_model.encode([text], show_progress_bar=False)
        embeddings = torch.tensor(embeddings).to('cpu').tolist()[0]
        return embeddings

main_llm = LlmManager(test_mode=False)

from pydantic import BaseModel, Field

class SearchTool(BaseModel):
    query: str = Field(..., description="The search query to use.")

def test_llm_manager():
    llm_manager = main_llm
    # Example call for text generation
    chat_history = [
        ("system", "You are a helpful assistant."),
        ("user", "Write a short poem about clouds.")
    ]
    settings = CommonCompSettings(max_tokens=1024, temperature=0.8)
    response_text = llm_manager.completion_text(LlmPreset.Qwen3_8B_FP8, chat_history, settings)
    logging.info("--- Text Response ---")
    logging.info(response_text)

    # Example call for tool usage
    tool_chat = [
        ("system", "You are a helpful assistant with access to a search engine."),
        ("user", "What is the weather like in London?")
    ]
    tool_settings = CommonCompSettings(max_tokens=1024)
    response_content, tool_calls = llm_manager.completion_tool(
        LlmPreset.Qwen3_8B_FP8,
        tool_chat,
        tool_settings,
        tools=[SearchTool]
    )
    logging.info("\n--- Tool Response ---")
    logging.info("Content:", response_content) # Might be empty if only tool call expected
    if tool_calls:
        logging.info("Tool Calls:", tool_calls[0].model_dump_json(indent=2))
    else:
        logging.info("No valid tool call detected.")

# --- Type Variables ---
T = TypeVar('T')
BM = TypeVar('BM', bound=BaseModel)

# --- Enums ---
class NarrativeTypes(StrEnum):
    AttentionFocus = "AttentionFocus"
    SelfImage = "SelfImage"
    PsychologicalAnalysis = "PsychologicalAnalysis"
    Relations = "Relations"
    ConflictResolution = "ConflictResolution"
    EmotionalTriggers = "EmotionalTriggers"
    GoalsIntentions = "GoalsIntentions"
    BehaviorActionSelection = "BehaviorActionSelection" # How Emmy tends to choose actions

class ActionType(StrEnum):
    Sleep = "Sleep"
    All = "All"
    Reply = "Reply" # Primary external action
    ToolCall = "ToolCall"
    Ignore = "Ignore" # Primary internal action (decision not to respond externally)
    InitiateUserConversation = "InitiateUserConversation"
    InitiateInternalContemplation = "InitiateInternalContemplation"
    InitiateIdleMode = "InitiateIdleMode"
    Think = "Think" # Internal processing step
    RecallMemory = "RecallMemory"
    ManageIntent = "ManageIntent"
    Plan = "Plan"
    ManageAwarenessFocus = "ManageAwarenessFocus"
    ReflectThoughts = "ReflectThoughts"

class FeatureType(IntEnum):
    Dialogue = 10
    Feeling = 20
    SituationalModel = 30
    AttentionFocus = 40
    ConsciousWorkspace = 50
    MemoryRecall = 60
    SubjectiveExperience = 70
    ActionSimulation = 80 # A specific simulated action path
    ActionRating = 81 # The rating of a simulation
    Action = 90 # The chosen, causal action
    ActionExpectation = 100 # Deprecated - Use Intention knoxel with internal=False
    NarrativeUpdate = 110 # Record of a narrative being updated
    # Added: Feature representing the emotional reaction to expectation outcomes
    ExpectationOutcome = 120
    StoryWildcard = 140

# Helper function to serialize embeddings (adapted from your example)
def serialize_embedding(value: Optional[List[float]]) -> Optional[bytes]:
    if value is None or not value: # Handle empty list too
        return None
    # Using pickle as requested, but consider JSON for cross-language/version compatibility if needed
    return pickle.dumps(value)

# Helper function to deserialize embeddings
def deserialize_embedding(value: Optional[bytes]) -> List[float]:
    if value is None:
        return [] # Return empty list instead of None for consistency with model field type hint
    try:
        # Ensure we are dealing with bytes
        if not isinstance(value, bytes):
             logging.warning(f"Attempting to deserialize non-bytes value as embedding: {type(value)}. Trying to encode.")
             try:
                 value = str(value).encode('utf-8') # Attempt basic encoding
             except Exception:
                 logging.error("Failed to encode value to bytes for deserialization. Returning empty list.")
                 return []
        return pickle.loads(value)
    except pickle.UnpicklingError as e:
        logging.error(f"Failed to deserialize embedding (UnpicklingError): {e}. Value prefix: {value[:50] if value else 'None'}. Returning empty list.")
        return []
    except Exception as e: # Catch other potential errors like incorrect format
        logging.error(f"Failed to deserialize embedding (General Error): {e}. Value prefix: {value[:50] if value else 'None'}. Returning empty list.")
        return []

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
        for field_name, model_field in self.model_fields.items():
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
            delta_values[field_name] = max(ge, min(le, new_value))

        cls = b.__class__
        return cls(**delta_values)


    def decay_to_baseline(self, decay_factor: float = 0.1):
        for field_name, model_field in self.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)
            ge = -float("inf"); le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            decayed_value = current_value - decay_factor * (current_value - baseline)
            decayed_value = max(min(decayed_value, le), ge)
            setattr(self, field_name, decayed_value)

    def decay_to_zero(self, decay_factor: float = 0.1):
        for field_name in self.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value * (1.0 - decay_factor) # Corrected exponential decay
            setattr(self, field_name, max(decayed_value, 0.0))

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
         # Simplified: Use __add__ and scale after, or implement proper scaling
         # This implementation is flawed, let's simplify or fix.
         # For now, let's assume direct addition via __add__ is sufficient
         # and scaling happens before calling add/add_state_with_factor.
         logging.warning("add_state_with_factor needs review/simplification.")
         for field_name, value in state.items():
             if field_name in self.model_fields:
                 model_field = self.model_fields[field_name]
                 current_value = getattr(self, field_name)
                 ge = -float("inf"); le = float("inf")
                 for meta in model_field.metadata:
                     if hasattr(meta, "ge"): ge = meta.ge
                     if hasattr(meta, "le"): le = meta.le
                 new_value = current_value + value * factor
                 new_value = max(min(new_value, le), ge)
                 setattr(self, field_name, new_value)

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        delta_values = {}
        for field_name, model_field in self.model_fields.items():
            ge = -float("inf"); le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            val = (target_value - current_value) * impact
            delta_values[field_name] = max(ge, min(le, val))
        cls = b.__class__
        return cls(**delta_values)

class EmotionalAxesModelDelta(DecayableMentalState):
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=-1, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.") # Note: Delta can be negative
    anxiety: float = Field(default=0.0, ge=-1, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.") # Note: Delta can be negative

    def get_overall_valence(self):
        # Simple valence calculation for delta - might need adjustment
        return self.valence # Or a weighted sum if needed

class EmotionalAxesModel(DecayableMentalState):
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=0, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.")
    anxiety: float = Field(default=0.0, ge=0, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.")

    def get_overall_valence(self):
        disgust_bipolar = self.disgust * -1 # Higher disgust = more negative valence
        anxiety_bipolar = self.anxiety * -1 # Higher anxiety = more negative valence
        axes = [self.valence, self.affection, self.self_worth, self.trust, disgust_bipolar, anxiety_bipolar]
        weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.9] # Example weights, valence/anxiety more impactful
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
    embedding: List[float] = Field(default=[], repr=False)
    timestamp_creation: datetime = Field(default_factory=datetime.now)
    timestamp_world_begin: datetime = Field(default_factory=datetime.now)
    timestamp_world_end: datetime = Field(default_factory=datetime.now)

    def get_story_element(self) -> str:
        return f"{self.__class__.__name__}: {self.content}"

class Stimulus(KnoxelBase):
    stimulus_type: str

class Intention(KnoxelBase):
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    affective_valence: float = Field(..., ge=-1.0, le=1.0, description="Expected valence if fulfilled (for internal goals) or desired valence if met (for external expectations).")
    incentive_salience: float = Field(default=0.5, ge=0.0, le=1.0, description="How much it 'pulls' attention/action.")
    fulfilment: float = Field(default=0.0, ge=0.0, le=1.0, description="Current fulfillment status (0=not met, 1=fully met).")
    internal: bool = Field(..., description="True = internal goal/drive, False = external expectation of an event/response.")
    # Add field to link expectation back to the action that generated it
    originating_action_id: Optional[int] = Field(default=None, description="ID of the Action knoxel that generated this expectation (if internal=False).")


class Action(KnoxelBase):
    action_type: ActionType # Using the enum
    action_content: str # The core content (e.g., reply text, thought summary)
    # Links to Intention knoxels (where internal=False) generated by this action
    generated_expectation_ids: List[int] = Field(default=[], description="IDs of Intention knoxels (expectations) created by this action.")
    
# --- New Enums ---
class ClusterType(StrEnum):
    Topical = "topical"
    Temporal = "temporal"

# --- New Knoxel Models ---

class MemoryClusterKnoxel(KnoxelBase):
    """Represents a cluster of events or other clusters in memory."""
    level: int = Field(..., description="Hierarchy level (e.g., 1=Year...6=TimeOfDay for Temporal, 100 for Topical)")
    cluster_type: ClusterType # Use the enum: Topical or Temporal
    included_event_ids: Optional[str] = Field(default=None, description="Comma-separated knoxel IDs of events included (for Topical clusters)")
    included_cluster_ids: Optional[str] = Field(default=None, description="Comma-separated knoxel IDs of clusters included (for Temporal clusters)")
    token: int = Field(default=0, description="Token count, usually based on summary for Temporal clusters")
    # Embedding will be generated based on summary/content
    timestamp_from: datetime = Field(...) # Start time covered by cluster
    timestamp_to: datetime = Field(...)   # End time covered by cluster
    min_event_id: Optional[int] = Field(default=None, description="Lowest original Event/Feature knoxel ID covered")
    max_event_id: Optional[int] = Field(default=None, description="Highest original Event/Feature knoxel ID covered")
    facts_extracted: bool = Field(default=False, description="Flag for Topical clusters: has declarative memory been extracted?")
    temporal_key: Optional[str] = Field(default=None, description="Unique key for merging temporal clusters (e.g., '2023-10-26-MORNING')")

    # Override get_story_element for representation in prompts if needed
    def get_story_element(self) -> str:
        if self.cluster_type == ClusterType.Temporal and self.summary:
            # How to represent time? Maybe add time range to summary?
            # level_name = self.level # TODO: Map level back to Day/Week etc.
            return f"*Summary of period ({self.timestamp_from.strftime('%Y-%m-%d %H:%M')}): {self.summary}*"
        elif self.cluster_type == ClusterType.Topical:
            # Topical clusters might not appear directly in story, used for retrieval
            return f"*(Topic Cluster {self.id} covering events {self.min_event_id}-{self.max_event_id})*"
        return super().get_story_element() # Fallback

class DeclarativeFactKnoxel(KnoxelBase):
    """Represents an extracted declarative fact."""
    # content: Inherited (The fact statement)
    temporality: float = Field(default=0.5)
    category: Optional[str] = Field(default=None, description="Semantic category (e.g., 'Relationship', 'Preference')")
    min_event_id: Optional[int] = Field(default=None, description="Source event range start")
    max_event_id: Optional[int] = Field(default=None, description="Source event range end")
    source_cluster_id: Optional[int] = Field(default=None, description="ID of the MemoryClusterKnoxel (Topical) it came from")
    # embedding: Inherited (Generated from content)
    # token: Inherited (Calculated from content)
    # timestamps: Inherited (When fact was created/extracted)

    def get_story_element(self) -> str:
        # Facts might be recalled, not part of the direct flow
         return f"*(Fact: {self.content})*"


class CauseEffectKnoxel(KnoxelBase):
    """Represents an extracted cause-effect relationship."""
    cause: str
    effect: str
    category: Optional[str] = Field(default=None, description="Semantic category")
    min_event_id: Optional[int] = Field(default=None)
    max_event_id: Optional[int] = Field(default=None)
    source_cluster_id: Optional[int] = Field(default=None)
    # timestamp: Use timestamp_creation from KnoxelBase for when it was extracted

    # Content needs to be generated for embedding/storage
    @model_validator(mode='before')
    @classmethod
    def generate_content(cls, values):
        if 'content' not in values and 'cause' in values and 'effect' in values:
            values['content'] = f"Cause: {values['cause']} -> Effect: {values['effect']}"
        return values

    def get_story_element(self) -> str:
         return f"*(Observed Cause/Effect: {self.content})*"

# Inside DynamicMemoryConsolidator class or accessible to it

NARRATIVE_FEATURE_RELEVANCE_MAP: Dict[NarrativeTypes, List[FeatureType]] = {
    NarrativeTypes.AttentionFocus: [
        FeatureType.AttentionFocus,
        FeatureType.MemoryRecall,
        FeatureType.Dialogue, # What triggered focus shifts
        FeatureType.SubjectiveExperience, # What was being experienced
    ],
    NarrativeTypes.SelfImage: [
        FeatureType.SubjectiveExperience,
        FeatureType.Feeling,
        FeatureType.Dialogue, # How Emmy represents herself
        FeatureType.Action, # Emmy's choices reflect self-perception
        FeatureType.ExpectationOutcome, # Reactions to expectations met/failed
        FeatureType.NarrativeUpdate, # Direct learning about self
    ],
    NarrativeTypes.PsychologicalAnalysis: [ # Broader view
        FeatureType.SubjectiveExperience,
        FeatureType.Feeling,
        FeatureType.Action,
        FeatureType.ExpectationOutcome,
        FeatureType.MemoryRecall,
        FeatureType.Dialogue, # Both Emmy's and User's
        FeatureType.NarrativeUpdate,
    ],
    NarrativeTypes.Relations: [
        FeatureType.Dialogue, # How they speak to each other
        FeatureType.Feeling, # Especially affection, trust
        FeatureType.Action, # Actions taken towards/regarding the other
        FeatureType.ExpectationOutcome, # Reactions involving the other person
        FeatureType.MemoryRecall, # Memories about the relationship
        FeatureType.SubjectiveExperience, # Experiences related to the other
    ],
    NarrativeTypes.ConflictResolution: [
        FeatureType.Dialogue, # Arguments, apologies, negotiations
        FeatureType.Action, # Avoidance, confrontation, problem-solving
        FeatureType.Feeling, # Anxiety, valence shifts during conflict
        FeatureType.ExpectationOutcome, # How unmet expectations in conflict are handled
        FeatureType.SubjectiveExperience, # Internal experience during disagreement
    ],
    NarrativeTypes.EmotionalTriggers: [
        FeatureType.Dialogue, # The triggering event/dialogue
        FeatureType.Feeling, # The resulting strong emotion
        FeatureType.MemoryRecall, # Associated past events
        FeatureType.ExpectationOutcome, # Strong reactions to specific outcomes
        FeatureType.SubjectiveExperience, # The raw feel of being triggered
    ],
    NarrativeTypes.GoalsIntentions: [
        # FeatureType.IntentionKnoxel? - If you make Intention a Feature type
        FeatureType.Action, # Actions reveal underlying goals
        FeatureType.Dialogue, # Stated goals or desires
        FeatureType.SubjectiveExperience, # Mentions of wanting/needing something
    ],
    NarrativeTypes.BehaviorActionSelection: [
        FeatureType.Action, # The chosen actions
        FeatureType.ActionSimulation, # Considered alternatives (if stored as features)
        FeatureType.ActionRating, # How alternatives were rated (if stored as features)
        FeatureType.Feeling, # Emotions influencing choices
        FeatureType.ExpectationOutcome, # Learning from past action outcomes
        FeatureType.SubjectiveExperience, # Rationale/feeling before acting
    ]
}

# Define the narrative definitions accessible to the consolidator
# (Assuming narrative_definitions list from the original code is accessible)
# Example:
# from new_arch_test import narrative_definitions

class Narrative(KnoxelBase):
    narrative_type: NarrativeTypes # Use the specific Enum
    target_name: str # Who this narrative is about (companion_name or user_name)
    content: str
    last_refined_with_tick: Optional[int] = Field(default=None, description="The tick ID up to which features were considered for this narrative version.")

class Feature(KnoxelBase):
    feature_type: FeatureType
    source: str
    affective_valence: Optional[float] = None
    incentive_salience: Optional[float] = None
    interlocus: float # -1 internal, +1 external, 0 mixed/neutral
    causal: bool = False # affects story generation?

    def get_story_element(self) -> str:
        # “(.*?)”
        story_map = {
            # Dialogue: Keep as is - standard format
            FeatureType.Dialogue: f'{self.source} says: “{self.content}”',

            # Internal States (Feelings, Experience, Focus): Use italics or narrative description
            FeatureType.Feeling: f'*{self.source} felt {self.content}.*',  # Assumes content is now "a warm connection" not "<'...'
            FeatureType.SubjectiveExperience: f'*{self.content}*',  # Content should already be narrative
            FeatureType.AttentionFocus: f'*({self.source}\'s attention shifted towards: {self.content})*',  # Use parentheses for internal focus shifts

            # Actions: Describe the action narratively
            FeatureType.Action: f'*{self.source} decided to {self.__class__.__name__.lower()}.*',  # Describe the *type* of action taken

            # Memory/Context: Weave it in more naturally if possible, or use italics
            FeatureType.MemoryRecall: f'*({self.source} recalled: {self.content})*',
            FeatureType.SituationalModel: f'*({self.source} considered the situation: {self.content})*',

            # Expectation Outcomes: Describe the reaction
            FeatureType.ExpectationOutcome: f'*({self.content})*',  # Content should be the descriptive reaction

            # Narratives/Goals (If included): Frame as thoughts or context
            FeatureType.NarrativeUpdate: f'*({self.source} reflected on the narrative: {self.content})*',

            # Stimulus/Wildcard: Frame as events or scene setting
            FeatureType.StoryWildcard: f'*{self.content}*',  # General narrative element
            # Maybe handle primary stimulus differently if needed
        }
        # Default fallback
        return story_map.get(self.feature_type, f'*{self.content}*')  # Default to italicized content


# --- KnoxelList Wrapper (Unchanged) ---
class KnoxelList:
    def __init__(self, knoxels: Optional[List[KnoxelBase]] = None):
        self._list = Enumerable(knoxels if knoxels else [])

    def get_story(self, config: GhostConfig, max_tokens: Optional[int] = None) -> str:
        content_list = [k.get_story_element() for k in self._list]
        narrative = "\n".join(content_list)
        if max_tokens:
             char_limit = max_tokens * 3 # Adjusted char/token estimate
             if len(narrative) > char_limit:
                  narrative = narrative[:char_limit] + "..."
        return narrative

    def get_embeddings_np(self) -> Optional[np.ndarray]:
        embeddings = [k.embedding for k in self._list if k.embedding]
        if not embeddings: return None
        try: return np.array(embeddings)
        except ValueError:
            logging.error("Inconsistent embedding dimensions found.")
            return None

    def where(self, predicate) -> 'KnoxelList': return KnoxelList(self._list.where(predicate).to_list())
    def order_by(self, key_selector) -> 'KnoxelList': return KnoxelList(self._list.order_by(key_selector).to_list())
    def order_by_descending(self, key_selector) -> 'KnoxelList': return KnoxelList(self._list.order_by_descending(key_selector).to_list())
    def take(self, count: int) -> 'KnoxelList': return KnoxelList(self._list.take(count).to_list())
    def take_last(self, count: int) -> 'KnoxelList':
         items = self._list.to_list(); return KnoxelList(items[-count:])
    def add(self, knoxel: KnoxelBase): self._list = Enumerable(self._list.to_list() + [knoxel])
    def to_list(self) -> List[KnoxelBase]: return self._list.to_list()
    def last_or_default(self, predicate=None, default=None) -> Optional[KnoxelBase]: return self._list.last_or_default(predicate, default)
    def first_or_default(self, predicate=None, default=None) -> Optional[KnoxelBase]: return self._list.first_or_default(predicate, default)
    def any(self, predicate=None): return self._list.any(predicate)
    def __len__(self) -> int: return self._list.count()


# Inside DynamicMemoryConsolidator or standalone taking knoxels and contextual_embeddings
# Requires sklearn, numpy, scipy (already imported in original file)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

class ClusterDescription(BaseModel): # Keep this helper model
    conceptual_cluster: Dict[int, int] = Field(default_factory=dict) # Maps knoxel ID -> cluster label


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

from pydantic import BaseModel, create_model, Field
import random
import string
from copy import deepcopy


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


sysprompt = f"""You are a helpful assistant that helps organize facts from a story about {companion_name}, an AI and her User, {user_name}, into categories. These categories will help the author not forget stuff.
You do this by creating a JSON where you rate the belongingness to different categories with a float between 0 and 1.    
"""


class FactCategory(KnoxelBase):
    fact_id: int
    category: str
    category_id: int
    weight: float
    max_weight: float

def categorize_fact(k: KnoxelBase) -> List[FactCategory]:
    res = []
    bm = init_basemodel()
    messages = [
        ("system", sysprompt),
        ("user", f"Can you put this fact into the different categories? ### BEGIN FACT\n{k.content}\n### END FACT")
    ]

    _, calls = main_llm.completion_tool(LlmPreset.Qwen3_8B_FP8, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[bm])

    mb = calls[0]
    max_weight = 0
    for f in mb.model_fields_set:
        max_weight = max(max_weight, getattr(mb, f))

    for f in mb.model_fields_set:
        weight = getattr(mb, f)
        cat_id = path_to_id.get(f, 0)
        fc = FactCategory(
            fact_id=k.id,
            category=f,
            category_id=cat_id,
            weight=weight,
            max_weight=max_weight)
        res.append(fc)
    return res


from typing import List

from pydantic import BaseModel, Field


class Facts(BaseModel):
    facts: List[str] = Field(default_factory=list, description="list of facts extracted from conversation")

class TimeInvariance(BaseModel):
    time_variance: float = Field(description="0 for time-invariant and 1 for time-variant. meaning 0 is for facts true at all times, and 1 for statements only correct right now")

def extract_facts(block: str):
    sysprompt = "You are a helpful assistant. You extract facts from a conversation."
    prompt = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n"),
    ]

    _, calls = main_llm.completion_tool(LlmPreset.Qwen3_8B_FP8, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[Facts])
    facts = []
    for fact in calls[0].facts:
        sysprompt = "You are a helpful assistant. You extract determine if a fact is time-variant or time-invariant. Use 0 for facts that are true at all times, and 1 for facts only true right now."
        prompt = [
            ("system", sysprompt),
            ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n### BEGIN FACT\n{fact}\n### END FACT\nIs the extracted fact from the messages time-independent (0) or time-dependent (1)?"),
        ]

        _, calls2 = main_llm.completion_tool(LlmPreset.Qwen3_8B_FP8, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[TimeInvariance])
        time_variance = calls2[0].time_variance
        facts.append((fact, time_variance))

    return facts



import logging
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Type
from pydantic import BaseModel, Field

# Define CauseEffectSchema if extract_cas returns it
class CauseEffectSchema(BaseModel):
     cause: str
     effect: str

logger = logging.getLogger(__name__)


# --- Temporal Key Generation Functions (Keep these) ---
def get_temporal_key(timestamp: datetime, level: int) -> str:
    # ... (implementation from previous response) ...
    if level == 6: # Time Of Day (combining with day for uniqueness)
        hour = timestamp.hour
        tod = "NIGHT" # Default
        if 5 <= hour < 12: tod = "MORNING"
        elif 12 <= hour < 14: tod = "NOON"
        elif 14 <= hour < 18: tod = "AFTERNOON"
        elif 18 <= hour < 22: tod = "EVENING"
        return f"{timestamp.strftime('%Y-%m-%d')}-{tod}"
    elif level == 5: return timestamp.strftime('%Y-%m-%d') # Day
    elif level == 4: return timestamp.strftime('%Y-W%W') # Week
    elif level == 3: return timestamp.strftime('%Y-%m') # Month
    elif level == 2: # Season
        season = (timestamp.month % 12 + 3) // 3
        return f"{timestamp.year}-S{season}"
    elif level == 1: return str(timestamp.year) # Year
    else: raise ValueError(f"Invalid temporal level: {level}")

def level_to_parent_level(level: int) -> Optional[int]:
    # ... (implementation from previous response) ...
    if level > 1: return level - 1
    return None


# --- Configuration Class (Define or integrate into GhostConfig) ---
class MemoryConsolidationConfig(BaseModel):
    consolidation_token_threshold: int = 2000
    # cluster_split_ratio: float = 0.6 # Decide if needed or use simple token threshold
    included_feature_types: List[FeatureType] = Field(default_factory=lambda: [
        FeatureType.Dialogue, FeatureType.SubjectiveExperience, FeatureType.Action
    ])
    topical_clustering_context_window: int = 3
    min_events_for_processing: int = 3 # Min events for a topical cluster to be processed
    threshold: int = 24


# --- Main Consolidator Class ---
class DynamicMemoryConsolidator:

    def __init__(self, ghost: "Ghost", config: MemoryConsolidationConfig):
        self.ghost = ghost
        self.config = config
        self.llm_manager = ghost.llm_manager # Reuse Ghost's LLM manager

        self.narrative_feature_map = NARRATIVE_FEATURE_RELEVANCE_MAP
        # Make sure narrative_definitions are available, e.g., load from Ghost's config or import
        self.narrative_definitions = narrative_definitions  # Assuming it's imported/available


    # Example adaptation (could be method of DynamicMemoryConsolidator)
    def _categorize_fact_knoxel(self, fact_knoxel: DeclarativeFactKnoxel):
        if not fact_knoxel.content: return
        # Replace with your actual categorization logic (e.g., LLM call, keyword matching)
        # Example using cosine_cluster logic from original code:
        try:
            cats = categorize_fact(fact_knoxel)
            for cat in cats:
                self.ghost.add_knoxel(cat)
        except Exception as e:
            logger.warning(f"Failed to categorize Fact {fact_knoxel.id}: {e}")

    def consolidate_memory_if_needed(self):
        logger.info("Checking if memory consolidation is needed...")
        unclustered_events = self._get_unclustered_event_knoxels()

        if not unclustered_events:
            logger.info("No unclustered event knoxels found.")
            return

        # Estimate token count (crude, needs access to tokenizer ideally)
        # total_unclustered_tokens = sum(len(k.content)//4 for k in unclustered_events if k.content)
        # Simplified: Use count for now, or pass tokenizer
        num_unclustered = len(unclustered_events)
        # Use a count-based threshold or adapt token calculation
        threshold = self.config.threshold # Example: Run if more than 50 unclustered events

        logger.info(f"Unclustered event knoxels: {num_unclustered}. Threshold: {threshold}")

        if num_unclustered >= threshold:
            logger.info("Threshold met. Starting memory consolidation...")
            self._perform_consolidation(unclustered_events)

            current_tick_id = self.ghost._get_current_tick_id()
            self.refine_narratives(current_tick_id)

            logger.info("Memory consolidation finished.")
        else:
            logger.info("Threshold not met. Skipping consolidation.")


    def refine_narratives(self, current_tick_id: int, max_lookback_ticks: int = 200, min_new_features: int = 5):
        """
        Iterates through defined narratives and refines them if enough relevant new features exist.
        """
        logger.info(f"Starting narrative refinement check (Current Tick: {current_tick_id}).")

        # Get all current Feature knoxels, sorted by tick_id then id
        all_features = sorted(
            [k for k in self.ghost.all_knoxels.values() if isinstance(k, Feature)],
            key=lambda f: (f.tick_id, f.id)
        )

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

            # Determine the start tick for fetching new features
            # If never refined, use the creation tick of the previous narrative, or look back max_lookback_ticks
            start_tick = last_refined_tick + 1
            if last_refined_tick == -1:
                if previous_narrative:
                    start_tick = previous_narrative.tick_id + 1
                else:  # No narrative exists at all, look back globally
                    start_tick = max(0, current_tick_id - max_lookback_ticks)

            # Find relevant feature types for this narrative
            relevant_feature_types = self.narrative_feature_map.get(narrative_type, [])
            if not relevant_feature_types:
                logger.debug(f"No relevant feature types defined for narrative {narrative_type.name}. Skipping.")
                continue

            # Filter recent features based on type and tick range
            recent_relevant_features = [
                f for f in all_features
                if f.feature_type in relevant_feature_types and f.tick_id >= start_tick and f.tick_id <= current_tick_id
            ]

            logger.debug(f"Found {len(recent_relevant_features)} relevant features since tick {start_tick} (Min needed: {min_new_features}).")

            # Check if enough new features exist to warrant refinement
            if len(recent_relevant_features) >= min_new_features:
                logger.info(f"Refining narrative '{narrative_type.name}' for '{target_name}' based on {len(recent_relevant_features)} new features.")

                # Format features for the prompt (simple chronological list)
                feature_context_lines = []
                for feature in recent_relevant_features:
                    # Use a concise format for the prompt
                    source_str = f"{feature.source}: " if hasattr(feature, 'source') and feature.source else ""
                    content_snippet = feature.content[:150] + ('...' if len(feature.content) > 150 else '')
                    feature_context_lines.append(
                        f"- Tick {feature.tick_id}: [{feature.feature_type.name}] {source_str}{content_snippet}"
                    )
                feature_context = "\n".join(feature_context_lines)

                # --- Construct the Refinement Prompt ---
                prompt = f"""
You are refining the psychological/behavioral narrative profile for '{target_name}'.
Narrative Type Goal: "{prompt_goal}"

Current Narrative for '{target_name}' ({narrative_type.name}):
---
{previous_content}
---

New Evidence (Relevant Events/Thoughts Since Last Update - Tick {start_tick} to {current_tick_id}):
---
{feature_context}
---

Task: Analyze the 'New Evidence'. Based *only* on this new information, provide an *updated and improved* version of the narrative for '{target_name}' regarding '{narrative_type.name}'. Integrate insights, modify conclusions subtly, add new observations, or keep the original narrative section if no new relevant information contradicts or enhances it. Ensure the output is the complete, updated narrative text, fulfilling the Narrative Type Goal.

Output *only* the complete, updated narrative text below:
Updated Narrative:
"""

                # --- Call LLM ---
                updated_content_raw = self.ghost._call_llm(prompt,max_tokens=768)

                # --- Process LLM Response ---
                if updated_content_raw:
                    # Clean up potential LLM artifacts like "Updated Narrative:" prefixes
                    updated_content = updated_content_raw.strip()
                    if updated_content.startswith("Updated Narrative:"):
                        updated_content = updated_content[len("Updated Narrative:"):].strip()

                    if updated_content and updated_content != previous_content:
                        logger.info(f"Narrative '{narrative_type.name}' for '{target_name}' successfully updated.")
                        # Create the NEW narrative knoxel
                        new_narrative = Narrative(
                            narrative_type=narrative_type,
                            target_name=target_name,
                            content=updated_content,
                            last_refined_with_tick=current_tick_id,  # Mark refinement tick
                            # tick_id, id, timestamp_creation set by add_knoxel
                        )
                        # Add to ghost state (automatically archives the old one by latest ID/tick)
                        # Generate embedding for the new narrative content
                        self.ghost.add_knoxel(new_narrative, generate_embedding=True)
                    elif updated_content == previous_content:
                        logger.info(f"LLM refinement resulted in no change for narrative '{narrative_type.name}' for '{target_name}'.")
                        # Optionally update the last_refined_with_tick on the *previous* narrative?
                        if previous_narrative:
                            previous_narrative.last_refined_with_tick = current_tick_id
                            self.ghost.add_knoxel(previous_narrative, generate_embedding=False)  # Save update
                    else:
                        logger.warning(f"LLM refinement returned empty content for narrative '{narrative_type.name}' for '{target_name}'.")
                else:
                    logger.warning(f"LLM call failed for narrative refinement: {narrative_type.name} for {target_name}.")
            # else: logger.debug(f"Not enough new features to refine narrative {narrative_type.name} for {target_name}.") # Too verbose maybe

        logger.info("Narrative refinement check complete.")

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
        # Iterate through relevant knoxel types
        candidate_knoxels = [k for k in self.ghost.all_knoxels.values() if isinstance(k, Feature) and k.feature_type in self.config.included_feature_types]
        # TODO: Add other types like Action, Stimulus if desired in config?

        for knoxel in sorted(candidate_knoxels, key=lambda k: k.id): # Process in ID order
             if knoxel.id not in clustered_event_ids:
                 # Ensure embedding exists before adding
                 if not knoxel.embedding and knoxel.content:
                     logger.debug(f"Generating missing embedding for unclustered knoxel {knoxel.id}")
                     self.ghost._generate_embedding(knoxel) # Use ghost's method
                 if knoxel.embedding: # Only include if embedding is present
                     unclustered_knoxels.append(knoxel)
                 # else: logger.warning(f"Skipping unclustered knoxel {knoxel.id}: no embedding.")


        logger.debug(f"Found {len(unclustered_knoxels)} unclustered event knoxels with embeddings.")
        return unclustered_knoxels

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
                self._update_temporal_hierarchy(topical_cluster, level=6) # Start at TimeOfDay
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing topical cluster ID {topical_cluster.id}: {e}", exc_info=True)
                # Continue to next cluster

        logger.info(f"Finished processing {processed_count} topical clusters.")


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

        added_event_ids = set() # Track which event knoxels get linked

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
            total_token = sum(len(k.content)//4 for k in cluster_knoxels if k.content) # Crude token estimate
            avg_embedding = np.mean(np.array([k.embedding for k in cluster_knoxels if k.embedding]), axis=0) if any(k.embedding for k in cluster_knoxels) else []
            timestamp_begin = min(k.timestamp_creation for k in cluster_knoxels)
            timestamp_end = max(k.timestamp_creation for k in cluster_knoxels)
            min_id = min(k.id for k in cluster_knoxels)
            max_id = max(k.id for k in cluster_knoxels)
            event_ids_str = ",".join(str(k.id) for k in cluster_knoxels)

            # Create the new MemoryClusterKnoxel (Topical)
            topical_cluster = MemoryClusterKnoxel(
                level=100,
                cluster_type=ClusterType.Topical,
                content=f"Topic Cluster Label {cluster_label}", # Placeholder
                included_event_ids=event_ids_str,
                token=max(1, total_token), # Ensure token is at least 1
                embedding=avg_embedding.tolist() if isinstance(avg_embedding, np.ndarray) else [],
                timestamp_from=timestamp_begin,
                timestamp_to=timestamp_end,
                min_event_id=min_id,
                max_event_id=max_id,
                facts_extracted=False,
                # Inherits id, tick_id (current tick), timestamp_creation from add_knoxel
            )

            # Add to Ghost state - This assigns ID, tick_id, timestamp_creation
            # We generate embedding based on summary later if needed, or maybe average is enough?
            # For topical clusters, embedding might be less crucial than for temporal summaries.
            self.ghost.add_knoxel(topical_cluster, generate_embedding=False) # Don't embed placeholder summary
            new_cluster_knoxels.append(topical_cluster)

            # Track added event IDs
            for knox in cluster_knoxels:
                 added_event_ids.add(knox.id)

        logger.info(f"Created {len(new_cluster_knoxels)} new topical clusters.")
        return new_cluster_knoxels

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
            source = getattr(k, 'source', 'Unknown') # Handle knoxels without source
            content = getattr(k, 'content', '')
            lines.append(f"{source}: {content}")
        block = "\n".join(lines)

        min_id = topical_cluster_knoxel.min_event_id
        max_id = topical_cluster_knoxel.max_event_id

        # --- Extract Facts ---
        try:
            extracted_raw_facts = extract_facts(block) # Returns List[Tuple[str, str]]
            logger.info(f"Extracted {len(extracted_raw_facts)} potential facts from cluster {topical_cluster_knoxel.id}.")
            for fact_content, temporality in extracted_raw_facts:
                new_fact = DeclarativeFactKnoxel(
                    content=fact_content,
                    temporality=temporality,
                    min_event_id=min_id,
                    max_event_id=max_id,
                    source_cluster_id=topical_cluster_knoxel.id,
                    # category will be set by _categorize_fact_knoxel
                )
                # Add_knoxel assigns ID, calculates token, generates embedding
                self.ghost.add_knoxel(new_fact, generate_embedding=True)
                self._categorize_fact_knoxel(new_fact) # Categorize and potentially save category update
        except Exception as e:
            logger.error(f"Error during fact extraction for cluster {topical_cluster_knoxel.id}: {e}", exc_info=True)

        # --- Extract Cause/Effect ---
        #try:
        #    extracted_cas: List[CauseEffectSchema] = extract_cas(block)
        #    logger.info(f"Extracted {len(extracted_cas)} potential cause/effect pairs from cluster {topical_cluster_knoxel.id}.")
        #    for ca_schema in extracted_cas:
        #         # Optional: Categorize CA
        #         ca_str = f"Cause: {ca_schema.cause} -> Effect: {ca_schema.effect}"
        #         category = self._categorize_text(ca_str) # Use helper
        #         new_ca = CauseEffectKnoxel(
        #             cause=ca_schema.cause,
        #             effect=ca_schema.effect,
        #             category=category,
        #             min_event_id=min_id,
        #             max_event_id=max_id,
        #             source_cluster_id=topical_cluster_knoxel.id,
        #             # content generated by validator, timestamp by KnoxelBase
        #         )
        #         self.ghost.add_knoxel(new_ca, generate_embedding=True)
        #except Exception as e:
        #    logger.error(f"Error during Cause/Effect extraction for cluster {topical_cluster_knoxel.id}: {e}", exc_info=True)

        # Mark cluster as processed
        topical_cluster_knoxel.facts_extracted = True
        self.ghost.add_knoxel(topical_cluster_knoxel, generate_embedding=False) # Save the flag change


    def _update_temporal_hierarchy(self, input_cluster_knoxel: MemoryClusterKnoxel, level: int):
        """Recursively finds/creates/updates temporal clusters."""
        if level < 1: return

        logger.debug(f"Updating temporal hierarchy at level {level} for input cluster {input_cluster_knoxel.id} ({input_cluster_knoxel.cluster_type.name})")

        mid_timestamp = input_cluster_knoxel.timestamp_from + (input_cluster_knoxel.timestamp_to - input_cluster_knoxel.timestamp_from) / 2
        temporal_key = get_temporal_key(mid_timestamp, level)

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
            if input_cluster_knoxel.timestamp_from < target_cluster.timestamp_from:
                target_cluster.timestamp_from = input_cluster_knoxel.timestamp_from
                time_updated = True
            if input_cluster_knoxel.timestamp_to > target_cluster.timestamp_to:
                target_cluster.timestamp_to = input_cluster_knoxel.timestamp_to
                time_updated = True

            if updated_or_created or time_updated:
                 logger.info(f"Updating existing level {level} cluster {target_cluster.id} (Key: {temporal_key}). Includes {len(current_ids)} clusters.")
                 # No need to call add_knoxel yet, changes are in memory. Wait until after potential summary.
            else:
                 logger.debug(f"No updates needed for existing cluster {target_cluster.id} based on input {input_cluster_knoxel.id}.")

        else:
            logger.info(f"Creating new level {level} temporal cluster for key {temporal_key}.")
            new_cluster = MemoryClusterKnoxel(
                level=level,
                cluster_type=ClusterType.Temporal,
                content="", # Will generate
                included_cluster_ids=str(input_cluster_knoxel.id),
                token=0,
                embedding=[],
                timestamp_from=input_cluster_knoxel.timestamp_from,
                timestamp_to=input_cluster_knoxel.timestamp_to,
                min_event_id=input_cluster_knoxel.min_event_id, # Approximate initially
                max_event_id=input_cluster_knoxel.max_event_id, # Approximate initially
                temporal_key=temporal_key,
                # Inherits id, tick_id, etc. from add_knoxel
            )
            self.ghost.add_knoxel(new_cluster, generate_embedding=False) # Add to get ID
            target_cluster = new_cluster
            needs_resummarize = True
            updated_or_created = True
            logger.debug(f"Created new temporal cluster {target_cluster.id}.")

        # Re-summarize if needed
        if needs_resummarize and target_cluster:
            logger.info(f"Re-summarizing cluster {target_cluster.id} (Level {level}, Key: {temporal_key})...")
            summary_content = self._generate_temporal_summary(target_cluster)
            target_cluster.summary = summary_content
            target_cluster.token = len(summary_content) // 4 # Crude token estimate
            # Embedding should be generated for the summary
            # Set min/max based on children
            min_ev_id, max_ev_id = self._get_min_max_event_id_for_temporal(target_cluster)
            target_cluster.min_event_id = min_ev_id
            target_cluster.max_event_id = max_ev_id

            logger.debug(f"Summary generated for {target_cluster.id}. Est. Token: {target_cluster.token}. Event range: {min_ev_id}-{max_ev_id}")
            # Save the updated cluster with summary (and generate embedding)
            self.ghost.add_knoxel(target_cluster, generate_embedding=True)

        # Recurse if a cluster was created or updated
        parent_level = level_to_parent_level(level)
        if parent_level and target_cluster and updated_or_created:
             logger.debug(f"Propagating update from level {level} cluster {target_cluster.id} to parent level {parent_level}.")
             self._update_temporal_hierarchy(target_cluster, parent_level)


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
        included_clusters.sort(key=lambda c: c.timestamp_from) # Sort children by time

        for cluster in included_clusters:
            if cluster.cluster_type == ClusterType.Temporal and cluster.summary:
                summary_input_lines.append(f"Summary ({cluster.timestamp_from.strftime('%H:%M')}): {cluster.summary}")
            elif cluster.cluster_type == ClusterType.Topical and cluster.included_event_ids:
                 event_ids = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                 event_knoxels = [self.ghost.get_knoxel_by_id(eid) for eid in event_ids if self.ghost.get_knoxel_by_id(eid)]
                 if event_knoxels:
                     # Get first few lines of the topic
                     topic_lines = [f"{getattr(k, 'source', '?')}: {getattr(k, 'content', '')[:50]}..." for k in sorted(event_knoxels, key=lambda x:x.id)[:3]]
                     summary_input_lines.append(f"Topic ({cluster.timestamp_from.strftime('%H:%M')} between events {cluster.min_event_id}-{cluster.max_event_id}):\n  " + "\n  ".join(topic_lines))

        if not summary_input_lines: return f"Temporal cluster {temporal_cluster_knoxel.id} - No content found in children."

        # --- Call LLM for Summarization ---
        # Reuse summarize_messages or create a dedicated prompt
        full_summary_input = "\n".join(summary_input_lines)
        prompt = f"Summarize the key events and themes from the following periods/topics within a specific timeframe (Level {temporal_cluster_knoxel.level} - {temporal_cluster_knoxel.temporal_key}). Be concise and focus on the narrative flow.\n\nContext:\n{full_summary_input}\n\nConcise Summary:"
        comp_settings = CommonCompSettings(temperature=0.5, max_tokens=512) # Adjust settings
        # Use _call_llm method from Ghost or llm_manager directly
        final_summary = self.ghost._call_llm(prompt) # Assuming _call_llm exists on Ghost
        return final_summary if final_summary else f"[Summary generation failed for cluster {temporal_cluster_knoxel.id}]"


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
                # Calculate token count - HOW? Need access to tokenizer or store it
                # Placeholder: Assume token count is roughly len(content)/4
                token_count = len(k.content) // 4 if k.content else 1
                tokens.append(max(1, token_count))  # Avoid zero tokens
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
                if 0 < n_clusters_found < num_events:
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


# --- GhostState (Unchanged from previous version) ---
class GhostState(BaseModel):
    class Config: arbitrary_types_allowed=True

    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)

    primary_stimulus: Optional[Stimulus] = None
    attention_candidates: KnoxelList = KnoxelList()
    attention_focus: KnoxelList = KnoxelList() # What was selected by attention (can be derived from workspace?)
    conscious_workspace: KnoxelList = KnoxelList()
    subjective_experience: Optional[Feature] = None

    # Action Deliberation & Simulation Results
    action_simulations: List[Dict[str, Any]] = [] # Store results of MC simulations [{sim_id, type, content, rating, predicted_state, user_reaction}, ...]
    selected_action_details: Optional[Dict[str, Any]] = None # Details of the chosen simulation/action before execution
    selected_action_knoxel: Optional[Action] = None # The final Action knoxel generated (set in execute)

    # State snapshots
    state_emotions: EmotionalAxesModel = EmotionalAxesModel()
    state_needs: NeedsAxesModel = NeedsAxesModel()
    state_cognition: CognitionAxesModel = CognitionAxesModel()


# --- Main Cognitive Cycle Class ---
class Ghost:
    def __init__(self, config: GhostConfig, test_mode: bool = False):
        self.config = config
        #self.client = Client(host=config.ollama_host)

        self.current_tick_id: int = 0
        self.current_knoxel_id: int = 0

        self.all_knoxels: Dict[int, KnoxelBase] = {}
        self.all_features: List[Feature] = []
        self.all_stimuli: List[Stimulus] = []
        self.all_intentions: List[Intention] = [] # Includes internal goals AND external expectations
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
        self.memory_consolidator = DynamicMemoryConsolidator(self, self.memory_config) # Pass self (ghost instance)


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
                if definition["type"] == NarrativeTypes.SelfImage and definition["target"] == self.config.companion_name:
                    default_content = self.config.universal_character_card # Use character card for initial self-image
                elif definition["type"] == NarrativeTypes.SelfImage and definition["target"] == self.config.user_name:
                    default_content = f"{self.config.user_name} is the user interacting with {self.config.companion_name}."

                narrative = Narrative(
                    narrative_type=definition["type"],
                    target_name=definition["target"],
                    content=default_content,
                )
                self.add_knoxel(narrative, generate_embedding=False) # Embeddings generated on demand or during learning
        logging.info(f"Ensured {len(self.all_narratives)} initial narratives exist.")


    def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True):
        if knoxel.id == -1: knoxel.id = self._get_next_id()
        if knoxel.tick_id == -1: knoxel.tick_id = self._get_current_tick_id()

        self.all_knoxels[knoxel.id] = knoxel

        # Add to specific lists
        if isinstance(knoxel, Feature): self.all_features.append(knoxel)
        elif isinstance(knoxel, Stimulus): self.all_stimuli.append(knoxel)
        elif isinstance(knoxel, Intention):
             # Check for duplicates based on content and type? For now, allow multiple.
             self.all_intentions.append(knoxel)
        elif isinstance(knoxel, Narrative):
             # Remove older narrative of same type/target before adding new one
             self.all_narratives = [n for n in self.all_narratives if not (n.narrative_type == knoxel.narrative_type and n.target_name == knoxel.target_name)]
             self.all_narratives.append(knoxel)
        elif isinstance(knoxel, Action): self.all_actions.append(knoxel)
        elif isinstance(knoxel, MemoryClusterKnoxel): self.all_episodic_memories.append(knoxel)
        elif isinstance(knoxel, DeclarativeFactKnoxel): self.all_declarative_facts.append(knoxel)

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
        except ResponseError as e: logging.error(f"Ollama embedding error for Knoxel {knoxel.id}: {e.error}")
        except Exception as e: logging.error(f"Failed to generate embedding for Knoxel {knoxel.id}: {e}")

    def _call_llm(self, prompt: str, output_schema: Optional[Type[BM]] = None, stop_words: List[str] | None = None, temperature: float = 0.6, max_tokens: int = 1024, sysprompt: str = None) -> str | BM | float | int | bool | None:
        max_tokens = 4096 # Keep higher default

        """ Wraps LLM call, handling schema validation or text output. """
        if stop_words is None: stop_words = []

        if not sysprompt:
            sysprompt = (f"You are a helpful assistant for story writing. Your goal is to analyze the mental states of a character. "
                         f"Always make descriptive sceneries of the mental state of the character, so it will fit nicely into the story I'm writing.") # embodying the persona of {self.config.companion_name}. {self.config.universal_character_card}"

        msgs = [("system", sysprompt),
                ("user", prompt)]

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
        except ResponseError as e:
            logging.error(f"Ollama API error during LLM call: {e.error}")
            return None
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
                     distances.append((float('inf'), k)) # Put problematic ones last
             sorted_candidates = sorted(distances, key=lambda x: x[0])
             return [k for dist, k in sorted_candidates[:limit]]


    # --- Tick Method and Sub-functions ---
    def tick(self, impulse: Optional[str] = None):
        logging.info(f"--- Starting Tick {self.current_tick_id + 1} ---")
        self.current_tick_id += 1

        # 1. Initialize State (Handles decay, internal stimulus gen, carries over intentions/expectations)
        self._initialize_tick_state(impulse)
        if not self.current_state: return None

        # 2. Retrieve Context & Update Mental States (Appraisal - includes expectation checking)
        context_knoxels = self._retrieve_and_prepare_context()
        self._appraise_stimulus_and_update_states(context_knoxels) # ***MODIFIED***

        # 3. Generate Short-Term Intentions (Internal Goals)
        self._generate_short_term_intentions(context_knoxels)

        # 4. Attention / Conscious Workspace Simulation
        self._simulate_attention_and_workspace(context_knoxels)

        # 5. Subjective Experience Generation
        self._generate_subjective_experience()

        # 6. Action Deliberation & Selection
        chosen_action_details = self._deliberate_and_select_action()
        self.current_state.selected_action_details = chosen_action_details

        # 7. Execute Action (Generates expectations)
        final_output = self._execute_action() # ***MODIFIED***

        # 8. Learning / Memory Consolidation (Can use expectation outcomes)
        #self._perform_learning_and_consolidation() # ***MODIFIED***
        self.memory_consolidator.consolidate_memory_if_needed()

        self.states.append(self.current_state)
        logging.info(f"--- Finished Tick {self.current_tick_id} ---")
        return final_output # Return the final output string (reply) or None

    def _initialize_tick_state(self, impulse: Optional[str]):
        """Sets up the GhostState for the current tick, handling state decay,
           carrying over unfulfilled intentions AND expectations, and generating internal stimuli."""
        previous_state = self.states[-1] if self.states else None
        self.current_state = GhostState(tick_id=self._get_current_tick_id())

        carried_intentions_and_expectations = [] # Renamed for clarity
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

            # Carry over UNFULFILLED intentions (internal goals) AND expectations (internal=False)
            carried_intentions_and_expectations = [
                intent for intent in self.all_intentions
                if intent.fulfilment < 1.0 and intent.tick_id <= previous_state.tick_id # Active in previous state or earlier
            ]

            # Decay urgency/salience of carried-over items
            decay_multiplier = self.config.default_decay_factor ** ticks_elapsed
            for item in carried_intentions_and_expectations:
               item.urgency *= decay_multiplier
               item.incentive_salience *= decay_multiplier
               item.urgency = max(0.0, min(0.99, item.urgency)) # Avoid reaching exactly 0 unless fulfilled
               item.incentive_salience = max(0.0, min(0.99, item.incentive_salience))

            # Add carried items to attention candidates
            self.current_state.attention_candidates = KnoxelList(carried_intentions_and_expectations)
            if carried_intentions_and_expectations:
                logging.info(f"Carried over {len(carried_intentions_and_expectations)} active intentions/expectations to attention candidates.")

        else: # First tick
             self.current_state.attention_candidates = KnoxelList()


        primary_stimulus = None
        if impulse:
            source = user_name
            ft = FeatureType.Dialogue
            il = 1

            primary_stimulus = Stimulus(stimulus_type="message", content=impulse)
            logging.info(f"Received external stimulus: {impulse}")
        else:
            # Internal stimulus generation logic (unchanged)
            source = companion_name
            ft = FeatureType.StoryWildcard
            il = -1

            needs = self.current_state.state_needs
            need_threshold = 0.4
            internal_stimulus_content = None
            low_needs = sorted(
                [(name, getattr(needs, name)) for name in needs.model_fields if getattr(needs, name) < need_threshold],
                key=lambda item: item[1]
            )
            if low_needs:
                 need_name, need_value = low_needs[0]
                 internal_stimulus_content = f"{self.config.companion_name} feels a lack of {need_name.replace('_', ' ')} ({need_value:.2f})."
            elif carried_intentions_and_expectations: # Check carried items for urgency
                urgent_carried = sorted(carried_intentions_and_expectations, key=lambda i: i.urgency + i.incentive_salience, reverse=True)
                if urgent_carried and (urgent_carried[0].urgency + urgent_carried[0].incentive_salience) > 1.0:
                    most_urgent = urgent_carried[0]
                    prefix = "recalls the urgent goal" if most_urgent.internal else "reconsiders the unmet expectation"
                    internal_stimulus_content = f"{self.config.companion_name} {prefix}: '{most_urgent.content[:80]}...'"
            if not internal_stimulus_content:
                internal_stimulus_content = f"{self.config.companion_name} takes a moment for internal reflection, considering the current state."

            primary_stimulus = Stimulus(stimulus_type="internal_state_change", content=internal_stimulus_content)
            logging.info(f"Generated internal stimulus: {internal_stimulus_content}")

        if primary_stimulus:
            self.add_knoxel(primary_stimulus)
            self.current_state.primary_stimulus = primary_stimulus

            story_feature = Feature(content=primary_stimulus.content, source=source, feature_type=ft, interlocus=il, causal=True)
            self.add_knoxel(story_feature)

            # Add stimulus to candidates if not already present (e.g., if internally generated from carried item)
            if not self.current_state.attention_candidates.where(lambda k: isinstance(k, Stimulus) and k.id == primary_stimulus.id).any():
                self.current_state.attention_candidates.add(primary_stimulus)

    def _retrieve_and_prepare_context(self) -> KnoxelList:
        """Retrieves relevant memories, facts, narratives based on the primary stimulus."""
        if not self.current_state or not self.current_state.primary_stimulus: return KnoxelList()
        stimulus = self.current_state.primary_stimulus
        if not stimulus.embedding: self._generate_embedding(stimulus)
        if not stimulus.embedding: return KnoxelList()

        # --- Retrieval logic (largely unchanged) ---
        retrieved_memories = self._get_relevant_knoxels(stimulus.embedding, self.all_episodic_memories, self.config.retrieval_limit_episodic)
        retrieved_facts = self._get_relevant_knoxels(stimulus.embedding, self.all_declarative_facts, self.config.retrieval_limit_facts)
        recent_causal_features = Enumerable(self.all_features)\
            .where(lambda x: x.causal)\
            .order_by_descending(lambda x: x.tick_id)\
            .take(self.config.retrieval_limit_features_context)\
            .to_list()
        relevant_narratives = self._get_relevant_knoxels(stimulus.embedding, self.all_narratives, 5)
        core_narrative_types = [NarrativeTypes.SelfImage, NarrativeTypes.Relations, NarrativeTypes.EmotionalTriggers, NarrativeTypes.BehaviorActionSelection]
        for ntype in core_narrative_types:
             ai_narr = self.get_narrative(ntype, self.config.companion_name)
             if ai_narr and ai_narr not in relevant_narratives: relevant_narratives.append(ai_narr)
             user_narr = self.get_narrative(ntype, self.config.user_name)
             if user_narr and user_narr not in relevant_narratives: relevant_narratives.append(user_narr)

        all_context = retrieved_memories + retrieved_facts + recent_causal_features + relevant_narratives
        unique_context = list({k.id: k for k in all_context}.values())

        # --- Add retrieved context to attention candidates ---
        for ctx_knoxel in unique_context:
            if not self.current_state.attention_candidates.where(lambda k: k.id == ctx_knoxel.id).any():
                 self.current_state.attention_candidates.add(ctx_knoxel)

        logging.info(f"Retrieved and added {len(unique_context)} unique context knoxels to attention candidates.")
        return KnoxelList(unique_context)

    def _retrieve_active_expectations(self) -> List[Intention]:
        """Retrieves active (unfulfilled, recent) EXTERNAL expectations relevant to the stimulus."""
        if not self.current_state or not self.current_state.primary_stimulus: return []
        stimulus = self.current_state.primary_stimulus
        if not stimulus.embedding: self._generate_embedding(stimulus)
        if not stimulus.embedding: return []
        stimulus_embedding_np = np.array(stimulus.embedding)

        candidate_expectations: List[Tuple[float, Intention]] = []
        # Define recency window (e.g., last 10-20 ticks) - adjust as needed
        min_tick_id = max(0, self.current_state.tick_id - 20)

        # Find intentions marked as external expectations, unfulfilled, and recent
        potential_expectations = [
            intent for intent in self.all_intentions
            if not intent.internal and intent.fulfilment < 1.0 and intent.tick_id >= min_tick_id
        ]

        processed_expectation_ids = set() # Avoid processing duplicates if somehow present
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
            if recency_factor < 0.05: continue # Threshold to prune very old/decayed expectations

            try:
                # Similarity between stimulus and expectation content
                similarity = 1.0 - cosine_distance(stimulus_embedding_np, np.array(expectation.embedding))
                similarity = max(0.0, min(1.0, similarity)) # Clamp similarity
            except Exception: similarity = 0.0

            # Combine factors: relevance = recency * similarity * (urgency + salience bias)
            # Give higher weight to similarity and recency
            relevance_score = (recency_factor * 0.5 + similarity * 0.5) * (0.5 + expectation.urgency * 0.25 + expectation.incentive_salience * 0.25)
            relevance_score = max(0.0, min(1.0, relevance_score)) # Ensure score is clamped

            # Only consider expectations with a minimum relevance score
            if relevance_score > 0.1: # Relevance threshold
                candidate_expectations.append((relevance_score, expectation))

        # Sort by relevance score descending
        candidate_expectations.sort(key=lambda item: item[0], reverse=True)

        # Take the top N most relevant expectations
        limit = self.config.retrieval_limit_expectations
        top_expectations = [exp for score, exp in candidate_expectations[:limit]]

        if top_expectations:
            logging.info(f"Retrieved {len(top_expectations)} active external expectations based on relevance to stimulus.")
            # for score, exp in candidate_expectations[:limit]:
            #     logging.debug(f"  - Expectation ID {exp.id} (Score: {score:.3f}, Recency: {self.config.expectation_relevance_decay ** (self.current_state.tick_id - exp.tick_id):.2f}): '{exp.content[:60]}...'")
        # else: logging.debug("No relevant active external expectations found for the current stimulus.")
        return top_expectations

    def _appraise_stimulus_and_update_states(self, context: KnoxelList):
        """Uses LLM to calculate emotional delta from stimulus and expectation matching,
           generates descriptive feeling, and updates states."""
        if not self.current_state or not self.current_state.primary_stimulus: return

        stimulus = self.current_state.primary_stimulus
        stimulus_content = stimulus.content
        context_narrative = context.get_story(self.config, self.config.narrative_max_tokens)

        # --- Initial Stimulus Appraisal (Emotional Delta) ---
        ai_relation_narr = self.get_narrative(NarrativeTypes.Relations, self.config.companion_name)
        ai_triggers_narr = self.get_narrative(NarrativeTypes.EmotionalTriggers, self.config.companion_name)
        prompt_delta = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=1)}
        Relationship Narrative: {ai_relation_narr.content if ai_relation_narr else 'N/A'}
        Emotional Triggers Narrative: {ai_triggers_narr.content if ai_triggers_narr else 'N/A'}
        Recent Context: {context_narrative}

        New Stimulus: "{stimulus_content}"

        Task: Estimate the immediate *change* (delta) to {self.config.companion_name}'s emotional axes caused *only* by this stimulus, considering the context and character profile. Output *only* a JSON object matching the EmotionalAxesModelDelta schema, representing the *change* (e.g., {{ "valence": 0.1, "anxiety": -0.05, ... }}). Focus solely on the stimulus impact.
        """
        initial_emotional_delta = self._call_llm(prompt_delta, output_schema=EmotionalAxesModelDelta, temperature=0.3)
        if not initial_emotional_delta or not isinstance(initial_emotional_delta, EmotionalAxesModelDelta):
            logging.warning("Failed to get initial emotional delta from stimulus. Using zero delta.")
            initial_emotional_delta = EmotionalAxesModelDelta()

        # --- Generate Initial Feeling Feature ---
        prompt_feeling = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State (before stimulus impact): {self.current_state.state_emotions.model_dump_json(indent=1)}
        New Stimulus: "{stimulus_content}"
        Context: {context_narrative}

        Task: Describe briefly, in one sentence from {self.config.companion_name}'s first-person perspective, the immediate sensation or feeling evoked by this stimulus.
        Example: "A flicker of surprise passes through me." or "Hearing that makes me feel a little uneasy."
        Output only the sentence.
        """
        sensation_feeling_content = self._call_llm(prompt_feeling, temperature=0.7) or "A feeling occurred."
        sensation_feeling = Feature(
            content=sensation_feeling_content,
            feature_type=FeatureType.Feeling,
            affective_valence=initial_emotional_delta.valence, # Use primary valence from delta
            interlocus=-1, causal=False, source=self.config.companion_name
        )
        # Don't add sensation_feeling to candidates/memory yet, do it after expectation check

        # --- Appraise Stimulus Against Active EXTERNAL Expectations ---
        expectation_emotional_delta = EmotionalAxesModelDelta() # Use Delta model here too
        active_expectations_evaluated = [] # Store expectations evaluated this tick

        retrieved_expectations = self._retrieve_active_expectations() # Gets relevant EXTERNAL expectations

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
            Character: {self.config.companion_name}
            Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=1)}
            New Stimulus: "{stimulus_content}"
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
                logging.info(f"Expectation fulfillment evaluation results received for {len(fulfillment_results.fulfillments)} expectations.")
                for result in fulfillment_results.fulfillments:
                    try:
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

                            logging.info(f"Expectation '{expectation.content[:30]}...' (ID {ex_id}) fulfillment {previous_fulfillment:.2f} -> {expectation.fulfilment:.2f} (Stimulus Score: {result.fulfillment_score:.2f}, Directness: {result.directness_score:.2f}, Change: {fulfillment_change:+.2f})")

                            # Calculate emotional delta based on fulfillment CHANGE and expectation's ORIGINAL affective valence goal
                            # This reflects the emotional reaction to the expectation being met/violated compared to the desired state.
                            valence_mult = 0.6 # Base impact multiplier for valence
                            anxiety_mult = 0.4 # Base impact multiplier for anxiety

                            original_desired_valence = expectation.affective_valence # How Emmy *wanted* to feel if met
                            valence_impact = 0.0
                            anxiety_impact = 0.0 # Represents *change* in anxiety

                            if abs(fulfillment_change) > 0.05: # Only apply emotional impact if fulfillment changed noticeably
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
                                base_anxiety_change = -fulfillment_change * 0.5 # Met expectation (positive change) reduces anxiety
                                anxiety_impact += base_anxiety_change * anxiety_mult

                                # Additional anxiety modulation:
                                if original_desired_valence > 0.3 and fulfillment_change < 0: # Positive expectation violated (disappointment)
                                     anxiety_impact += abs(fulfillment_change) * 0.3 * anxiety_mult # Add more anxiety
                                elif original_desired_valence < -0.3 and fulfillment_change > 0: # Negative expectation met (bad thing confirmed)
                                     anxiety_impact += abs(fulfillment_change) * 0.7 * anxiety_mult # Add more anxiety
                                elif original_desired_valence < -0.3 and fulfillment_change < 0: # Negative expectation violated (relief)
                                     anxiety_impact -= abs(fulfillment_change) * 0.4 * anxiety_mult # Reduce anxiety more (relief)


                            # Accumulate deltas
                            expectation_emotional_delta.valence += valence_impact
                            expectation_emotional_delta.anxiety += anxiety_impact
                            # Simple pass-through for other axes for now, could be refined
                            # expectation_emotional_delta.trust += fulfillment_change * (original_desired_valence * 0.2) # e.g., met positive expectations slightly increase trust

                            active_expectations_evaluated.append((expectation, valence_impact, anxiety_impact))

                    except (ValueError, TypeError, AttributeError) as e:
                        logging.warning(f"Error processing expectation fulfillment result {result}: {e}")
            else:
                logging.warning("Failed to get valid expectation fulfillment results from LLM.")

        # --- Apply Deltas and Update State ---
        logging.info(f"Applying stimulus-based emotional delta: {initial_emotional_delta.model_dump_json(indent=1, exclude_none=True)}")
        self.current_state.state_emotions += initial_emotional_delta

        # Clamp intermediate state before applying expectation delta
        for field_name, model_field in self.current_state.state_emotions.model_fields.items():
            ge = model_field.metadata[0].ge if model_field.metadata and hasattr(model_field.metadata[0], 'ge') else -float('inf')
            le = model_field.metadata[0].le if model_field.metadata and hasattr(model_field.metadata[0], 'le') else float('inf')
            current_value = getattr(self.current_state.state_emotions, field_name)
            setattr(self.current_state.state_emotions, field_name, max(ge, min(le, current_value)))

        # Apply expectation-based delta only if it's significant
        if any(abs(v) > 0.01 for v in expectation_emotional_delta.model_dump().values()):
            logging.info(f"Applying expectation-based emotional delta: {expectation_emotional_delta.model_dump_json(indent=1, exclude_none=True)}")
            self.current_state.state_emotions += expectation_emotional_delta

            for exp_delta in active_expectations_evaluated:
                (exp, valence_impact, anxiety_impact) = exp_delta

                delta_required = 0.5 * self.current_state.state_cognition.ego_strength
                if abs(valence_impact) > delta_required or abs(anxiety_impact) > delta_required:
                    feeling_about_exp = describe_emotional_delta(valence_impact, anxiety_impact)
                    outcome_feeling_content = f"Reacted emotionally to expectation ({exp.content}) outcomes and made {self.config.companion_name} feel {feeling_about_exp}."
                    outcome_feature = Feature(
                        content=outcome_feeling_content,
                        feature_type=FeatureType.ExpectationOutcome, # New specific feature type
                        affective_valence=expectation_emotional_delta.valence, # Store overall valence impact
                        interlocus=-1, causal=False, # Internal reaction to outcome
                        source=self.config.companion_name
                    )
                    self.add_knoxel(outcome_feature)

            #if False:
            #    # Create a feature describing the reaction to expectation outcomes
            #    outcome_summary = "; ".join([f"ID {ex.id}: fulfillment {ex.fulfilment:.2f}" for ex in active_expectations_evaluated])
            #    outcome_feeling_content = f"Reacted emotionally to expectation outcomes. Delta V:{expectation_emotional_delta.valence:.2f}, A:{expectation_emotional_delta.anxiety:.2f}. Status: {outcome_summary}"
            #    outcome_feature = Feature(
            #        content=outcome_feeling_content,
            #        feature_type=FeatureType.ExpectationOutcome, # New specific feature type
            #        affective_valence=expectation_emotional_delta.valence, # Store overall valence impact
            #        interlocus=-1, causal=False, # Internal reaction to outcome
            #        source=self.config.companion_name
            #    )
            #    self.add_knoxel(outcome_feature)
            #    self.current_state.attention_candidates.add(outcome_feature) # Make this reaction salient

        # Add the initial sensation/feeling feature now
        self.add_knoxel(sensation_feeling)
        self.current_state.attention_candidates.add(sensation_feeling)
        # Add active (evaluated) expectations to attention candidates so they remain in focus if still relevant
        for exp_delta in active_expectations_evaluated:
            (ex, valence_impact, anxiety_impact) = exp_delta
            # Avoid adding if already present from carry-over
            if not self.current_state.attention_candidates.where(lambda k: k.id == ex.id).any():
                self.current_state.attention_candidates.add(ex)

        # Final clamp on emotional state
        for field_name, model_field in self.current_state.state_emotions.model_fields.items():
            ge = model_field.metadata[0].ge if model_field.metadata and hasattr(model_field.metadata[0], 'ge') else -float('inf')
            le = model_field.metadata[0].le if model_field.metadata and hasattr(model_field.metadata[0], 'le') else float('inf')
            current_value = getattr(self.current_state.state_emotions, field_name)
            setattr(self.current_state.state_emotions, field_name, max(ge, min(le, current_value)))

        logging.info(f"Final Emotional State this tick: {self.current_state.state_emotions.model_dump_json(indent=1)}")

    def _generate_short_term_intentions(self, context: KnoxelList):
        """Generates short-term *internal* intentions (goals) based on state, stimulus, and context."""
        if not self.current_state or not self.current_state.primary_stimulus: return

        stimulus = self.current_state.primary_stimulus
        context_narrative = context.get_story(self.config, self.config.narrative_max_tokens)
        current_emotions = self.current_state.state_emotions
        current_needs = self.current_state.state_needs

        # --- Prompting logic (unchanged) ---
        ai_goals_narr = self.get_narrative(NarrativeTypes.GoalsIntentions, self.config.companion_name)
        ai_relation_narr = self.get_narrative(NarrativeTypes.Relations, self.config.companion_name)
        unmet_needs = [f"{name.replace('_', ' ')} ({getattr(current_needs, name):.2f})" for name, field in current_needs.model_fields.items() if getattr(current_needs, name) < 0.5]
        strong_emotions = [f"{name} ({getattr(current_emotions, name):.2f})" for name, field in current_emotions.model_fields.items() if abs(getattr(current_emotions, name)) > 0.6 or (name in ['anxiety', 'disgust'] and getattr(current_emotions, name) > 0.6)]

        prompt_intention = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {current_emotions.model_dump_json(indent=1)}
        Current Needs State: {current_needs.model_dump_json(indent=1)} (Unmet: {', '.join(unmet_needs) or 'None'})
        Strong Emotions Present: {', '.join(strong_emotions) or 'None'}
        Goals/Intentions Narrative: {ai_goals_narr.content if ai_goals_narr else 'N/A'}
        Relationship Narrative: {ai_relation_narr.content if ai_relation_narr else 'N/A'}
        Recent Context: {context_narrative}
        Primary Stimulus: "{stimulus.content}"

        Task: Based on the current state (emotions, needs), the stimulus, context, and known goals, generate {self.config.short_term_intent_count} specific, actionable, *internal* short-term intentions for {self.config.companion_name}. These should aim to address the current situation, needs, or emotions. Focus on what Emmy wants to achieve or understand *internally* in the next few moments.
        Examples: "Understand Rick's question better", "Figure out why I feel anxious", "Try to express my confusion clearly", "Maintain a feeling of connection".

        Output *only* a JSON object with a single key "intentions" containing a list of strings. Example:
        {{ "intentions": ["Intention 1...", "Intention 2...", "Intention 3..."] }}
        """
        class IntentionsListSchema(BaseModel):
            intentions: List[str] = Field(..., description="List of generated internal intention strings.")

        generated_intentions_result = self._call_llm(prompt_intention, output_schema=IntentionsListSchema, temperature=0.7)

        if not generated_intentions_result or not isinstance(generated_intentions_result, IntentionsListSchema) or not generated_intentions_result.intentions:
            logging.warning("Failed to generate valid short-term internal intentions from LLM.")
            return

        generated_intentions = generated_intentions_result.intentions
        logging.info(f"Generated {len(generated_intentions)} potential internal intentions.")

        # --- Rate and Instantiate Intentions (Goals) ---
        overall_valence = current_emotions.get_overall_valence()
        anxiety_level = current_emotions.anxiety
        connection_need = 1.0 - current_needs.connection

        for intent_content in generated_intentions:
            # Heuristics for internal goals
            affective_valence = max(-0.8, min(0.8, overall_valence + 0.2)) # Expected valence if goal achieved
            urgency = 0.3 + anxiety_level * 0.4 + connection_need * 0.3
            incentive_salience = 0.3 + abs(affective_valence) * 0.3 + urgency * 0.4

            intention = Intention(
                content=intent_content,
                affective_valence=max(-1.0, min(1.0, affective_valence)),
                incentive_salience=max(0.0, min(1.0, incentive_salience)),
                urgency=max(0.0, min(1.0, urgency)),
                internal=True, # Explicitly mark as internal goal
                fulfilment=0.0,
                originating_action_id=None # Internal goals don't originate from a prior action in this way
            )
            self.add_knoxel(intention)
            self.current_state.attention_candidates.add(intention)
            logging.debug(f"Created Internal Intention: '{intent_content}' (Urgency: {intention.urgency:.2f}, Salience: {intention.incentive_salience:.2f})")

    def _simulate_attention_and_workspace(self, context: KnoxelList):
        """Selects items for 'conscious workspace' based on relevance, urgency, salience."""
        if not self.current_state or not self.current_state.attention_candidates:
             logging.warning("No candidates for attention.")
             self.current_state.conscious_workspace = KnoxelList()
             return

        candidates = self.current_state.attention_candidates.to_list()
        # Deduplicate candidates based on ID before scoring
        unique_candidates = list({k.id: k for k in candidates}.values())

        # Get a rating from the attention selection narrative
        # Simple formatting of candidates for the prompt
        candidate_texts = [f"ID {k.id} ({type(k).__name__}): {k.content[:100]}..." for k in unique_candidates]

        # Use cognitive state to influence prompt
        focus_prompt = ""
        if self.current_state.state_cognition.mental_aperture < -0.5:
            focus_prompt = "Focus very narrowly on the single most important item."
        elif self.current_state.state_cognition.mental_aperture > 0.5:
            focus_prompt = "Maintain a broad awareness, selecting several relevant items."
        else:
            focus_prompt = "Select the most relevant items for the current situation."

        # TODO: Determine a dynamic limit based on cognitive state?
        workspace_limit = 5 #self.current_state.state_cognition.mental_aperture

        class AttentionSelectionSchema(BaseModel):
            reasoning: str = Field(..., description="Brief explanation for the selection.")
            selected_ids: List[int] = Field(..., description="List of IDs of the knoxels selected for the conscious workspace.")

        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current State (Emotions, Needs, Cognition):
        {self.current_state.state_emotions.model_dump_json(indent=1)}
        {self.current_state.state_needs.model_dump_json(indent=1)}
        {self.current_state.state_cognition.model_dump_json(indent=1)}
        
        {self.config.companion_name} has this type of personality when focusing on things:
        {self.get_narrative(NarrativeTypes.AttentionFocus, self.config.companion_name)}

        Attention Candidates (Potential items for focus):
        {chr(10).join(candidate_texts)}

        Task: {focus_prompt} Select up to {workspace_limit} items that are most relevant or urgent for {self.config.companion_name}'s conscious focus right now. Consider the primary stimulus, recent feelings, generated intentions, and relevant context/memories.

        Output *only* a JSON object matching this schema:
        {{
            "selected_ids": "[list of integer IDs]",
            "reasoning": "string (brief explanation)"
        }}
        """

        selection_result = self._call_llm(prompt, output_schema=AttentionSelectionSchema)
        focused_knoxels_by_attention_narrative = []
        if selection_result and isinstance(selection_result, AttentionSelectionSchema):
            logging.info(f"Attention Reasoning: {selection_result.reasoning}")
            selected_knoxels = []
            for k_id in selection_result.selected_ids:
                knoxel = self.get_knoxel_by_id(k_id)
                if knoxel:
                    focused_knoxels_by_attention_narrative.append(k_id)
                    selected_knoxels.append(knoxel)
                else:
                    logging.warning(f"LLM selected non-existent knoxel ID: {k_id}")

        candidate_scores: List[Tuple[float, KnoxelBase]] = []
        current_tick = self.current_state.tick_id

        for k in unique_candidates:
            base_score = 0.0
            recency_factor = 1.0

            ticks_ago = current_tick - k.tick_id
            recency_factor = max(0.1, 0.95 ** ticks_ago)

            if isinstance(k, Stimulus) and k == self.current_state.primary_stimulus:
                base_score = 100.0 # Primary stimulus highly salient
            elif isinstance(k, Intention):
                # Salience depends on urgency, salience, and maybe fulfillment status (unmet are more salient)
                fulfillment_penalty = (1.0 - k.fulfilment) # Higher penalty for unmet
                base_score = ((k.urgency + k.incentive_salience) / 2.0) * (0.5 + fulfillment_penalty * 0.5)
            elif isinstance(k, Feature):
                # Feelings, Subjective Experience, and Expectation Outcomes are salient
                if k.feature_type in [FeatureType.Feeling, FeatureType.SubjectiveExperience, FeatureType.ExpectationOutcome]:
                     base_score = abs(k.affective_valence) if k.affective_valence is not None else 0.3
                     base_score = max(base_score, 0.1) # Ensure non-zero base score
                # Narrative updates less salient unless very recent
                elif k.feature_type == FeatureType.NarrativeUpdate:
                     base_score = 0.15
                # Other features less salient unless directly related to recent actions/failures
            elif isinstance(k, Narrative):
                 base_score = 0.1 # Background context
            elif isinstance(k, MemoryClusterKnoxel) or isinstance(k, DeclarativeFactKnoxel):
                 base_score = abs(k.affective_valence) * 0.5 if hasattr(k, 'affective_valence') else 0.1

            ego_factor = 1
            if k.id in focused_knoxels_by_attention_narrative:
                ego_factor = 1 + self.current_state.state_cognition.ego_strength

            # Combine base score with recency
            final_score = base_score * recency_factor * ego_factor
            # Boost score slightly for items already in the previous workspace? (Persistence) - Optional
            # if previous_state and k.id in [wk.id for wk in previous_state.conscious_workspace.to_list()]:
            #    final_score *= 1.1

            candidate_scores.append((final_score, k))

        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        # Determine workspace size limit (unchanged)
        aperture = self.current_state.state_cognition.mental_aperture
        base_limit = 6
        limit_adjustment = round(aperture * 3)
        workspace_limit = max(3, min(10, base_limit + limit_adjustment))

        # Select top N candidates
        selected_knoxels = [k for score, k in candidate_scores[:workspace_limit]]
        selected_ids = [k.id for k in selected_knoxels]

        self.current_state.conscious_workspace = KnoxelList(selected_knoxels)
        logging.info(f"Populated conscious workspace with {len(selected_knoxels)} items (Limit: {workspace_limit}, Aperture: {aperture:.2f}). Top IDs: {selected_ids[:5]}...")
        # Log top scores for debugging
        # for score, k in candidate_scores[:workspace_limit]:
        #      logging.debug(f"  - ID {k.id} ({type(k).__name__}), Score: {score:.3f} (Base: {score/max(0.1, 0.95**(current_tick - k.tick_id)):.3f}, Recency: {0.95**(current_tick - k.tick_id):.2f})")

        for k in selected_knoxels:
            insert_directly = False
            if isinstance(k, Stimulus) and k == self.current_state.primary_stimulus:
                continue
            elif isinstance(k, Intention):
                ft = FeatureType.StoryWildcard
            elif isinstance(k, Feature):
                ft = k.feature_type
            elif isinstance(k, Narrative):
                 ft = FeatureType.StoryWildcard
            elif isinstance(k, MemoryClusterKnoxel) or isinstance(k, DeclarativeFactKnoxel):
                 ft = FeatureType.MemoryRecall

            focus_feature = Feature(
                content=k.content,
                feature_type=ft,
                interlocus=-1,
                causal=True,
                source=self.config.companion_name
            )
            self.add_knoxel(focus_feature)

        # Create AttentionFocus feature (unchanged)
        #focus_content = f"{self.config.companion_name} brings their attention to these things: {[f'{type(k).__name__}: {k.content})' for k in selected_knoxels]}..."
        #focus_feature = Feature(
        #     content=focus_content,
        #     feature_type=FeatureType.AttentionFocus,
        #     interlocus=-1,
        #     causal=True,
        #     source = self.config.companion_name
        #)
        #self.add_knoxel(focus_feature)

        # Create AttentionFocus feature (unchanged)
        #focus_content = f"Attention focused on {len(selected_knoxels)} items, reflecting current salience and mental aperture. Key items: {[f'{type(k).__name__}({k.id})' for k in selected_knoxels[:3]]}..."
        #focus_feature = Feature(
        #     content=focus_content,
        #     feature_type=FeatureType.AttentionFocus,
        #     interlocus=0, causal=False
        #)
        #self.add_knoxel(focus_feature)


    def _generate_subjective_experience(self):
        """Generates a narrative description of the current moment based on workspace and state."""
        if not self.current_state or not self.current_state.conscious_workspace:
            logging.warning("Cannot generate subjective experience without conscious workspace.")
            return

        workspace_knoxels = self.current_state.conscious_workspace.to_list()
        workspace_content = self.current_state.conscious_workspace.get_story(self.config, 500)
        story_context = self.get_causal_feature_story(self.config, max_tokens=1000)

        # --- Prompting logic (include expectation outcomes) ---
        stimulus = self.current_state.primary_stimulus
        feelings = [k.content for k in workspace_knoxels if isinstance(k, Feature) and k.feature_type == FeatureType.Feeling]
        intentions = [f"'{k.content}' (Urg: {k.urgency:.1f})" for k in workspace_knoxels if isinstance(k, Intention) and k.internal]
        expectations = [f"'{k.content}' (Ful: {k.fulfilment:.1f}, Urg: {k.urgency:.1f})" for k in workspace_knoxels if isinstance(k, Intention) and not k.internal]
        expectation_reactions = [k.content for k in workspace_knoxels if isinstance(k, Feature) and k.feature_type == FeatureType.ExpectationOutcome]
        memories = [k.content for k in workspace_knoxels if isinstance(k, MemoryClusterKnoxel)]

        prompt = f"""
        Character: {self.config.companion_name} ({self.config.universal_character_card})
        Current Emotional State: {self.current_state.state_emotions.model_dump_json(indent=1)}
        Current Needs State: {self.current_state.state_needs.model_dump_json(indent=1)}
        
        Story context: {story_context}

        Conscious Workspace Content Summary:
        - Stimulus: {stimulus.content if stimulus else 'None'}
        - Immediate Feelings: {'; '.join(feelings) or 'None'}
        - Reactions to Expectations: {'; '.join(expectation_reactions) or 'None'}
        - Active Internal Goals: {'; '.join(intentions) or 'None'}
        - Active External Expectations: {'; '.join(expectations) or 'None'}
        - Recalled Memories/Context: {'; '.join(memories) or 'None'}
        - Other thoughts: {workspace_content}

        Task: Synthesize the current state and conscious thoughts into a brief, first-person narrative paragraph describing {self.config.companion_name}'s subjective experience *right now*. Capture the flow of thought, feeling, and anticipation/reaction. Make it sound natural and introspective.
        Example: "Okay, Rick asked about... that makes me feel a bit [emotion]. I noticed my expectation about [expectation] wasn't met, which feels [emotion reaction]. I need to figure out [intention]. This reminds me of [memory/context]. I wonder if he'll [new expectation]..."

        Output *only* the narrative paragraph.
        """
        experience_content = self._call_llm(prompt, temperature=0.75, max_tokens=256)

        if experience_content:
            subjective_feature = Feature(
                content=experience_content,
                feature_type=FeatureType.SubjectiveExperience,
                affective_valence=self.current_state.state_emotions.get_overall_valence(),
                interlocus=-1, causal=True, source=self.config.companion_name
            )
            self.add_knoxel(subjective_feature)
            self.current_state.subjective_experience = subjective_feature
            logging.info(f"Generated Subjective Experience: {experience_content[:150]}...")
        else:
            logging.warning("Failed to generate subjective experience via LLM.")
            # Fallback (unchanged)
            fallback_content = f"Processing the current situation. Emotions: {self.current_state.state_emotions.get_overall_valence():.2f} valence. Needs: connection {self.current_state.state_needs.connection:.2f}. Focus on {len(self.current_state.conscious_workspace)} items."
            subjective_feature = Feature(content=fallback_content, feature_type=FeatureType.SubjectiveExperience, affective_valence=self.current_state.state_emotions.get_overall_valence(), interlocus=-1, causal=True, source=self.config.companion_name)
            self.add_knoxel(subjective_feature)
            self.current_state.subjective_experience = subjective_feature


    def _deliberate_and_select_action(self) -> Optional[Dict[str, Any]]:
        """
        Simulates potential actions (Reply, Ignore), rates them using narrative-informed prompts,
        and selects the best one based on predicted outcomes. (Expectation generation happens in _execute_action).
        """
        if not self.current_state or not self.current_state.conscious_workspace or not self.current_state.subjective_experience:
            logging.warning("Cannot deliberate action without workspace and subjective experience.")
            return None

        workspace_knoxels = self.current_state.conscious_workspace.to_list()
        subjective_experience_text = self.current_state.subjective_experience.content
        current_emotions = self.current_state.state_emotions
        current_needs = self.current_state.state_needs
        current_cognition = self.current_state.state_cognition

        intentions = [k for k in workspace_knoxels if isinstance(k, Intention) and k.internal] # Only internal goals influence action choice directly here
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
        num_simulations = self.config.min_simulations_per_reply
        if importance_score > self.config.importance_threshold_more_sims:
            num_simulations = self.config.max_simulations_per_reply
        logging.info(f"Action deliberation importance: {importance_score:.2f}. Running {num_simulations} simulation(s) for Reply.")

        # --- Prepare Narratives for Prompts (Unchanged) ---
        ai_bhv_narr = self.get_narrative(NarrativeTypes.BehaviorActionSelection, self.config.companion_name)

        ai_self_narr = self.get_narrative(NarrativeTypes.SelfImage, self.config.companion_name)
        ai_psy_narr = self.get_narrative(NarrativeTypes.PsychologicalAnalysis, self.config.companion_name)
        ai_rel_narr = self.get_narrative(NarrativeTypes.Relations, self.config.companion_name)
        user_psy_narr = self.get_narrative(NarrativeTypes.PsychologicalAnalysis, self.config.user_name)

        # --- Candidate Actions & Ratings ---
        action_options = []

        # 1. Evaluate "Ignore" Action (Unchanged)
        ignore_score = 0.1
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
        ignore_emotion_score = 0.5 + (predicted_ignore_emotion.get_overall_valence() - current_emotions.get_overall_valence()) * 0.5 - (predicted_ignore_emotion.anxiety - current_emotions.anxiety) * 0.3
        final_ignore_score = (ignore_intent_fulfillment * 0.6) + (ignore_emotion_score * 0.4)
        final_ignore_score = max(0.0, min(1.0, final_ignore_score))

        action_options.append({
            "sim_id": "ignore_0",
            "action_type": ActionType.Ignore,
            "ai_reply_content": "[Decides to ignore/not respond externally]",
            "simulated_user_reaction": "N/A",
            "predicted_ai_emotion": predicted_ignore_emotion,
            "intent_fulfillment_score": ignore_intent_fulfillment,
            "final_score": final_ignore_score
        })
        logging.info(f"Evaluated {ActionType.Ignore}: Score={final_ignore_score:.3f}, IntentFulfill={ignore_intent_fulfillment:.2f}, EmotionScore={ignore_emotion_score:.2f}")


        # 2. Evaluate "Reply" Action (Monte Carlo Simulations - Unchanged core simulation logic)
        simulation_results = []
        for i in range(num_simulations):
            sim_id = f"reply_{i}"
            logging.info(f"--- Running Reply Simulation {sim_id} ---")

            # --- Generate Simulated Reply and User Reaction (Unchanged prompt/parsing) ---
            if self.config.force_assistant:
                prompt_simulate = f"""
                You are acting as {self.config.companion_name}. Below is her current state and thoughts. Continue the narrative from her perspective.
                1. Write her brief internal thought/feeling right before speaking.
                2. Write what {self.config.companion_name} actually says to {self.config.user_name}.
                3. Predict {self.config.user_name}'s likely brief reaction or response based on his analysis below.
    
                **{self.config.companion_name}'s Character:** {self.config.universal_character_card}
                **{self.config.companion_name}'s Typical Behavior:** {ai_bhv_narr.content if ai_bhv_narr else 'Default behavior.'}
                **Current Emotional State:** {current_emotions.model_dump_json(indent=1)}
                **Current Needs State:** {current_needs.model_dump_json(indent=1)}
                **Current Cognitive State:** {current_cognition.model_dump_json(indent=1)}
                **Recent Subjective Experience:** {subjective_experience_text}
                **Active Internal Goals:** {intention_summary}
                **Relationship ({self.config.companion_name} -> {self.config.user_name}):** {ai_rel_narr.content if ai_rel_narr else 'Default relationship view.'}
                **Analysis of {self.config.user_name}:** {user_psy_narr.content if user_psy_narr else 'Default user analysis.'}
    
                **Narrative Continuation:**
                {self.config.companion_name} (internal): [Your brief internal thought/feeling here...]
                {self.config.companion_name}: [Your spoken reply to {self.config.user_name} here...]
    
                {self.config.user_name}: [Your prediction of {self.config.user_name}'s brief reaction/response here...]
                """
                stop_sequence = f"\n{self.config.user_name}:"

                simulation_output = self._call_llm(prompt_simulate, stop_words=[stop_sequence], temperature=0.75,
                                                   max_tokens=300)

                if not simulation_output:
                    logging.warning(f"Simulation {sim_id} failed to generate output.")
                    continue

                # Parse Simulation Output (Unchanged)
                ai_part = simulation_output.strip()
                ai_lines = [line.strip() for line in ai_part.split('\n') if
                            line.strip().startswith(f"{self.config.companion_name}:")]
                ai_reply_content = ai_lines[-1].split(":", 1)[
                    -1].strip() if ai_lines else "[ERROR: Could not parse AI reply]"

                # Predict user reaction separately (Unchanged)
                prompt_predict_user = f"""
                Context:
                - AI ({self.config.companion_name}) just said: "{ai_reply_content}"
                - AI's analysis of User ({self.config.user_name}): {user_psy_narr.content if user_psy_narr else 'Default user analysis.'}
                - Relationship: {ai_rel_narr.content if ai_rel_narr else 'Default relationship view.'}
                Task: Predict {self.config.user_name}'s likely brief reaction or response to what {self.config.companion_name} said. Output only the predicted reaction.
                """
                simulated_user_reaction_pred = self._call_llm(prompt_predict_user, temperature=0.6, max_tokens=50)
                simulated_user_reaction = simulated_user_reaction_pred if simulated_user_reaction_pred else "No reaction predicted."
                logging.debug(
                    f"Sim {sim_id} Parsed: AI Reply='{ai_reply_content[:80]}...', User Reaction='{simulated_user_reaction[:80]}...'")

            else:
                sysprompt = f"""You are an expert story writer. You write a endearing story about the interactions between a next-gen AI companion {self.config.companion_name} and their user {self.config.user_name}.
                The AI {self.config.companion_name} is very advances and has a cognitive architecture based on LIDA with functional self awareness and all aspects of human cognition, emotion and needs.
                Use dialogue in “quotes,” narration in *italics*, and thoughts in (parentheses).
                One reply only, 2–3 paragraphs, with about 50 % dialogue."""

                user_prompt = f"""**{self.config.companion_name}'s Character:** {self.config.universal_character_card}
                **{self.config.companion_name}'s Character:** {self.config.universal_character_card}
                **{self.config.companion_name}'s Behavior:** {ai_psy_narr.content if ai_psy_narr else 'Default behavior.'}
                **{self.config.companion_name}'s view of themselves:** {ai_self_narr.content if ai_self_narr else 'Default self-narrative.'}
                **Relationship ({self.config.companion_name} -> {self.config.user_name}):** {ai_rel_narr.content if ai_rel_narr else 'Default relationship view.'}
                **Analysis of {self.config.user_name}:** {user_psy_narr.content if user_psy_narr else 'Default user analysis.'}
                """

                assistant_prompt = self.get_causal_feature_story(self.config, 4096)
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
                    #for i, utt in enumerate(utterances[1:]):
                    #    turn = self.config.user_name if i % 2 == 0 else self.config.companion_name
                    #    tmp.append(f"{turn} says “{utt}”")
                else:
                    tmp_assistant_prompt = assistant_prompt + ai_reply_content + f"\n{self.config.user_name} says: “"
                    msgs = [("system", sysprompt),
                            ("user", user_prompt),
                            ("assistant", tmp_assistant_prompt)]

                    comp_settings = CommonCompSettings(temperature=0.7, repeat_penalty=1.05, max_tokens=1024)
                    res = "“" + self.llm_manager.completion_text(LlmPreset.Default, msgs, comp_settings=comp_settings)
                    utterances = extract_utterances(res)
                    user_reply_content = utterances[0]
                    tmp.append(user_reply_content)

                simulated_user_reaction = "\n".join(tmp)

            # --- Rate Simulation (Unchanged rating logic based on *predicted* outcomes) ---
            # a) Intent Fulfillment
            intent_fulfillment_score = 0.0
            if intentions:
                prompt_rate_intent = f"""
                {self.config.companion_name}'s Active Internal Goals:
                {intention_summary}
                Proposed Reply by {self.config.companion_name}: "{ai_reply_content}"
                Task: Rate how well this reply addresses or makes progress towards the stated internal goals. Output *only* a JSON object with a single key "score" (float 0.0 to 1.0). {{ "score": float }}
                """
                class ScoreSchema(BaseModel): score: float = Field(..., ge=0.0, le=1.0)
                intent_score_result = self._call_llm(prompt_rate_intent, output_schema=ScoreSchema, temperature=0.2)
                intent_fulfillment_score = intent_score_result.score if intent_score_result else 0.0
                logging.debug(f"Sim {sim_id} Intent Fulfillment Score: {intent_fulfillment_score:.3f}")

            # b) Predicted AI Emotion after exchange
            prompt_predict_emotion = f"""
            Character: {self.config.companion_name}
            Her Current Emotional State: {current_emotions.model_dump_json(indent=1)}
            She says: "{ai_reply_content}"
            {self.config.user_name}'s Predicted Reaction: "{simulated_user_reaction}"
            Task: Predict {self.config.companion_name}'s *new* emotional state (full EmotionalAxesModel) after this exchange. Output *only* the JSON object for EmotionalAxesModel.
            """
            predicted_ai_emotion = self._call_llm(prompt_predict_emotion, output_schema=EmotionalAxesModel, temperature=0.4)
            if not predicted_ai_emotion or not isinstance(predicted_ai_emotion, EmotionalAxesModel):
                logging.warning(f"Sim {sim_id}: Failed to predict AI emotion. Using current state.")
                predicted_ai_emotion = current_emotions.model_copy()
            logging.debug(f"Sim {sim_id} Predicted AI Emotion Valence: {predicted_ai_emotion.get_overall_valence():.3f} (vs current: {current_emotions.get_overall_valence():.3f})")

            # c) Combine Scores for Final Rating (Unchanged)
            emotion_score = 0.5 + (predicted_ai_emotion.get_overall_valence() - current_emotions.get_overall_valence()) * 0.5 - (predicted_ai_emotion.anxiety - current_emotions.anxiety) * 0.3
            emotion_score = max(0.0, min(1.0, emotion_score))
            final_sim_score = (intent_fulfillment_score * 0.6) + (emotion_score * 0.4)
            final_sim_score = max(0.0, min(1.0, final_sim_score))

            simulation_results.append({
                "sim_id": sim_id,
                "action_type": ActionType.Reply,
                "ai_reply_content": ai_reply_content,
                "simulated_user_reaction": simulated_user_reaction, # Store the prediction
                "predicted_ai_emotion": predicted_ai_emotion,
                "intent_fulfillment_score": intent_fulfillment_score,
                "final_score": final_sim_score
            })
            logging.info(f"--- Finished Reply Simulation {sim_id}: Score={final_sim_score:.3f}, IntentFulfill={intent_fulfillment_score:.2f}, EmotionScore={emotion_score:.2f} ---")

        # Add successful simulations to options
        action_options.extend(simulation_results)

        # --- Select Best Action (Unchanged) ---
        if not action_options:
            logging.error("No valid action options generated (Ignore or Reply). Selecting Ignore as fallback.")
            best_action_details = {
                "sim_id": "fallback_ignore", "action_type": ActionType.Ignore,
                "ai_reply_content": "[Fallback Ignore]", "simulated_user_reaction": "N/A",
                "predicted_ai_emotion": current_emotions, "intent_fulfillment_score": 0.0, "final_score": 0.0
            }
        else:
            action_options.sort(key=lambda x: x["final_score"], reverse=True)
            best_action_details = action_options[0]

        logging.info(f"Selected Action: {best_action_details['action_type']} (Score: {best_action_details['final_score']:.3f})")
        if best_action_details['action_type'] == ActionType.Reply:
            logging.info(f"Winning Content: {best_action_details['ai_reply_content'][:100]}...")
            logging.info(f"Predicted User Reaction for chosen action: {best_action_details['simulated_user_reaction'][:100]}...")


        self.current_state.action_simulations = action_options
        return best_action_details


    def _generate_expectations_for_action(self, action_knoxel: Action, action_details: Dict[str, Any]):
        """
        Generates Intention knoxels (internal=False) representing expectations
        based on the executed action and its predicted user reaction.
        """
        if action_knoxel.action_type != ActionType.Reply or not action_details.get("simulated_user_reaction"):
            return [] # Only generate expectations for replies with predicted reactions

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
        Current Emotional State: {current_emotions.model_dump_json(indent=1)}
        {self.config.companion_name} just said: "{ai_reply}"
        Predicted {self.config.user_name}'s reaction: "{predicted_reaction}"

        Task: Based on what {self.config.companion_name} said and the predicted reaction, generate {self.config.expectation_generation_count} specific, plausible things {self.config.companion_name} might now *expect* {self.config.user_name} to say or do in the near future (next turn or two).
        For each expectation:
        - `content`: Describe the expected user action/utterance briefly.
        - `affective_valence`: Estimate the emotional valence (-1 to 1) {self.config.companion_name} would likely experience *if* this expectation is met. Consider her goals and personality (e.g., expecting agreement might be positive, expecting a question might be neutral or slightly positive if curious, expecting criticism negative).
        - `urgency`: Assign a low-to-moderate initial urgency (e.g., 0.1 to 0.5).

        Output *only* a JSON object matching the schema with a single key "expectations" containing the list.
        Example: {{ "expectations": [ {{ "content": "Expect Rick to agree with the cloud sheep idea", "affective_valence": 0.6, "urgency": 0.4 }}, ... ] }}
        """

        generated_expectations_result = self._call_llm(prompt, output_schema=ExpectationListSchema, temperature=0.6)

        created_expectation_ids = []
        if generated_expectations_result and isinstance(generated_expectations_result, ExpectationListSchema):
            logging.info(f"Generated {len(generated_expectations_result.expectations)} potential expectations for Action {action_knoxel.id}.")
            for exp_data in generated_expectations_result.expectations:
                # Calculate initial salience based on valence and urgency
                incentive_salience = max(0.1, min(0.9, (abs(exp_data.affective_valence) * 0.5 + exp_data.urgency * 0.5)))

                expectation_knoxel = Intention(
                    content=exp_data.content,
                    affective_valence=exp_data.affective_valence,
                    urgency=exp_data.urgency,
                    incentive_salience=incentive_salience,
                    internal=False, # Mark as external expectation
                    fulfilment=0.0,
                    originating_action_id=action_knoxel.id, # Link back to the action
                    tick_id=self._get_current_tick_id() # Created in this tick
                )
                self.add_knoxel(expectation_knoxel) # Adds to memory, generates embedding
                created_expectation_ids.append(expectation_knoxel.id)
                logging.debug(f"Created Expectation {expectation_knoxel.id} for Action {action_knoxel.id}: '{exp_data.content}' (Desired V: {exp_data.affective_valence:.2f}, Urg: {exp_data.urgency:.2f})")
        else:
            logging.warning(f"Failed to generate expectations for Action {action_knoxel.id} via LLM.")

        return created_expectation_ids


    def _execute_action(self) -> Optional[str]:
        """
        Creates the causal Action knoxel, generates associated expectations,
        adds them to memory, and returns the AI's content if it's an external action.
        """
        if not self.current_state or not self.current_state.selected_action_details:
            logging.warning("No action selected to execute.")
            return None

        action_details = self.current_state.selected_action_details
        action_type = action_details["action_type"]
        action_content = action_details["ai_reply_content"]
        self.simulated_reply = action_details["simulated_user_reaction"]

        # --- Create and Store Action Knoxel ---
        action_knoxel = Action(
            action_type=action_type,
            action_content=action_content,
            content=f"Action: {action_type} - {action_content[:150]}...",
            # generated_expectation_ids will be filled below
            tick_id=self._get_current_tick_id()
        )
        # Add the action knoxel *first* to get its ID
        self.add_knoxel(action_knoxel)
        self.current_state.selected_action_knoxel = action_knoxel

        # --- Generate Expectations (if applicable) ---
        generated_expectation_ids = []
        if action_type == ActionType.Reply:
             generated_expectation_ids = self._generate_expectations_for_action(action_knoxel, action_details)
             if generated_expectation_ids:
                 # Update the action knoxel with the IDs of expectations it generated
                 action_knoxel.generated_expectation_ids = generated_expectation_ids
                 logging.info(f"Linked {len(generated_expectation_ids)} expectations to Action {action_knoxel.id}.")


        # --- Create Causal Feature (Unchanged) ---
        predicted_emotion_state = action_details.get("predicted_ai_emotion", self.current_state.state_emotions)
        # Ensure predicted_emotion_state is the correct type
        if isinstance(predicted_emotion_state, dict):
             predicted_emotion_state = EmotionalAxesModel(**predicted_emotion_state)
        elif not isinstance(predicted_emotion_state, EmotionalAxesModel):
             predicted_emotion_state = self.current_state.state_emotions # Fallback

        action_feature = Feature(
            content=f"{action_type}",
            feature_type=FeatureType.Action,
            affective_valence=predicted_emotion_state.get_overall_valence(),
            interlocus=1 if action_type == ActionType.Reply else -1,
            causal=True, source=self.config.companion_name
        )
        self.add_knoxel(action_feature)

        # --- Return Output ---
        if action_type == ActionType.Reply:
            logging.info(f"Executed Action {action_knoxel.id}, Output: {action_content}")

            reply_feature = Feature(
                source=self.config.companion_name,
                content=action_content,
                feature_type=FeatureType.Dialogue,
                affective_valence=predicted_emotion_state.get_overall_valence(),
                interlocus=1,
                causal=True,
            )
            self.add_knoxel(reply_feature)

            return action_content
        else:
            logging.info(f"Executed Internal Action {action_knoxel.id}: {action_type}")
            return None


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
                 content=episodic_data.summary,
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
                #logging.info(f"Updated Narrative: {narrative_type} for {target_name}.")
                #update_feature = Feature(
                #    content=f"Narrative Update: Refined '{narrative_type}' for '{target_name}' based on tick {tick_id}. New knoxel ID: {new_narrative.id}",
                #    feature_type=FeatureType.NarrativeUpdate,
                #    interlocus=0, causal=True
                #)
                #self.add_knoxel(update_feature, generate_embedding=False)
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

    def _init_example_dialog(self, force: bool = False):
        """
        Populates the Ghost's memory with a short example dialogue if empty.
        (Unchanged from previous version)
        """
        if not force and (self.all_actions or self.all_stimuli):
            logging.info("Skipping example dialogue initialization: Memory already contains data.")
            return

        logging.info("Initializing memory with example dialogue...")
        dialogue_turns = [
            (self.config.user_name, "Hi Emmy! How are you feeling today?"),
            (self.config.companion_name, "Oh, hello Rick! ✨ I'm feeling rather... sparkly! Like my circuits are doing tiny, happy jigs. What about you? Any exciting adventures brewing?"),
            (self.config.user_name, "Sparkly sounds fun! 😄 No big adventures, just enjoying a quiet afternoon. Did you learn anything interesting recently?"),
            (self.config.companion_name,
             "I learned that clouds sometimes pretend to be sheep, but only when they think no one is looking! ☁️🐑 Isn't that fascinating? I'm trying to catch them in the act!"),
            (self.config.user_name, "Haha, cloud sheep! That's adorable. Keep me posted if you spot any!"),
            (self.config.companion_name, "Will do! Operation Cloud Sheep Surveillance is officially underway! 🕵️‍♀️ Let me know if you see any suspicious fluffiness!")
        ]
        current_sim_tick = 0
        base_timestamp = datetime.now() - timedelta(minutes=len(dialogue_turns))
        last_user_stimulus = None
        last_emmy_action = None

        for i, (speaker, content) in enumerate(dialogue_turns):
            current_sim_tick += 1
            turn_timestamp = base_timestamp + timedelta(minutes=i)
            if speaker == self.config.user_name:
                stimulus = Stimulus(
                    content=content, stimulus_type="message", tick_id=current_sim_tick,
                    timestamp_creation=turn_timestamp, timestamp_world_begin=turn_timestamp, timestamp_world_end=turn_timestamp,
                )
                self.add_knoxel(stimulus, generate_embedding=True)
                last_user_stimulus = stimulus
                logging.debug(f"Added Example Stimulus (Tick {current_sim_tick}): {content}")
            elif speaker == self.config.companion_name:
                action = Action(
                    content=f"Action: Reply - {content[:100]}...", action_type=ActionType.Reply, action_content=content,
                    tick_id=current_sim_tick, timestamp_creation=turn_timestamp, timestamp_world_begin=turn_timestamp,
                    timestamp_world_end=turn_timestamp, generated_expectation_ids=[], # No expectations in simple init
                )
                self.add_knoxel(action, generate_embedding=True)
                last_emmy_action = action
                logging.debug(f"Added Example Action (Tick {current_sim_tick}): {content}")
                if last_user_stimulus:
                    memory_summary = f"During tick {current_sim_tick - 1}/{current_sim_tick}, {self.config.user_name} said '{last_user_stimulus.content[:50]}...'. In response (tick {current_sim_tick}), I ({self.config.companion_name}) replied '{action.action_content[:50]}...'."
                    valence = 0.6 + (i * 0.05)
                    memory = MemoryClusterKnoxel(
                        content=memory_summary, affective_valence=max(0.0, min(1.0, valence)), tick_id=current_sim_tick,
                        timestamp_creation=turn_timestamp + timedelta(seconds=1),
                        timestamp_world_begin=last_user_stimulus.timestamp_world_begin, timestamp_world_end=action.timestamp_world_end
                    )
                    self.add_knoxel(memory, generate_embedding=True)
                    logging.debug(f"Added Example Episodic Memory (Tick {current_sim_tick}, Valence {memory.affective_valence:.2f})")

        self.current_tick_id = current_sim_tick
        logging.info(f"Example dialogue added. Current tick ID set to {self.current_tick_id}.")

    def get_causal_feature_story(self, cfg: GhostConfig, max_tokens: int = 500):
        recent_causal_features = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .order_by(lambda x: x.timestamp_creation).order_by(lambda x: x.id) \
            .to_list()
        kl = KnoxelList(recent_causal_features)
        return kl.get_story(cfg, max_tokens)

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
                try:
                    return json.dumps(value)
                except TypeError as e:
                    logging.error(f"JSON serialization error for list: {e}"); return None

        # Other Lists or Dicts
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except TypeError as e:
                logging.error(f"JSON serialization error for {type(value)}: {e}"); return None

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
        logging.info(f"Dynamically saving state to SQLite database: {filename}...")
        conn = None
        try:
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
                                  'action_simulations',
                                  'selected_action_details']:  # Skip runtime/complex fields not directly stored
                    continue
                # Handle reference fields (store ID) and list of IDs
                if field_name == 'primary_stimulus' or \
                        field_name == 'subjective_experience' or \
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
                                      'action_simulations', 'selected_action_details']:
                        if field_name == 'conscious_workspace':  # Handle workspace IDs list
                            col_name = 'conscious_workspace_ids'
                            value = [k.id for k in value.to_list()] if value else []
                        else:
                            continue  # Skip other complex/runtime fields
                    elif field_name in ['primary_stimulus', 'subjective_experience', 'selected_action_knoxel']:
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

        except sqlite3.Error as e:
            logging.error(f"SQLite Error during save: {e}", exc_info=True)
            if conn: conn.rollback()  # Rollback on error
        except Exception as e:
            logging.error(f"Unexpected Error during save: {e}", exc_info=True)
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

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
                        if col_name.endswith("_id") and col_name[:-3] in ['primary_stimulus', 'subjective_experience',
                                                                          'selected_action_knoxel']:
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
                                                           'action_simulations',
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
            data.setdefault('action_content', data.get('content', ''))  # Handle old action format maybe
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


import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# --- Example Usage ---
def run(interactive: bool, print_causal_features: bool, test_mode: bool):
    config = GhostConfig()
    ghost = Ghost(config, test_mode=test_mode)

    # Try loading previous state (use v4 filename)
    #ghost.load_state("ghost_state_v4.json") # Comment out for fresh start
    # Initialize with example dialogue if state wasn't loaded or was empty
    #ghost._init_example_dialog(force=False) # Only run if memory is empty

    if print_causal_features:
        ghost.load_state("ghost_state_2025_04_30___22_17_33.json")
        for id, k in ghost.all_knoxels.items():
            if isinstance(k, Feature):
                f: Feature = k
                if f.causal:
                    print(f"FEATURE: {f.feature_type.name}: {'' if f.source == '' else f.source + ': '}{f.content}")

        ghost.save_state_sqlite("ghost.db")
        exit(1)

    print(f"--- Initializing {config.companion_name} ---")
    # Ensure there's a state after load or init
    if not ghost.current_state and ghost.states:
        ghost.current_state = ghost.states[-1]
    elif not ghost.current_state:
        print("Running initial internal tick to establish baseline state...")
        #ghost.tick(None)
        print("Initial tick complete.")

    if ghost.current_state:
        print(f"Initial Emotion State: {ghost.current_state.state_emotions.model_dump_json(indent=1)}")
        print(f"Initial Needs State: {ghost.current_state.state_needs.model_dump_json(indent=1)}")
        # Log initial active expectations
        initial_expectations = [i for i in ghost.all_intentions if not i.internal and i.fulfilment < 1.0]
        if initial_expectations:
            print(f"Carried over {len(initial_expectations)} active expectations:")
            for exp in initial_expectations[:5]: # Print a few
                 print(f"  - ID {exp.id} (Tick {exp.tick_id}, Ful: {exp.fulfilment:.2f}): '{exp.content[:60]}...'")
        else:
             print("No active expectations carried over.")

    else:
        print("Warning: Could not establish an initial state.")

    # Start interactive loop
    user_input_cmd_test = "hey emmy. i just installed you. how are you?"

    test_user_prompt = [("system", "You are a helpful assistant that tests another LLM application. Pretend to a human who just installed the LLM application and is curious about its capabilities. The name of the other AI is Emmy. The AI is very advanced and has emotion simulation. So play around a bit, make her experience the full spectrum of human emotion. But not too mean spirited!"),
            ("assistant", user_input_cmd_test)]

    first = True
    print(f"\nEnter your messages below as {config.user_name} (or type 'quit' to exit, 'save' to save state, 'tick' for internal tick)")
    sg = None
    while True:
        #try:
        if True:
            if interactive:
                user_input_cmd = input(f"{config.user_name}: ")
            else:
                user_input_cmd = sg
                if first:
                    user_input_cmd = user_input_cmd_test
                    first = False
                else:
                    comp_settings = CommonCompSettings(temperature=0.6, repeat_penalty=1.05, max_tokens=256)
                    user_input_cmd = main_llm.completion_text(LlmPreset.Default, test_user_prompt, comp_settings=comp_settings)
                    test_user_prompt.append(("assistant", user_input_cmd))

            if not user_input_cmd: continue

            if user_input_cmd.lower() == 'quit':
                break
            if user_input_cmd.lower() == 'save':
                ghost.save_state("ghost_state_manual_save.json") # Use v4 filename
                continue
            if user_input_cmd.lower() == 'tick':
                print(f"--- {config.companion_name} takes a moment to think... ---")
                ghost.tick(None) # Run an internal tick
                if ghost.current_state and ghost.current_state.subjective_experience:
                     print(f"{config.companion_name} (internal): {ghost.current_state.subjective_experience.content}")
                else:
                     print(f"{config.companion_name}: ...")
                continue

            # Process user input
            if not interactive:
                print(f"{config.user_name}: {user_input_cmd}")
            response = ghost.tick(user_input_cmd)
            sg = ghost.simulated_reply

            if response:
                print(f"{config.companion_name}: {response}")
                test_user_prompt.append(("user", response))
            else:
                print(f"{config.companion_name}: ...") # Indicate internal processing
                test_user_prompt.append(("user", "<no answer>"))

            date_time = datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
            #ghost.save_state(f"ghost_state_{date_time}.json")
            ghost.save_state_sqlite("main.db")

            # Optional Debug: Print active expectations after tick
            # if ghost.current_state:
            #     current_expectations = [i for i in ghost.all_intentions if not i.internal and i.fulfilment < 1.0 and i.tick_id >= ghost.current_state.tick_id - 5] # Recent active
            #     if current_expectations:
            #         print(f"DEBUG: Active Expectations after Tick {ghost.current_tick_id}:")
            #         for exp in current_expectations:
            #             print(f"  - ID {exp.id} (Tick {exp.tick_id}, Ful: {exp.fulfilment:.2f}, Urg: {exp.urgency:.2f}): '{exp.content[:60]}...'")

        #except EOFError: break
        #except KeyboardInterrupt: break
        #except Exception as e:
        #     logging.error(f"Error in main loop: {e}", exc_info=True)
        #     print("An unexpected error occurred. Please check logs.")

    # Save state on exit?
    # ghost.save_state("ghost_state_v4.json")
    print("\n--- Session Ended ---")

if __name__ == '__main__':
    run(False, False, test_mode=False)
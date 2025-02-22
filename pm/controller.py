import gc
import json
import logging
import os
import random
import string
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Tuple, Type, Dict

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

from json_repair import repair_json
from llama_cpp import LlamaGrammar
from pydantic import BaseModel
from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from llama_cpp import Llama
from sqlmodel import Session, create_engine
from jinja2 import Environment, BaseLoader

from pm.character import companion_name, user_name, embedding_model, database_uri, model_map, sysprompt_addendum, sysprompt
from pm.database.tables import Event
from pm.embedding.token import get_token
from pm.utils.string_utils import remove_strings
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.log_utils import setup_logger
from pm.nlp.nlp_spacy import NlpSpacy

logger = logging.getLogger(__name__)

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


class Controller:
    def __init__(self, test_mode: bool, backtest_messages: bool):
        self.chat_template = None
        self.current_ctx = None
        self.session = None
        setup_logger("main.log")
        self.db_lock = threading.Lock()
        self.model_path = None
        self.llm: Llama | None = None
        self.test_mode = test_mode
        self.backtest_messages = backtest_messages

        self.cache_user_input: str = ""
        self.cache_emotion: str = ""
        self.cache_tot: str = ""

        self.model = None
        if not self.test_mode:
            self.model = SentenceTransformer(embedding_model, trust_remote_code=True)
            if torch.cuda.is_available():
                device = 'cuda'
                self.model = self.model.to(device)

        self.spacy: NlpSpacy = NlpSpacy()
        self.spacy.init_model()

    def get_conscious_context(self) -> int:
        preset = LlmPreset.Conscious
        ctx = model_map[preset.value]["context"]
        return ctx

    def get_context_sys_prompt(self) -> int:
        return get_token(sysprompt) + get_token(sysprompt_addendum)

    def load_model(self, preset: LlmPreset):
        if preset == LlmPreset.CurrentOne:
            return

        if preset.value in model_map.keys():
            ctx = model_map[preset.value]["context"]
            last_n_tokens_size = model_map[preset.value]["last_n_tokens_size"]
            model_path  = model_map[preset.value]["path"]
            layers = model_map[preset.value]["layers"]
        else:
            raise Exception("Unknown model!")

        if self.model_path != model_path:
            self.llm = None
            time.sleep(1)
            gc.collect()
            time.sleep(1)

            self.llm = Llama(model_path=model_path, n_gpu_layers=layers, n_ctx=ctx, verbose=False, last_n_tokens_size=last_n_tokens_size, flash_attn=True)
            #self.llm.cache = LlamaDiskCache(capacity_bytes=int(42126278829))
            self.current_ctx = ctx
            self.model_path = model_path
            self.chat_template = Environment(loader=BaseLoader).from_string(self.llm.metadata["tokenizer.chat_template"])

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None,
                        discard_thinks: bool = True
                        ) -> (str, List[BaseModel]):
        self.load_model(preset)

        inp_formatted = []
        for msg in inp:
            inp_formatted.append((self.format_str(msg[0]), self.format_str(msg[1])))

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

        inp_formatted = merged_inp_formatted

        if comp_settings is None:
            comp_settings = CommonCompSettings()

        if tools is None:
            tools = []

        if len(tools) > 0:
            comp_settings.tools_json += tools
            inp_formatted[0] = ("system", inp_formatted[0][1] + "\nTool Documentation JSON Schema:\n" + json.dumps(tools[0].model_json_schema()))

        log_filename = f"{str(uuid.uuid4())}_{preset.value}.log"
        log_dir = os.path.dirname(os.path.abspath(__file__)) + "/../logs/"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_filename)
        log_conversation(inp_formatted, log_file_path, 200)

        openai_inp = []
        for msg in inp_formatted:
            openai_inp.append({"role": msg[0], "content": msg[1]})
            
        comp_args = {
            "max_tokens": comp_settings.max_tokens,
            "repeat_penalty": comp_settings.repeat_penalty,
            "temperature": comp_settings.temperature,
            "stop": comp_settings.stop_words
        }

        if preset == LlmPreset.Conscious:
            comp_args["frequency_penalty"] = comp_settings.frequency_penalty
            comp_args["presence_penalty"] = comp_settings.presence_penalty
            comp_args["mirostat_mode"] = 1
            comp_args["mirostat_tau"] = 8
            comp_args["mirostat_eta"] = 0.1

        # remove add generation prompts if last turn is from assistant
        openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test3"}, {"role": "assistant", "content": "test4"}]
        data_remove_ass = self.chat_template.render(
            messages=openai_inp_remove_ass,
            add_generation_prompt=True
        )

        remove_ass_string = data_remove_ass.split("test4")[1]

        # remove messages until it fits in context
        while True:
            rendered_data = self.chat_template.render(
                messages=openai_inp,
                add_generation_prompt=True
            )

            if openai_inp[-1]["role"] == "assistant":
                rendered_data = rendered_data.removesuffix(remove_ass_string)

            rendered_data = remove_strings(rendered_data, "</think>", "<think>")

            tokenized_prompt = self.llm.tokenize(
                rendered_data.encode("utf-8"),
                add_bos=True,
                special=True,
            )

            # start deleting in middle
            if len(tokenized_prompt) > (self.current_ctx - comp_settings.max_tokens) - 16:
                del openai_inp[int(len(openai_inp) / 2)]
            else:
                break

        content = None
        tools = comp_settings.tools_json
        if len(tools) == 0:
            res = self.llm.create_completion(tokenized_prompt, **comp_args)
            finish_reason = res["choices"][0]["finish_reason"]
            content = res["choices"][0]["text"]
            calls = []
            if discard_thinks:
                content = content.split("</think>")[-1]
        else:
            gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([tools[0]])
            grammar = LlamaGrammar(_grammar=gbnf_grammar)
            while True:
                try:
                    res = self.llm.create_chat_completion_openai_v1(openai_inp, grammar=grammar, **comp_args)
                    content = res.choices[0].message.content
                    finish_reason = res.choices[0].finish_reason
                    good_json_string = repair_json(content)
                    calls = [tools[0].model_validate_json(good_json_string)]
                    break
                except Exception as e:
                    print(e)

        content = content.strip()
        inp_formatted.append(("assistant", content))
        log_conversation(inp_formatted, log_file_path + f".completion_{finish_reason}.log", 200)
        return content, calls

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        discard_thinks: bool = True) -> str:
        if self.test_mode:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(128))
        self.load_model(preset)
        content, models = self.completion_tool(preset, inp, comp_settings, discard_thinks=discard_thinks)

        return content.strip()

    def start(self):
        pass

    def get_user_input(self, override: str = None) -> Event:
        if override is None:
            inp = input(">")
        else:
            inp = override.strip()

        if inp.strip() == "":
            print("no input")
            exit(1)

        event = Event(
            source=f"{user_name}",
            content=inp,
            embedding=self.get_embedding(inp),
            token=get_token(inp),
            timestamp=datetime.now(),
            interlocus=1
        )
        return event

    @staticmethod
    def format_str(template: str, extra: Dict = None) -> str:
        if extra is None:
            extra = {}

        values = {
            "companion_name": companion_name,
            "user_name": user_name
        }
        values.update(extra)

        for i in range(10):
            for k, v in values.items():
                template = template.replace("{" + k + "}", v)

        return template

    def get_embedding(self, text: str) -> List[float]:
        if self.model is None:
            return np.random.random((768,)).tolist()
        embeddings = self.model.encode([text], show_progress_bar=False)
        embeddings = torch.tensor(embeddings).to('cpu').tolist()[0]
        return embeddings

    @staticmethod
    def get_timestamp() -> datetime:
        return datetime.now()

    def init_db(self):
        self.session = Session(create_engine(database_uri))

    def get_session(self) -> Session:
        return self.session

    def rollback_db(self):
        self.session.rollback()

    def commit_db(self):
        self.session.commit()

controller = Controller(False, False)
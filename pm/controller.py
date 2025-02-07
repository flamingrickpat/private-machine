import gc
import json
import logging
import os
import random
import string
import time
import uuid
from datetime import datetime
from typing import Tuple, Type, Dict
import threading

from json_repair import repair_json
from lancedb import LanceDBConnection
from llama_cpp import LlamaGrammar
from pydantic import BaseModel
from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from llama_cpp import Llama
from sqlmodel import Field, Session, SQLModel, create_engine, select, col
from sqlalchemy import Column, text, INTEGER, func

from pm.character import companion_name, user_name, MODEL_FAST, MODEL_GOOD, MODEL_BEST, MODEL_FAST_LAYERS, MODEL_DEFAULT, MODEL_DEFAULT_LAYERS, MODEL_GOOD_LAYERS, MODEL_BEST_LAYERS, \
    EMB_MODEL, CONTEXT_SIZE, database_uri
from pm.config.config import read_config_file
from pm.database.tables import Event
from pm.embedding.token import get_token
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.log_utils import setup_logger

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
        self.session = None
        #self.config = read_config_file("config.json")
        setup_logger("main.log")
        self.db_lock = threading.Lock()
        self.model_path = None
        self.llm = None
        self.test_mode = test_mode
        self.backtest_messages = backtest_messages

        self.model = None
        if not self.test_mode:
            self.model = SentenceTransformer(EMB_MODEL, trust_remote_code=True)
            if torch.cuda.is_available():
                device = 'cuda'
                self.model = self.model.to(device)


    def load_model(self, preset: LlmPreset):
        ctx = CONTEXT_SIZE
        last_n_tokens_size = 64
        if preset == LlmPreset.Fast:
            model_path = MODEL_FAST
            layers = MODEL_FAST_LAYERS
        elif preset == LlmPreset.Default:
            model_path = MODEL_DEFAULT
            layers = MODEL_DEFAULT_LAYERS
        elif preset == LlmPreset.Good:
            model_path = MODEL_GOOD
            layers = MODEL_GOOD_LAYERS
            last_n_tokens_size = 1024
        elif preset == LlmPreset.Best:
            model_path = MODEL_BEST
            layers = MODEL_BEST_LAYERS
            last_n_tokens_size = 1024
        else:
            raise Exception("Unknown model!")

        if self.model_path != model_path:
            self.llm = None
            time.sleep(1)
            gc.collect()
            time.sleep(1)

            self.llm = Llama(model_path=model_path, n_gpu_layers=layers, n_ctx=ctx, verbose=False, last_n_tokens_size=last_n_tokens_size)
            self.model_path = model_path
        pass

    @property
    def db(self):
        db = LanceDBConnection(self.config.db_path)
        return db

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None
                        ) -> (str, List[BaseModel]):
        self.load_model(preset)

        inp_formatted = []
        for msg in inp:
            inp_formatted.append((self.format_str(msg[0]), self.format_str(msg[1])))

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
        }

        if preset == LlmPreset.Best or preset == LlmPreset.Good:
            comp_args["frequency_penalty"] = comp_settings.frequency_penalty
            comp_args["presence_penalty"] = comp_settings.presence_penalty
            comp_args["mirostat_mode"] = 1
            comp_args["mirostat_tau"] = 8
            comp_args["mirostat_eta"] = 0.1

        tools = comp_settings.tools_json
        if len(tools) == 0:
            res = self.llm.create_chat_completion_openai_v1(openai_inp, **comp_args)
            content = res.choices[0].message.content
            return content, []
        else:
            gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([tools[0]])
            grammar = LlamaGrammar(_grammar=gbnf_grammar)
            while True:
                try:
                    res = self.llm.create_chat_completion_openai_v1(openai_inp, grammar=grammar, **comp_args)
                    content = res.choices[0].message.content
                    good_json_string = repair_json(content)
                    calls = [tools[0].model_validate_json(good_json_string)]
                    return content, calls
                except Exception as e:
                    print(e)
                    pass

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None) -> str:
        if self.test_mode:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(128))
        self.load_model(preset)
        content, models = self.completion_tool(preset, inp, comp_settings)
        return content

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

    def get_session(self):
        return self.session

    def rollback_db(self):
        self.session.rollback()

    def commit_db(self):
        self.session.commit()

controller = Controller(False, False)
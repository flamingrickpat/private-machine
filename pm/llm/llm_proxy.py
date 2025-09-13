import copy
import copy
import os
import queue
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Any
from typing import List
from typing import (
    Tuple,
)
from typing import Type

from pydantic import BaseModel

from pm.data_structures import KnoxelBase
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_manager_llama import LlmManagerLLama
from pm.utils.exec_utils import get_call_stack_str


# simple Task object that carries the prompt, a result-queue, and (implicitly) its priority.
@dataclass(order=True)
class LlmTask:
    priority: int
    fname: str
    args: Dict[str, Any]
    result_queue: queue.Queue = field(compare=False)


# central "LLM worker" thread
def llama_worker(model_map: Dict[str, Any], task_queue: Queue[LlmTask]):
    llama = LlmManagerLLama(model_map)
    while True:
        task: LlmTask = task_queue.get()
        if task is None:
            # sentinel to shut down cleanly
            break

        if task.fname == "completion_agentic":
            ass, story = llama.completion_agentic(**task.args)
            task.result_queue.put((ass, story))
        elif task.fname == "completion_text":
            comp = llama.completion_text(**task.args)
            task.result_queue.put(comp)
        elif task.fname == "completion_tool":
            comp, tools = llama.completion_tool(**task.args)
            task.result_queue.put((comp, tools))
        elif task.fname == "get_embedding":
            emb = llama.get_embedding(**task.args)
            task.result_queue.put(emb)

        task_queue.task_done()


class LlmManagerProxy:
    def __init__(self, task_queue: Queue[LlmTask], priority: int, log_path: str):
        self.task_queue = task_queue
        self.priority = priority
        self.log_path = log_path

    def get_embedding(self, text: str | KnoxelBase) -> List[float]:
        if isinstance(text, KnoxelBase):
            text = text.content

        args = copy.copy(locals())
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self.task_queue.put(LlmTask(priority=self.priority, fname="get_embedding", args=args, result_queue=result_q))
        emb = result_q.get()
        return emb

    def completion_agentic(self,
                           preset: LlmPreset,
                           url: str,
                           inp: List[Tuple[str, str]],
                           comp_settings: CommonCompSettings | None = None,
                           discard_thinks: bool = True
                           ) -> Tuple[str, str]:
        args = copy.copy(locals())
        log_filename = f"{get_call_stack_str(comp_settings.caller_id if comp_settings else None)}_{uuid.uuid4()}.log"
        os.makedirs(self.log_path, exist_ok=True)
        log_file_path = os.path.join(self.log_path, log_filename)

        args["log_file_path"] = log_file_path
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self.task_queue.put(LlmTask(priority=self.priority, fname="completion_agentic", args=args, result_queue=result_q))
        ass, story = result_q.get()
        return ass, story

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        discard_thinks: bool = True) -> str:
        args = copy.copy(locals())
        log_filename = f"{get_call_stack_str(comp_settings.caller_id if comp_settings else None)}_{uuid.uuid4()}.log"
        os.makedirs(self.log_path, exist_ok=True)
        log_file_path = os.path.join(self.log_path, log_filename)

        args["log_file_path"] = log_file_path
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self.task_queue.put(LlmTask(priority=self.priority, fname="completion_text", args=args, result_queue=result_q))
        comp = result_q.get()
        return comp

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None,
                        discard_thinks: bool = True
                        ) -> (str, List[BaseModel]):
        args = copy.copy(locals())
        log_filename = f"{get_call_stack_str(comp_settings.caller_id if comp_settings else None)}_{uuid.uuid4()}.log"
        os.makedirs(self.log_path, exist_ok=True)
        log_file_path = os.path.join(self.log_path, log_filename)

        args["log_file_path"] = log_file_path
        del args["self"]
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self.task_queue.put(LlmTask(priority=self.priority, fname="completion_tool", args=args, result_queue=result_q))
        comp, tools = result_q.get()
        return comp, tools

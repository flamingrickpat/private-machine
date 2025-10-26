import copy
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Dict, List, Tuple, Type, Optional
from multiprocessing import Manager

from pydantic import BaseModel

from pm.data_structures import KnoxelBase
from pm.llm.emb_manager_transformer import EmbManagerTransformer
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_manager_llama import LlmManagerLLama
from pm.utils.exec_utils import get_call_stack_str

# ===========================================
TIMEOUT = 0.05
USE_MULTIPROCESSING = False
# ===========================================

if USE_MULTIPROCESSING:
    from multiprocessing import Process as WorkerType, Queue as WorkerQueue
else:
    from threading import Thread as WorkerType
    from queue import Queue as WorkerQueue


@dataclass(order=True)
class LlmTask:
    priority: int
    fname: str = field(compare=False)
    args: Dict[str, Any] = field(compare=False)
    result_queue: Any = field(compare=False)


# =====================================================
# Worker definitions
# =====================================================

def emb_worker(model_map: Dict[str, Any], task_queue: WorkerQueue):
    emb_model = EmbManagerTransformer(model_map)
    while True:
        try:
            task: LlmTask = task_queue.get(timeout=TIMEOUT)
            if task is None:
                break
            if task.fname == "get_embedding":
                emb = emb_model.get_embedding(**task.args)
                task.result_queue.put(emb)
            task_queue.task_done() if hasattr(task_queue, "task_done") else None
        except queue.Empty:
            time.sleep(TIMEOUT)


def llama_worker(model_map: Dict[str, Any], task_queue: WorkerQueue):
    llama = LlmManagerLLama(model_map)
    while True:
        try:
            task: LlmTask = task_queue.get(timeout=TIMEOUT)
            if task is None:
                break

            fname = task.fname
            if fname == "completion_agentic":
                ass, story = llama.completion_agentic(**task.args)
                task.result_queue.put((ass, story))
            elif fname == "completion_text":
                comp = llama.completion_text(**task.args)
                task.result_queue.put(comp)
            elif fname == "completion_tool":
                comp, tools = llama.completion_tool(**task.args)
                task.result_queue.put((comp, tools))

            task_queue.task_done() if hasattr(task_queue, "task_done") else None
        except queue.Empty:
            time.sleep(TIMEOUT)


# =====================================================
# Router worker (always thread)
# =====================================================

def llm_worker(model_map: Dict[str, Any], task_queue: WorkerQueue):
    emb_queue = WorkerQueue()
    emb_process = WorkerType(target=emb_worker, args=(model_map, emb_queue), daemon=True)
    emb_process.name = "emb_worker_process" if USE_MULTIPROCESSING else "emb_worker_thread"
    emb_process.start()

    model_instances = {}
    model_mapping = {}
    model_mapping_reverse = {}
    model_instance_queue_map = {}

    for key, value in model_map.items():
        if isinstance(value, dict):
            model_instances[value["model_key"]] = value
            model_mapping[key] = value["model_key"]
            model_mapping_reverse[value["model_key"]] = key

    for key, value in model_instances.items():
        q = WorkerQueue()
        model_instance_queue_map[key] = q
        mm = {LlmPreset.Internal.value: value}
        worker = WorkerType(target=llama_worker, args=(mm, q), daemon=True)
        worker.name = f"llama_worker_{key}"
        worker.start()

    # Router loop (thread)
    while True:
        try:
            task: LlmTask = task_queue.get(timeout=TIMEOUT)
            if task is None:
                break

            if task.fname == "get_embedding":
                emb_queue.put(task)
            else:
                task_preset = task.args["preset"]
                target_instance = model_mapping[task_preset.value]
                task.args["preset"] = LlmPreset.Internal
                model_instance_queue_map[target_instance].put(task)

            task_queue.task_done() if hasattr(task_queue, "task_done") else None
        except queue.Empty:
            time.sleep(TIMEOUT)


# =====================================================
# Proxy API
# =====================================================

class LlmManagerProxy:
    def __init__(self, task_queue: WorkerQueue, priority: int, log_path: str):
        self.task_queue = task_queue
        self.priority = priority
        self.log_path = log_path
        self.manager = Manager() if USE_MULTIPROCESSING else None

    def get_queue(self):
        if self.manager:
            return self.manager.Queue()
        else:
            return queue.Queue()

    def get_max_tokens(self, preset: LlmPreset):
        from pm.config_loader import model_map
        return int(model_map[preset.value]["context"])

    def _submit_task(self, fname: str, args: Dict[str, Any], block=True):
        result_q = self.get_queue()
        task = LlmTask(priority=self.priority, fname=fname, args=args, result_queue=result_q)

        #try:
        #    # TEMP DEBUG
        #    import pickle
        #    pickle.dumps(task)  # 🔎 if this fails, you’ll see where
        #except Exception as e:
        #    print("Pickle error in task:", fname, type(e), e)
        #    for k, v in task.args.items():
        #        print("  arg", k, "=", type(v))
        #    raise

        self.task_queue.put(task)
        if not block:
            return None
        while True:
            try:
                return result_q.get(timeout=TIMEOUT)
            except queue.Empty:
                time.sleep(TIMEOUT)

    def get_embedding(self, text: str | KnoxelBase) -> List[float]:
        if isinstance(text, KnoxelBase):
            text = text.content
        args = dict(text=text)
        return self._submit_task("get_embedding", args)

    def completion_agentic(self, preset: LlmPreset, url: str,
                           inp: List[Tuple[str, str]],
                           comp_settings: Optional[CommonCompSettings] = None,
                           discard_thinks: bool = True) -> Tuple[str, str]:
        log_filename = f"{get_call_stack_str(comp_settings.caller_id if comp_settings else None)}_AGENTIC_{uuid.uuid4()}.log"
        os.makedirs(self.log_path, exist_ok=True)
        args = dict(preset=preset, url=url, inp=inp, comp_settings=comp_settings,
                    discard_thinks=discard_thinks,
                    log_file_path=os.path.join(self.log_path, log_filename))
        return self._submit_task("completion_agentic", args)

    def completion_text(self, preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        discard_thinks: bool = True) -> str:
        log_filename = f"{get_call_stack_str(comp_settings.caller_id if comp_settings else None)}_TEXT_{uuid.uuid4()}.log"
        os.makedirs(self.log_path, exist_ok=True)
        args = dict(preset=preset, inp=inp, comp_settings=comp_settings,
                    discard_thinks=discard_thinks,
                    log_file_path=os.path.join(self.log_path, log_filename))
        return self._submit_task("completion_text", args)

    def completion_tool(self, preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        tools: Optional[List[Type[BaseModel]]] = None,
                        discard_thinks: bool = True) -> Tuple[str, List[BaseModel]]:
        log_filename = f"{get_call_stack_str(comp_settings.caller_id if comp_settings else None)}_TOOL_{uuid.uuid4()}.log"
        os.makedirs(self.log_path, exist_ok=True)
        args = dict(preset=preset, inp=inp, comp_settings=comp_settings, tools=tools,
                    discard_thinks=discard_thinks,
                    log_file_path=os.path.join(self.log_path, log_filename))
        return self._submit_task("completion_tool", args)

def start_llm_thread() -> LlmManagerProxy:
    from pm.config_loader import log_path, model_map
    task_queue: queue.PriorityQueue[LlmTask] = queue.PriorityQueue()

    worker_thread = threading.Thread(target=llm_worker, args=(model_map, task_queue), daemon=True)
    worker_thread.start()

    main_llm = LlmManagerProxy(task_queue, priority=0, log_path=log_path)

    return main_llm
